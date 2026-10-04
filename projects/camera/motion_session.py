"""Consecutive image-motion measurements, with explicit gaps in the trace.

The ROI stays fixed in image coordinates. Accumulated displacement belongs to
one uninterrupted accepted segment, never to a physical camera/robot pose.
"""
from collections import deque
import math
from numbers import Integral
from time import perf_counter

import cv2
import numpy as np

from operators.measurement_ops import estimate_scene_motion


def _json_value(value):
    """Keep diagnostics portable, including rejected nonfinite FFT candidates."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class MotionSession:
    """Keep only adjacent frames; rejection ends the current trace segment."""

    MIN_SIZE = 96
    MAX_PATH_POINTS = 2000

    def __init__(self, roi=None, max_analysis_width=480, max_shift=None):
        if type(max_analysis_width) is not int or not 32 <= max_analysis_width <= 4096:
            raise ValueError('max_analysis_width must be an integer in 32..4096')
        if max_shift is not None:
            limits = np.asarray(max_shift, dtype=np.float64)
            if limits.shape != (2,) or not np.isfinite(limits).all() or (limits < 0).any():
                raise ValueError('max_shift must be finite nonnegative (dx,dy) limits')
            max_shift = tuple(float(v) for v in limits)
        self.max_analysis_width = max_analysis_width
        self.max_shift = max_shift
        self._roi = None
        self._shape = None
        self._previous = None
        self._timestamp = None
        self._pending_reason = 'first_frame'
        self._frame_index = 0
        self._segment_count = 0
        self._active_segment = None
        self._accumulated = np.zeros(2, dtype=np.float64)
        self._path = deque(maxlen=self.MAX_PATH_POINTS)
        self.set_roi(roi)
        self._pending_reason = 'first_frame'

    @property
    def roi(self):
        return self._roi

    @property
    def path(self):
        return list(self._path)

    def _validate_roi(self, roi):
        if roi is None:
            return None
        try:
            values = tuple(roi)
        except TypeError as error:
            raise ValueError('ROI must be (x,y,width,height) or None') from error
        if len(values) != 4 or any(isinstance(v, (bool, np.bool_)) or
                                   not isinstance(v, Integral) for v in values):
            raise ValueError('ROI must contain four integers')
        x, y, w, h = values = tuple(int(v) for v in values)
        if x < 0 or y < 0 or min(w, h) < self.MIN_SIZE:
            raise ValueError('ROI must be nonnegative and at least 96x96')
        if self._shape is not None:
            height, width = self._shape
            if x + w > width or y + h > height:
                raise ValueError('ROI must be entirely inside the frame')
        return values

    def set_roi(self, roi):
        roi = self._validate_roi(roi)
        if roi != self._roi:
            self._roi = roi
            self.reset('roi_changed')

    def _end_segment(self):
        self._active_segment = None
        self._accumulated.fill(0.)
        self._path.clear()

    def reset(self, reason='manual_reset'):
        if not isinstance(reason, str) or not reason:
            raise ValueError('reset reason must be a nonempty string')
        self._previous = None
        self._timestamp = None
        self._pending_reason = reason
        self._end_segment()

    def process(self, frame, timestamp_s=None, expected_shift=None):
        started = perf_counter()
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.ndim != 3 or frame.shape[2] != 3 or
                min(frame.shape[:2]) < self.MIN_SIZE):
            raise ValueError('frame must be uint8 BGR, at least 96x96')
        timestamp = perf_counter() if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('timestamp_s must be finite and nonnegative')
        expected = None
        if expected_shift is not None:
            expected = np.asarray(expected_shift, dtype=np.float64)
            if expected.shape != (2,) or not np.isfinite(expected).all():
                raise ValueError('expected_shift must be finite (dx,dy)')
        shape = frame.shape[:2]
        if self._shape is not None and shape != self._shape:
            self._roi = None
            self.reset('resolution_changed')
        # Validate a constructor-supplied ROI before committing any frame state.
        if self._roi is not None:
            x, y, w, h = self._roi
            if x + w > shape[1] or y + h > shape[0]:
                raise ValueError('ROI must be entirely inside the frame')
        self._shape = shape
        if self._timestamp is not None and timestamp <= self._timestamp:
            self.reset('timestamp_discontinuity')
        roi = self._roi or (0, 0, shape[1], shape[0])
        x, y, w, h = roi
        gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) * (1. / 255.)
        paired = self._previous is not None
        record = dict(frame_index=self._frame_index, frame_shape_hw=list(shape),
                      timestamp_s=timestamp, pair_interval_s=(timestamp - self._timestamp)
                      if paired else None, roi_xywh=list(roi), success=False,
                      reason=self._pending_reason, baseline=not paired,
                      shift_xy=None, candidate_shift_xy=None, response=None,
                      segment_id=None, accumulated_xy=None, api_ms=0.,
                      expected_shift_xy=expected.tolist() if paired and expected is not None else None,
                      error_px=None)
        if paired:
            api_start = perf_counter()
            result = estimate_scene_motion(self._previous, gray,
                                           max_analysis_width=self.max_analysis_width,
                                           max_shift=self.max_shift)
            record['api_ms'] = (perf_counter() - api_start) * 1000.
            shift = result.get('shift_xy')
            valid_shift = (shift is not None and np.shape(shift) == (2,) and
                           np.isfinite(shift).all())
            accepted = bool(result['success']) and valid_shift
            record.update({key: _json_value(value) for key, value in result.items()
                           if key not in ('success', 'shift_xy', 'alignment_matrix')})
            record['success'] = bool(accepted)
            record['candidate_shift_xy'] = _json_value(shift)
            if result['success'] and not valid_shift:
                record['reason'] = 'invalid_result'
            if accepted:
                if self._active_segment is None:
                    self._segment_count += 1
                    self._active_segment = self._segment_count
                    self._path.append((0., 0.))
                shift = np.asarray(shift, dtype=np.float64)
                self._accumulated += shift
                self._path.append(tuple(self._accumulated.tolist()))
                record.update(shift_xy=shift.tolist(), segment_id=self._active_segment,
                              accumulated_xy=self._accumulated.tolist(),
                              alignment_matrix=_json_value(result.get('alignment_matrix')))
                if expected is not None:
                    record['error_px'] = float(np.linalg.norm(shift - expected))
            else:
                self._end_segment()
        self._previous = gray
        self._timestamp = timestamp
        self._pending_reason = None
        self._frame_index += 1
        record['processing_ms'] = (perf_counter() - started) * 1000.
        return record
