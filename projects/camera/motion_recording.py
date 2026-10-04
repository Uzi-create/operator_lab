"""Durable motion-test records and verified MJPG videos.

AVI uses a fixed *nominal* FPS. Per-frame timestamps in ``frames.jsonl`` are
the timing evidence; video playback speed is not measured camera throughput.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
from time import perf_counter
import uuid

import cv2
import numpy as np


BASELINE_REASONS = frozenset({
    'no_previous_frame', 'manual_reset', 'roi_changed', 'frame_size_changed',
    'resumed', 'reset', 'paused',
})
CSV_FIELDS = (
    'frame_index', 'timestamp_s', 'timestamp_source', 'frame_shape_hw', 'roi_xywh', 'success',
    'reason', 'baseline', 'pair_interval_s', 'shift_xy', 'candidate_shift_xy', 'response', 'segment_id',
    'accumulated_xy', 'api_ms', 'processing_ms', 'read_ms', 'capture_ms',
    'render_ms', 'frame_ms', 'loop_ms', 'video_write_ms', 'recorded_frame_ms',
    'expected_shift_xy', 'error_px', 'raw_video', 'overlay_video', 'nonfinite_fields',
)
TIMING_FIELDS = (
    'read_ms', 'capture_ms', 'api_ms', 'processing_ms', 'render_ms',
    'frame_ms', 'loop_ms', 'video_write_ms', 'recorded_frame_ms',
)


def _safe_json(value, issues, path=''):
    """Normalize NumPy values; invalid numeric values are explicit JSON null."""
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist(), issues, path)
    if isinstance(value, np.generic):
        return _safe_json(value.item(), issues, path)
    if isinstance(value, float) and not math.isfinite(value):
        issues.append(path or '<root>')
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json(v, issues, f'{path}.{k}' if path else str(k))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v, issues, f'{path}[{i}]') for i, v in enumerate(value)]
    raise TypeError(f'Unsupported record value at {path}: {type(value).__name__}')


def _atomic_text(path, text):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _stats(values):
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'p95': None, 'max': None}
    data = np.asarray(values, dtype=np.float64)
    return {'count': len(values), 'mean': float(data.mean()),
            'median': float(np.median(data)), 'p95': float(np.percentile(data, 95)),
            'max': float(data.max())}


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _video_image(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError('Video frames must be uint8 NumPy arrays')
    if frame.ndim == 2:
        image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        image = frame
    elif frame.ndim == 3 and frame.shape[2] == 4:
        image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError('Video frames must be grayscale, BGR, or BGRA')
    height, width = image.shape[:2]
    if height < 2 or width < 2:
        raise ValueError('Video frame width and height must be at least 2')
    if height % 2 or width % 2:
        image = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2,
                                   cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(image)


class MotionRecorder:
    """One unique session, streaming JSONL/CSV and independently sized videos.

    ``append`` returns video-write milliseconds. ``close`` always attempts to
    release all resources and returns a summary whose ``errors`` determine
    whether the CLI should exit unsuccessfully. Short videos (<= 300 frames)
    are fully decoded; longer ones are checked at their first and last frame.
    MJPG is lossy; ``raw`` means before overlay, not lossless sensor data.
    """
    FULL_DECODE_LIMIT = 300

    def __init__(self, output_root, metadata=None, save_video=True, fps=30):
        if isinstance(fps, bool) or not _number(fps) or fps <= 0:
            raise ValueError('fps must be a finite positive nominal frame rate')
        metadata_issues = []
        self.metadata = _safe_json(metadata or {}, metadata_issues)
        if not isinstance(self.metadata, dict):
            raise ValueError('metadata must be a mapping')
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
        # mkdir is the collision check; no pre-existing session is overwritten.
        for _ in range(10):
            self.directory = root / f'motion_{stamp}_{uuid.uuid4().hex[:10]}'
            try:
                self.directory.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError('Could not allocate a unique session directory')
        self.save_video = bool(save_video)
        self.fps = float(fps)
        self._started = perf_counter()
        self._started_iso = datetime.now(timezone.utc).isoformat()
        self._closed = False
        self._reports_saved = False
        self._summary = None
        self._errors = []
        self._streams = {}
        self._videos = []
        self._frame_count = self._baseline_count = self._accepted = 0
        self._reasons = Counter()
        self._segments = set()
        self._timings = {name: [] for name in TIMING_FIELDS}
        self._reference_errors = []
        self._first_timestamp = self._last_timestamp = None
        self._timestamp_discontinuities = 0
        self._nonfinite_count = len(metadata_issues)
        self._metadata_issues = metadata_issues
        self._jsonl = self._csv_file = None
        try:
            self._jsonl = (self.directory / 'frames.jsonl').open('x', encoding='utf-8')
            self._csv_file = (self.directory / 'frames.csv').open(
                'x', encoding='utf-8-sig', newline='')
            self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
            self._csv.writeheader()
            self._csv_file.flush()
            self._save_session()
        except Exception:
            for handle in (self._jsonl, self._csv_file):
                if handle is not None:
                    handle.close()
            raise

    def _save_session(self):
        _atomic_text(self.directory / 'session.json', json.dumps({
            'schema_version': 1, 'started_at': self._started_iso,
            'metadata': self.metadata, 'nominal_video_fps': self.fps,
            'save_video': self.save_video,
            'metadata_nonfinite_fields': self._metadata_issues,
            'video_note': 'MJPG is lossy. Fixed AVI FPS is playback timing. '
                          'Check timestamp_source: replay/synthetic timestamps are nominal; '
                          'camera timestamps are read completion times, not exposure times.',
        }, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def configure_source(self, fps, metadata=None):
        """Update the source description before any frames or video streams exist."""
        if self._closed or self._frame_count or self._videos:
            raise RuntimeError('Source configuration must precede the first frame')
        if isinstance(fps, bool) or not _number(fps) or fps <= 0:
            raise ValueError('fps must be a finite positive nominal frame rate')
        issues = []
        values = _safe_json(metadata or {}, issues)
        if not isinstance(values, dict):
            raise ValueError('metadata must be a mapping')
        self.fps = float(fps)
        self.metadata.update(values)
        self._metadata_issues.extend(issues)
        self._nonfinite_count += len(issues)
        self._save_session()

    def _error(self, message):
        message = str(message)
        if message not in self._errors:
            self._errors.append(message)

    def _write_video(self, kind, frame):
        image = _video_image(frame)
        original_shape = tuple(int(x) for x in frame.shape[:2])
        stream = self._streams.get(kind)
        if stream is None or tuple(stream['info']['source_shape_hw']) != original_shape:
            if stream is not None:
                stream['writer'].release()
                stream['writer'] = None
            part = sum(info['kind'] == kind for info in self._videos)
            filename = f'{kind}_{part:03d}.avi'
            height, width = image.shape[:2]
            writer = cv2.VideoWriter(str(self.directory / filename),
                                     cv2.VideoWriter_fourcc(*'MJPG'), self.fps,
                                     (width, height), True)
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f'Cannot open MJPG video writer: {filename}')
            info = {'kind': kind, 'path': filename, 'codec': 'MJPG',
                    'nominal_fps': self.fps, 'source_shape_hw': list(original_shape),
                    'encoded_shape_hw': [height, width],
                    'padding_bottom_right': [height-original_shape[0], width-original_shape[1]],
                    'frames': 0, 'first_record_index': self._frame_count,
                    'last_record_index': self._frame_count, 'validation': None}
            stream = {'writer': writer, 'info': info}
            self._streams[kind] = stream
            self._videos.append(info)
        stream['writer'].write(image)
        stream['info']['frames'] += 1
        stream['info']['last_record_index'] = self._frame_count
        return stream['info']['path']

    def append(self, frame, overlay, record):
        """Append one frame; caller must close in ``finally`` or use ``with``."""
        if self._closed:
            raise RuntimeError('Cannot append to a closed motion session')
        issues = []
        try:
            row = _safe_json(dict(record), issues)
            row.setdefault('frame_index', self._frame_count)
            row.setdefault('frame_shape_hw', list(frame.shape[:2]))
            # Rejected estimates may have a diagnostic candidate, never an accepted shift.
            if not row.get('success', False):
                row['shift_xy'] = None
            write_start = perf_counter()
            if self.save_video:
                row['raw_video'] = self._write_video('raw', frame)
                row['overlay_video'] = self._write_video('overlay', overlay)
            else:
                row['raw_video'] = row['overlay_video'] = None
            video_ms = (perf_counter() - write_start) * 1000 if self.save_video else 0.
            row['video_write_ms'] = video_ms
            if _number(row.get('frame_ms')):
                row['recorded_frame_ms'] = row['frame_ms'] + video_ms
                row['loop_ms'] = row['recorded_frame_ms']
            if issues:
                row['nonfinite_fields'] = issues
            self._jsonl.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            csv_row = {key: row.get(key) for key in CSV_FIELDS}
            for key, value in csv_row.items():
                if isinstance(value, (list, dict)):
                    csv_row[key] = json.dumps(value, ensure_ascii=False, allow_nan=False)
            self._csv.writerow(csv_row)
            self._jsonl.flush()
            self._csv_file.flush()
            self._observe(row, issues)
            return video_ms
        except Exception as exc:
            self._error(f'Frame {self._frame_count} recording failed: {type(exc).__name__}: {exc}')
            raise

    def _observe(self, row, issues):
        self._frame_count += 1
        self._nonfinite_count += len(issues)
        reason = str(row.get('reason') or ('accepted' if row.get('success') else 'unspecified'))
        self._reasons[reason] += 1
        baseline = row.get('baseline') is True or reason in BASELINE_REASONS
        if baseline:
            self._baseline_count += 1
        elif row.get('success'):
            self._accepted += 1
        if row.get('segment_id') is not None:
            self._segments.add(str(row['segment_id']))
        timestamp = row.get('timestamp_s')
        if _number(timestamp):
            if self._first_timestamp is None:
                self._first_timestamp = timestamp
            if self._last_timestamp is not None and timestamp <= self._last_timestamp:
                self._timestamp_discontinuities += 1
            self._last_timestamp = timestamp
        for key in TIMING_FIELDS:
            value = row.get(key)
            if key == 'api_ms' and baseline:
                continue
            if _number(value) and value >= 0:
                self._timings[key].append(value)
        error = row.get('error_px')
        if row.get('success') and _number(error) and error >= 0:
            self._reference_errors.append(error)

    def _validate_video(self, info):
        capture = None
        try:
            capture = cv2.VideoCapture(str(self.directory / info['path']))
            if not capture.isOpened():
                raise RuntimeError('saved video cannot be opened')
            expected = info['frames']
            declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            decoded = 0

            def read_frame():
                ok, image = capture.read()
                if not ok or image is None:
                    return False
                if list(image.shape[:2]) != info['encoded_shape_hw']:
                    raise RuntimeError('decoded video dimensions differ from recorded dimensions')
                return True

            if expected <= self.FULL_DECODE_LIMIT:
                while read_frame():
                    decoded += 1
                if decoded != expected:
                    raise RuntimeError(f'expected {expected} frames, decoded {decoded}')
                mode = 'all_frames'
            else:
                if declared != expected:
                    raise RuntimeError(f'expected {expected} frames, container declares {declared}')
                if not read_frame():
                    raise RuntimeError('first frame cannot be decoded')
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, expected - 1) or not read_frame():
                    raise RuntimeError('last frame cannot be decoded')
                decoded = 2
                mode = 'first_and_last_frame'
            info['validation'] = {'status': 'passed', 'mode': mode,
                                  'decoded_frames': decoded, 'container_frames': declared}
        except Exception as exc:
            info['validation'] = {'status': 'failed', 'error': str(exc)}
            self._error(f"Video validation failed ({info['path']}): {exc}")
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    self._error(f"Video validation release failed ({info['path']}): {exc}")

    def close(self, error=None):
        """Release, verify, and save reports. Inspect summary['errors'] at CLI exit."""
        if self._closed:
            if not self._reports_saved:
                self._save_reports()
            return self._summary
        elapsed = perf_counter() - self._started
        if error is not None:
            self._error(str(error))
        for stream in self._streams.values():
            writer = stream.get('writer')
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:
                    self._error(f'Video release failed: {exc}')
                stream['writer'] = None
        for handle in (self._jsonl, self._csv_file):
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:
                    self._error(f'Record file close failed: {exc}')
        for info in self._videos:
            self._validate_video(info)
        evaluated = self._frame_count - self._baseline_count
        source = self.metadata.get('source')
        synthetic = (source == 'synthetic' or self.metadata.get('synthetic') is True
                     or isinstance(source, dict) and source.get('kind') == 'synthetic')
        duration = (self._last_timestamp - self._first_timestamp
                    if self._first_timestamp is not None else None)
        summary = {
            'schema_version': 1, 'status': 'error' if self._errors else 'ok',
            'started_at': self._started_iso,
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'metadata': self.metadata, 'frames': self._frame_count,
            'baseline_frames': self._baseline_count, 'evaluated_frames': evaluated,
            'accepted_frames': self._accepted, 'rejected_frames': evaluated - self._accepted,
            'acceptance_rate': self._accepted / evaluated if evaluated else None,
            'reason_counts': dict(self._reasons), 'segments': len(self._segments),
            'timestamp_span_s': duration, 'recording_wall_s': elapsed,
            'timestamp_discontinuities': self._timestamp_discontinuities,
            'source_timeline_frame_rate_hz': ((self._frame_count - 1) / duration
                if duration is not None and duration > 0 and self._frame_count > 1
                and not self._timestamp_discontinuities else None),
            'timing_ms': {key: _stats(values) for key, values in self._timings.items()},
            'synthetic_error_px': _stats(self._reference_errors) if synthetic else None,
            'nonfinite_values_replaced_with_null': self._nonfinite_count,
            'metadata_nonfinite_fields': self._metadata_issues,
            'nominal_video_fps': self.fps, 'videos': self._videos, 'errors': list(self._errors),
            'notes': [
                'Acceptance rate excludes baseline/reset frames and is not motion accuracy.',
                'Response is a correlation diagnostic, not a probability.',
                'AVI playback uses fixed nominal FPS; source timeline rate is not processing throughput.',
                'Check timestamp_source: synthetic/video timelines are nominal; camera times are read completion times.',
                'API timing excludes capture, drawing, recording, and display; it is not end-to-end FPS.',
                'loop_ms / recorded_frame_ms are frame_ms + video_write_ms; file flushing and GUI waiting are excluded.',
                'MJPG raw video is before overlay and lossy; odd dimensions are padded on bottom/right.',
                'Accumulated pixel shifts are segmented image motion, not a metric robot trajectory.',
            ],
        }
        self._summary = summary
        self._closed = True
        self._save_reports()
        return summary

    def _save_reports(self):
        # Retrying a failed report write must not release/validate video twice.
        _atomic_text(self.directory / 'summary.json', json.dumps(
            self._summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        _atomic_text(self.directory / 'RESULTS.md', self._markdown(self._summary))
        self._reports_saved = True

    @staticmethod
    def _markdown(summary):
        def display(value):
            return 'N/A' if value is None else f'{value:.3f}'
        lines = ['# Motion test results', '', f"Status: **{summary['status']}**", '',
                 f"Recorded frames: {summary['frames']}; baseline frames: {summary['baseline_frames']}; "
                 f"evaluated pairs: {summary['evaluated_frames']}.", '',
                 f"Accepted: {summary['accepted_frames']}; rejected: {summary['rejected_frames']}; "
                 f"acceptance rate: {display(summary['acceptance_rate'])} (not accuracy).", '',
                 '| Timing (ms) | Count | Mean | Median | p95 | Maximum |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, stats in summary['timing_ms'].items():
            lines.append(f"| {name} | {stats['count']} | " + ' | '.join(
                display(stats[key]) for key in ('mean', 'median', 'p95', 'max')) + ' |')
        if summary['synthetic_error_px'] is not None:
            values = summary['synthetic_error_px']
            lines += ['', f"Synthetic accepted-pair reference error: median {display(values['median'])} px; "
                      f"p95 {display(values['p95'])} px; maximum {display(values['max'])} px."]
        else:
            lines += ['', 'No synthetic ground truth: field accuracy is not measured.']
        lines += ['', '## Outcomes', '']
        lines += [f'- {name}: {count}' for name, count in summary['reason_counts'].items()]
        lines += ['', '## Videos', '']
        for info in summary['videos']:
            validation = info['validation'] or {}
            lines.append(f"- [{info['path']}]({info['path']}): {info['frames']} frames; "
                         f"source {info['source_shape_hw']}, encoded {info['encoded_shape_hw']}; "
                         f"validation {validation.get('status')} ({validation.get('mode', 'failed')}).")
        if not summary['videos']:
            lines.append('Video saving disabled or no frames received.')
        lines += ['', '## Interpretation', '']
        lines += [f'- {note}' for note in summary['notes']]
        if summary['errors']:
            lines += ['', '## Errors', ''] + [f'- {error}' for error in summary['errors']]
        return '\n'.join(lines) + '\n'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(error=exc)
        return False
