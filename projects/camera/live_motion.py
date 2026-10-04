"""Camera motion test viewer: fixed-background ROI, segmented image motion and recording.

The green arrow is image displacement from the previous frame to this frame.
The trajectory is accumulated image displacement in pixels, never a robot pose.
Run ``python projects/camera/live_motion.py --demo`` without a camera.
"""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from projects.camera.motion_session import MotionSession
from projects.camera.motion_recording import MotionRecorder

WINDOW = 'Scene motion test - image displacement'
CANVAS_SIZE = (1280, 800)
ARROW_DISPLAY_SCALE = 8.
VIEW_RECT = (20, 115, 860, 540)
BUTTONS = {
    'full_frame': (20, 61, 150, 34),
    'reset': (184, 61, 115, 34),
    'pause': (313, 61, 145, 34),
    'quit': (472, 61, 100, 34),
}


@dataclass(frozen=True)
class ViewTransform:
    """Exact mapping using the *rounded* image dimensions drawn in a letterbox."""

    image_width: int
    image_height: int
    left: int
    top: int
    display_width: int
    display_height: int

    @classmethod
    def fit(cls, frame_shape_hw, rect_xywh):
        height, width = map(int, frame_shape_hw[:2])
        x, y, view_width, view_height = map(int, rect_xywh)
        if min(height, width, view_width, view_height) <= 0:
            raise ValueError('Image and viewport dimensions must be positive')
        scale = min(view_width / width, view_height / height)
        dw = min(view_width, max(1, round(width * scale)))
        dh = min(view_height, max(1, round(height * scale)))
        return cls(width, height, x + (view_width - dw) // 2,
                   y + (view_height - dh) // 2, dw, dh)

    def image_to_canvas(self, x, y):
        return (self.left + float(x) * self.display_width / self.image_width,
                self.top + float(y) * self.display_height / self.image_height)

    def canvas_to_image(self, x, y, clip=False):
        if not clip and not (self.left <= x < self.left + self.display_width and
                             self.top <= y < self.top + self.display_height):
            return None
        return (float(np.clip((x - self.left) * self.image_width / self.display_width,
                              0, self.image_width)),
                float(np.clip((y - self.top) * self.image_height / self.display_height,
                              0, self.image_height)))


class ROIController:
    """Mouse selection in original-image pixels; letterbox bars cannot start a ROI."""

    def __init__(self, on_action=None, on_roi=None, min_size=96):
        self.on_action = on_action or (lambda action: None)
        self.on_roi = on_roi or (lambda roi: None)
        self.min_size = int(min_size)
        self.transform = None
        self.start = None
        self.end = None
        self.message = ''

    def set_transform(self, transform):
        if self.transform is not None and self.transform != transform:
            self.start = self.end = None
        self.transform = transform

    @property
    def selection(self):
        if self.start is None or self.end is None:
            return None
        x = int(np.floor(min(self.start[0], self.end[0])))
        y = int(np.floor(min(self.start[1], self.end[1])))
        right = int(np.ceil(max(self.start[0], self.end[0])))
        bottom = int(np.ceil(max(self.start[1], self.end[1])))
        return x, y, right - x, bottom - y

    def handle_mouse(self, event, x, y, flags=0, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, (bx, by, bw, bh) in BUTTONS.items():
                if bx <= x < bx + bw and by <= y < by + bh:
                    self.start = self.end = None
                    self.on_action(action)
                    return
            point = self.transform.canvas_to_image(x, y) if self.transform else None
            self.start = self.end = point
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.end = self.transform.canvas_to_image(x, y, clip=True)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            self.end = self.transform.canvas_to_image(x, y, clip=True)
            roi = self.selection
            self.start = self.end = None
            if min(roi[2:]) < self.min_size:
                self.message = f'ROI must be at least {self.min_size} x {self.min_size} source pixels'
            else:
                self.message = ''
                self.on_roi(roi)


def _point(value):
    return tuple(np.rint(np.clip(value, -32760, 32760)).astype(int))


def _text(canvas, text, x, y, color=(220, 220, 220), scale=.51):
    cv2.putText(canvas, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def _number(value, digits=2):
    return '--' if value is None else f'{float(value):.{digits}f}'


def render_motion(frame, record, path, roi=None, paused=False, drag_roi=None,
                  full_ms=None, size=CANVAS_SIZE):
    """Render without modifying inputs or running the estimator.

    Displacements smaller than 2 source pixels use a stationary marker. A rejected
    or paused frame never draws an accepted green arrow or marker.
    """
    if tuple(size) != CANVAS_SIZE:
        raise ValueError(f'The mouse layout uses a fixed canvas of {CANVAS_SIZE}')
    canvas = np.full((size[1], size[0], 3), (24, 24, 24), np.uint8)
    tf = ViewTransform.fit(frame.shape, VIEW_RECT)
    interpolation = cv2.INTER_AREA if tf.display_width < frame.shape[1] else cv2.INTER_LINEAR
    display = cv2.resize(frame, (tf.display_width, tf.display_height), interpolation=interpolation)
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    canvas[tf.top:tf.top + tf.display_height, tf.left:tf.left + tf.display_width] = display
    _text(canvas, 'SCENE MOTION TEST', 20, 34, (245, 245, 245), .78)
    _text(canvas, 'Image pixels; this is not a robot trajectory', 332, 34, (180, 195, 240), .57)
    labels = {'full_frame': 'Full frame [F]', 'reset': 'Reset [R]',
              'pause': 'Resume [Space]' if paused else 'Pause [Space]', 'quit': 'Quit [Q]'}
    for action, (x, y, w, h) in BUTTONS.items():
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (62, 62, 62), -1)
        _text(canvas, labels[action], x + 9, y + 23, scale=.47)
    _text(canvas, 'Drag on image: select a fixed background ROI (>= 96 x 96 px)', 20, 686)
    _text(canvas, 'Green arrow: image shift x8 (display only). Cyan: ROI. Orange: current segment.', 20, 711)
    _text(canvas, 'Static scene: camera translation points opposite. Rotation / depth need geometry.', 20, 736)
    _text(canvas, 'Reject / pause / ROI change / reset breaks the accumulated path.', 20, 761)
    for box, color in ((roi, (255, 210, 30)), (drag_roi, (255, 170, 70))):
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(canvas, _point(tf.image_to_canvas(x, y)),
                          _point(tf.image_to_canvas(x + w, y + h)), color, 2)
    accepted = bool(record.get('success')) and not paused
    if accepted and record.get('shift_xy') is not None:
        dx, dy = record['shift_xy']
        x, y, w, h = roi or (0, 0, frame.shape[1], frame.shape[0])
        origin = tf.image_to_canvas(x + w / 2, y + h / 2)
        destination = tf.image_to_canvas(x + w / 2 + dx * ARROW_DISPLAY_SCALE,
                                         y + h / 2 + dy * ARROW_DISPLAY_SCALE)
        if np.hypot(dx, dy) < 2:
            cv2.circle(canvas, _point(origin), 4, (0, 240, 0), 2, cv2.LINE_AA)
        else:
            visible, start, end = cv2.clipLine(
                (tf.left, tf.top, tf.display_width, tf.display_height),
                _point(origin), _point(destination))
            if visible:
                preview = canvas[tf.top:tf.top + tf.display_height,
                                 tf.left:tf.left + tf.display_width]
                start = (start[0] - tf.left, start[1] - tf.top)
                end = (end[0] - tf.left, end[1] - tf.top)
                cv2.arrowedLine(preview, start, end, (0, 240, 0),
                                2, cv2.LINE_AA, tipLength=.25)
    status = 'PAUSED' if paused else ('ACCEPTED' if accepted else 'NO RELIABLE SHIFT')
    # Keep green reserved exclusively for the accepted marker, including on rejected frames.
    _text(canvas, status, 902, 130, (80, 220, 245) if accepted else (90, 150, 255), .64)
    reason = 'paused' if paused else str(record.get('reason') or 'ok')
    shift = record.get('shift_xy') if accepted else None
    candidate = record.get('candidate_shift_xy')
    accumulated = record.get('accumulated_xy') if accepted else None
    lines = [
        'Reason: ' + reason,
        f'Frame: {record.get("frame_index", "--")}',
        f'Source: {frame.shape[1]} x {frame.shape[0]}',
        ('ROI: full frame' if roi is None else 'ROI: ' + ', '.join(map(str, roi))),
        'dx / dy px: ' + (' / '.join(_number(v) for v in shift) if shift is not None else '--'),
        'Candidate: ' + (' / '.join(_number(v) for v in candidate) if candidate is not None else '--'),
        'Response: ' + _number(record.get('response'), 3),
        'API: ' + _number(record.get('api_ms')) + ' ms',
        'Preprocess + API: ' + _number(record.get('processing_ms')) + ' ms',
        'Last full frame: ' + _number(full_ms) + ' ms',
        'Segment: ' + str(record.get('segment_id') if record.get('segment_id') is not None else '--'),
        'Sum px: ' + (' / '.join(_number(v) for v in accumulated) if accumulated is not None else '--'),
    ]
    if record.get('error_px') is not None:
        lines.append('Synthetic truth error: ' + _number(record['error_px'], 3) + ' px')
    for index, line in enumerate(lines):
        _text(canvas, line[:48], 902, 163 + index * 23, scale=.45)
    tx, ty, tw, th = 906, 493, 350, 220
    cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (50, 50, 50), 1)
    _text(canvas, 'Accumulated IMAGE displacement [px]', 906, 480, scale=.45)
    _text(canvas, '+x right; +y down; current segment only', 906, 738, scale=.43)
    if accepted and len(path):
        points = np.asarray(path, dtype=float).reshape(-1, 2)
        if np.isfinite(points).all():
            lower = np.minimum(points.min(axis=0), -5.)
            upper = np.maximum(points.max(axis=0), 5.)
            scale = min((tw - 36) / (upper[0] - lower[0]), (th - 36) / (upper[1] - lower[1]))
            center = (lower + upper) / 2
            display_points = (points - center) * scale + (tx + tw / 2, ty + th / 2)
            if len(display_points) > 1:
                cv2.polylines(canvas, [np.rint(display_points).astype(np.int32)], False,
                              (30, 175, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, _point(display_points[-1]), 3, (30, 175, 255), -1)
            _text(canvas, f'View scale: {scale:.2f} display px / image px', 912, 707, scale=.40)
    else:
        _text(canvas, 'Waiting for a continuous valid segment', tx + 15, ty + th // 2, scale=.42)
    return canvas, tf


def demo_frames(width=960, height=540, frames=120, fps=30.):
    """Known +/- (4,-3) pixel crops, including a blank/recovery interval.

    Frames 48..52 of each 120-frame cycle have no texture. Those frames and the
    first recovered frame have no measurable reference displacement.
    """
    rng = np.random.default_rng(7319)
    margin = 64
    world = rng.integers(0, 256, (height + 2 * margin, width + 2 * margin, 3), dtype=np.uint8)
    world = cv2.GaussianBlur(world, (0, 0), 1.1)
    for index in range(45):
        center = tuple(map(int, rng.integers((0, 0), (world.shape[1], world.shape[0]))))
        color = tuple(map(int, rng.integers(25, 235, 3)))
        cv2.circle(world, center, int(rng.integers(4, 19)), color, -1)
    cv2.putText(world, 'STATIC WORLD / MOVING VIEW', (margin + 12, margin + 60),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (240, 235, 30), 2, cv2.LINE_AA)
    previous = None
    for index in range(frames):
        phase = index % 48
        distance = phase if phase <= 24 else 48 - phase
        position = (4. * distance - 48., -3. * distance + 36.)
        center = (margin + (width - 1) / 2 - position[0],
                  margin + (height - 1) / 2 - position[1])
        frame = cv2.getRectSubPix(world, (width, height), center)
        blank = 48 <= index % 120 <= 52
        if blank:
            frame.fill(127)
        expected = None if previous is None or blank else (position[0] - previous[0], position[1] - previous[1])
        yield frame, index / fps, expected
        previous = None if blank else position


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--camera', type=int, default=1, help='Camera index (default: 1; try 0 for an integrated webcam)')
    source.add_argument('--video', type=Path, help='Replay a video file at its original resolution')
    source.add_argument('--demo', action='store_true', help='Known synthetic translation; no camera required')
    parser.add_argument('--headless', action='store_true', help='Process and record without opening a window')
    parser.add_argument('--frames', type=int, default=0, help='Processed frame limit; 0 means EOF (demo: 120)')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'output/camera/motion')
    parser.add_argument('--no-video', action='store_true', help='Save JSONL/CSV/summary only')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'), help='Fixed background ROI in source pixels, minimum 96 x 96')
    parser.add_argument('--width', type=int, default=960, help='Requested camera or synthetic width; videos keep their own size')
    parser.add_argument('--height', type=int, default=540, help='Requested camera or synthetic height')
    parser.add_argument('--analysis-width', type=int, default=480, help='Estimator analysis width cap (source-pixel output is unchanged)')
    return parser


def run(args):
    """Process one source, always release it, and finalize the recording on failure."""
    if args.frames < 0:
        raise ValueError('--frames must be nonnegative')
    if min(args.width, args.height) < 96:
        raise ValueError('--width and --height must be at least 96')
    if args.analysis_width < 96:
        raise ValueError('--analysis-width must be at least 96')
    if args.headless and not args.demo and args.video is None and args.frames == 0:
        raise ValueError('A headless camera run needs --frames to set a capture limit')
    if args.roi is not None and (min(args.roi[:2]) < 0 or min(args.roi[2:]) < 96):
        raise ValueError('--roi needs nonnegative X/Y and W/H at least 96')
    capture = recorder = None
    window_opened = False
    session = MotionSession(roi=None if args.roi is None else tuple(args.roi),
                            max_analysis_width=args.analysis_width)
    actions = []
    controller = ROIController(lambda action: actions.append((action, None)),
                               lambda roi: actions.append(('roi', roi)))
    paused = False
    processed = 0
    frame = record = None
    last_full_ms = None
    started = perf_counter()
    failure = None
    summary = None
    declared_frames = None
    try:
        source = 'synthetic' if args.demo else (str(args.video.resolve()) if args.video else f'camera:{args.camera}')
        timestamp_source = ('synthetic_nominal_frame_index' if args.demo else
                            'video_nominal_frame_index' if args.video else 'monotonic_after_read')
        fps = 30.
        recorder = MotionRecorder(args.output, metadata={
            'source': source, 'analysis_width': args.analysis_width,
            'requested_size_wh': [args.width, args.height], 'initial_roi_xywh': args.roi,
            'source_fps': None, 'headless': args.headless,
            'timestamp_source': timestamp_source,
            'meaning': 'previous-to-current image displacement in original source pixels; not robot pose',
            'timing': 'frame_ms includes read, preprocessing, API and render; loop_ms/recorded_frame_ms also include video encoding; JSON/CSV flush and GUI pacing excluded; Last full frame also includes log flush',
            'video_timing': 'AVI has nominal source FPS. Video replay timestamps are frame index / nominal FPS, not original variable-frame-rate PTS. Camera timestamps are monotonic time after read, not sensor exposure timestamps.',
        }, save_video=not args.no_video, fps=fps)
        print(f'Motion test output: {recorder.directory}', flush=True)
        if args.demo:
            iterator = iter(demo_frames(args.width, args.height, args.frames or 120, fps))
        else:
            capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
            if not capture.isOpened():
                raise RuntimeError(f'Could not open source: {source}')
            if args.video is None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            measured_fps = capture.get(cv2.CAP_PROP_FPS)
            fps = float(measured_fps) if np.isfinite(measured_fps) and measured_fps > 0 else 30.
            if args.video is not None:
                count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                if np.isfinite(count) and count > 0:
                    declared_frames = round(count)
        # No video stream exists until append(); set measured FPS before creating writers.
        recorder.configure_source(fps, {'source_fps': fps, 'declared_video_frames': declared_frames})
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            window_opened = True
            cv2.setMouseCallback(WINDOW, controller.handle_mouse)
        while True:
            loop_start = perf_counter()
            quit_requested = False
            for action, value in actions:
                if action == 'quit':
                    quit_requested = True
                elif action == 'pause':
                    paused = not paused
                    session.reset(reason='paused' if paused else 'resumed')
                elif action == 'reset':
                    session.reset(reason='manual_reset')
                elif action == 'full_frame':
                    session.set_roi(None)
                elif action == 'roi':
                    session.set_roi(value)
            actions.clear()
            if quit_requested:
                break
            if not paused:
                if args.frames and processed >= args.frames:
                    break
                read_start = perf_counter()
                if args.demo:
                    try:
                        frame, timestamp_s, expected = next(iterator)
                    except StopIteration:
                        break
                else:
                    ok, frame = capture.read()
                    if not ok:
                        if args.video is not None and processed:
                            if declared_frames is not None and processed < declared_frames:
                                raise RuntimeError(f'Video ended early: decoded {processed} of '
                                                   f'{declared_frames} declared frames')
                            break
                        raise RuntimeError(f'Source stopped yielding frames: {source}')
                    if args.video is not None:
                        # Uniform nominal timestamps avoid codec backends reporting zero or repeated PTS.
                        timestamp_s = processed / fps
                    else:
                        timestamp_s = perf_counter() - started
                    expected = None
                read_ms = (perf_counter() - read_start) * 1000
                record = session.process(frame, timestamp_s=timestamp_s, expected_shift=expected)
                record['timestamp_source'] = timestamp_source
                record['read_ms'] = read_ms
                render_start = perf_counter()
                canvas, transform = render_motion(frame, record, session.path, roi=session.roi,
                                                  drag_roi=controller.selection, full_ms=last_full_ms)
                controller.set_transform(transform)
                record['render_ms'] = (perf_counter() - render_start) * 1000
                record['frame_ms'] = (perf_counter() - loop_start) * 1000
                recorder.append(frame, canvas, record)
                last_full_ms = (perf_counter() - loop_start) * 1000
                processed += 1
            elif frame is not None:
                canvas, transform = render_motion(frame, record, session.path, roi=session.roi,
                                                  paused=True, drag_roi=controller.selection,
                                                  full_ms=last_full_ms)
                controller.set_transform(transform)
            if not args.headless and frame is not None:
                if controller.message:
                    _text(canvas, controller.message, 610, 83, (90, 150, 255), .45)
                cv2.imshow(WINDOW, canvas)
                delay = 20 if paused else max(1, round(1000 / fps - (perf_counter() - loop_start) * 1000))
                key = cv2.waitKey(min(100, delay)) & 0xff
                action = {ord('q'): 'quit', 27: 'quit', ord('f'): 'full_frame',
                          ord('r'): 'reset', ord(' '): 'pause'}.get(key)
                if action:
                    actions.append((action, None))
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except BaseException as error:
        failure = error
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception as error:
                if failure is None:
                    failure = error
        if window_opened:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass
        if recorder is not None:
            summary = recorder.close(error=None if failure is None else f'{type(failure).__name__}: {failure}')
    if failure is not None:
        raise failure
    if summary is not None and (summary.get('status') == 'error' or summary.get('errors')):
        raise RuntimeError('Recording verification failed: ' + '; '.join(map(str, summary.get('errors', []))))
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print('Stopped; partial recording was finalized.', flush=True)
        return 130
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f'Motion test failed: {error}', file=sys.stderr)
        return 1
    print('Finished; JSONL, CSV and summary saved' + (' (videos disabled).' if args.no_video else '; both videos verified.'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
