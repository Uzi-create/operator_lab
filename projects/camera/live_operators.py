"""Live USB camera: original frame beside a selectable custom CPU operator."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from array import array
import argparse
from collections import deque
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from operators import run, canny
from operators.ridge_ops import ridge_response
from operators.vision_ops import dynamic_threshold

OPERATORS = ('gray', 'adaptive', 'clahe', 'canny', 'dynamic', 'ridge', 'open_close', 'mean')


def process_frame(frame, operator):
    """BGR uint8 -> gray uint8; custom kernels receive float32 values in [0,1]."""
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
            frame.ndim != 3 or frame.shape[2] != 3 or not frame.size):
        raise ValueError('Expected a nonempty uint8 BGR camera frame')
    if operator not in OPERATORS:
        raise ValueError('Unknown operator: ' + str(operator))
    gray8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray8.astype(np.float32) / 255.0
    height, width = gray.shape
    if operator == 'gray':
        return gray8
    if operator in ('adaptive', 'clahe', 'mean', 'canny'):
        pixels = array('f')
        pixels.frombytes(gray.tobytes())
        if operator == 'canny':
            result = canny(pixels, width, height, low=.06, high=.18)
        elif operator == 'clahe':
            result = run(pixels, width, height, mode='clahe', radius=32, parameter=2.0)
        elif operator == 'adaptive':
            result = run(pixels, width, height, mode='adaptive', radius=3, parameter=.08)
        else:
            result = run(pixels, width, height, mode='mean', radius=3)
        output = np.frombuffer(result, dtype=np.float32).reshape(height, width)
    elif operator == 'ridge':
        output = ridge_response(gray, backend='native')
    else:
        output = dynamic_threshold(gray, radius=9, offset=.04, backend='native')
        if operator == 'open_close':
            pixels = array('f')
            pixels.frombytes(output.astype(np.float32).tobytes())
            opened = run(pixels, width, height, mode='open', radius=1)
            closed = run(opened, width, height, mode='close', radius=1)
            output = np.frombuffer(closed, dtype=np.float32).reshape(height, width)
    return np.rint(np.clip(output, 0, 1)*255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0, help='Camera index; this Mac USB RGB is 1')
    parser.add_argument('--operator', choices=OPERATORS, default='adaptive')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--max-width', type=int, default=640, help='Downscale processing while keeping aspect ratio')
    parser.add_argument('--frames', type=int, default=0, help='Stop after N frames; 0 means until Q/Esc')
    parser.add_argument('--headless', action='store_true', help='Read and process actual camera frames without a window')
    parser.add_argument('--report', type=Path, help='Save timings/metadata; no camera images are saved')
    args = parser.parse_args()
    if min(args.width, args.height, args.max_width) < 1 or args.camera < 0 or args.frames < 0:
        parser.error('Dimensions must be positive; camera/frames must be nonnegative')
    backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    count, samples = 0, []
    per_operator = {name: [] for name in OPERATORS}
    original_shape = processed_shape = None
    operator = args.operator
    intervals = deque(maxlen=30)
    previous = None
    title = 'Operator Lab | USB Camera'
    started = perf_counter()
    try:
        if not capture.isOpened():
            raise RuntimeError('Camera cannot open. Check the index, macOS Camera permission, and other camera apps.')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not args.headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        print('Keys: 0 gray, 1 adaptive, 2 CLAHE, 3 Canny, 4 dynamic, 5 ridge, 6 open/close, 7 mean; Q/Esc quit', flush=True)
        while args.frames == 0 or count < args.frames:
            ok, frame = capture.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError('Camera opened but no frame arrived (device disconnected or busy).')
            original_shape = list(frame.shape)
            if frame.shape[1] > args.max_width:
                height = max(1, round(frame.shape[0]*args.max_width/frame.shape[1]))
                frame = cv2.resize(frame, (args.max_width, height), interpolation=cv2.INTER_AREA)
            processed_shape = list(frame.shape[:2])
            begin = perf_counter()
            output = process_frame(frame, operator)
            milliseconds = (perf_counter()-begin)*1000
            samples.append(milliseconds)
            per_operator[operator].append(milliseconds)
            now = perf_counter()
            if previous is not None:
                intervals.append(now-previous)
            previous = now
            count += 1
            if count == 1:
                print(f'LIVE: camera={args.camera}, captured={original_shape}, processing={processed_shape}, operator={operator}', flush=True)
            if not args.headless:
                right = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                panel = np.concatenate((frame, right), axis=1)
                fps = len(intervals)/sum(intervals) if intervals else 0
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 64), (20, 20, 20), -1)
                cv2.putText(panel, f'Original | {operator} | processing {milliseconds:.1f} ms | loop {fps:.1f} FPS',
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                cv2.putText(panel, '0 gray  1 adaptive  2 CLAHE  3 Canny  4 dynamic  5 ridge  6 morph  7 mean  Q quit',
                            (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .43, (220, 220, 220), 1)
                cv2.imshow(title, panel)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q'), ord('Q')):
                    break
                if ord('0') <= key <= ord('7'):
                    operator = OPERATORS[key-ord('0')]
                visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
                if count == 1 and visible >= 1:
                    print('PREVIEW: window is visible; click it and press 0-7 to switch operators', flush=True)
                if visible < 1:
                    break
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
    report = {'camera': args.camera, 'frames': count, 'capture_shape': original_shape,
              'process_shape': processed_shape, 'last_operator': operator,
              'elapsed_seconds': perf_counter()-started,
              'processing_median_ms': float(np.median(samples)) if samples else None,
              'per_operator_median_ms': {name: float(np.median(values))
                                         for name, values in per_operator.items() if values},
              'scope': 'Gray conversion + operator API + output conversion, not pure native kernel timing'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
