"""USB 相机清晰度对照：在同一目标、固定光照下记录不同距离的分数。

分数是局部纹理高频能量，不是焦距、对焦距离或画质的绝对测量。
物体大小、纹理、噪声、曝光和运动也影响分数，因此不能只看最大值。
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import deque
import json
from pathlib import Path
import sys
from time import time_ns

import cv2
import numpy as np


def measure(frame):
    """只读相机原图中心区域，不做增强/锐化/算子平滑。"""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 96:
        raise ValueError('Expected uint8 BGR frame, dimensions >=96')
    h, w = frame.shape[:2]
    rw, rh = min(640, w), min(480, h)
    x, y = (w-rw)//2, (h-rh)//2
    crop = frame[y:y+rh, x:x+rw]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Laplacian 方差：高频变化越多通常越高，噪声同样会抬高它。
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = float(np.mean(gx*gx+gy*gy))
    return crop, {'laplacian': laplacian, 'gradient_energy': gradient,
                  'mean_brightness': float(gray.mean()),
                  'clipped_fraction': float(np.mean((gray <= 2) | (gray >= 253))),
                  'roi_xywh': [x, y, rw, rh]}


def draw(frame, crop, stats, records, count):
    canvas = np.full((800, 1100, 3), 24, np.uint8)
    # ROI 不缩小，左侧直接复制原始像素，方便观察文字是否真的清楚。
    h, w = crop.shape[:2]
    canvas[75:75+h, 10:10+w] = crop
    scale = min(420/frame.shape[1], 300/frame.shape[0])
    overview = cv2.resize(frame, (round(frame.shape[1]*scale), round(frame.shape[0]*scale)))
    # 概览缩放不参与评分；框表示左侧显示的区域。
    x, y, rw, rh = stats['roi_xywh']
    cv2.rectangle(overview, (round(x*scale), round(y*scale)), (round((x+rw)*scale), round((y+rh)*scale)), (0, 255, 255), 1)
    canvas[75:75+overview.shape[0], 670:670+overview.shape[1]] = overview
    texts = [(10, 30, 'Focus comparison | Left: original pixels | Right: whole scene'),
             (10, 59, f'Captured {frame.shape[1]}x{frame.shape[0]} | samples: {count}'),
             (10, 590, f'Laplacian median: {stats["laplacian"]:.1f}'),
             (10, 622, f'Gradient energy median: {stats["gradient_energy"]:.1f}'),
             (10, 654, f'Brightness: {stats["mean_brightness"]:.1f} | clipped: {stats["clipped_fraction"]*100:.1f}%'),
             (10, 695, 'B: record current position + PNG | C: clear current history | Q: quit'),
             (10, 726, 'Same printed target, steady camera, same light; wait 1 second after moving.'),
             (10, 757, 'Scores depend on texture/scale/noise too. No universal sharpness threshold.')]
    for i, record in enumerate(records[-8:]):
        texts.append((675, 405+i*30, f'#{record["sample"]}: L={record["laplacian"]:.1f} G={record["gradient_energy"]:.1f}'))
    for x, y, text in texts:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/camera/focus')
    args = parser.parse_args()
    if args.camera < 0 or args.frames < 0 or min(args.width, args.height) < 96: parser.error('Invalid camera, dimensions or frame count')
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
    history, records, count = deque(maxlen=15), [], 0
    title = 'Operator Lab | Focus comparison'
    session = str(time_ns())
    last_capture_shape, last_stats = None, None
    try:
        if not cap.isOpened(): raise RuntimeError('Cannot open camera; close other preview windows')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not args.headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 1100, 800)
        while not args.frames or count < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None: raise RuntimeError('No camera frame received')
            last_capture_shape = list(frame.shape)
            crop, raw = measure(frame)
            history.append(raw)
            stats = dict(raw)
            for key in ('laplacian', 'gradient_energy', 'mean_brightness', 'clipped_fraction'):
                stats[key] = float(np.median([s[key] for s in history]))
            last_stats, count = stats, count+1
            if count == 1 or args.headless:
                print(json.dumps({'frame': count, 'capture_shape': last_capture_shape, 'stats': stats}), flush=True)
            if args.headless: continue
            cv2.imshow(title, draw(frame, crop, stats, records, count))
            key = cv2.waitKey(1) & 255
            if count == 1: print('PREVIEW: visible='+str(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)), flush=True)
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1: break
            if key in (ord('c'), ord('C')):
                # 清空当前评分历史，但保留已有样本与照片，以便继续比较。
                history.clear()
            if key in (ord('b'), ord('B')):
                # B 前自动清空旧滑动分数没有意义，用户移动后应等待统计窗口更新。
                args.output.mkdir(parents=True, exist_ok=True)
                path = args.output/f'{session}_sample_{len(records)+1}.png'
                if not cv2.imwrite(str(path), frame): raise IOError('Cannot save original frame')
                record = dict(stats, sample=len(records)+1, image=str(path), history_frames=len(history), frame_metrics=raw)
                records.append(record)
                print('RECORDED: '+json.dumps(record), flush=True)
                history.clear()
    finally:
        cap.release()
        if not args.headless: cv2.destroyAllWindows()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'session': session, 'camera': args.camera, 'capture_shape': last_capture_shape,
              'frames': count, 'last_stats': last_stats, 'records': records,
              'scope': 'Relative local texture sharpness; median of up to 15 frames. Not focal length, working distance, focus motor position, or calibrated image quality.'}
    path = args.output/f'{session}_report.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print('REPORT: '+str(path), flush=True)


if __name__ == '__main__': main()
