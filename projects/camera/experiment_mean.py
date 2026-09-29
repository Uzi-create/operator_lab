"""实验 1：iPhone 同一帧，比较优化均值与朴素均值的结果和 API 耗时。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from array import array
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from operators import run
from projects.camera.live_all_operators import image_view


def prepare(frame):
    # 相机 BGR uint8 -> 灰度 float32 [0,1] -> C++ 接口使用的连续 float 缓冲区。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    pixels = array('f')
    pixels.frombytes(gray.tobytes())
    return gray, pixels


def compare(pixels, width, height, radius, reverse=False):
    """两种实现必须使用完全相同的输入；返回结果、耗时和像素误差。"""
    outputs, timings = {}, {}
    order = ('mean_naive', 'mean') if reverse else ('mean', 'mean_naive')
    for name in order:
        start = perf_counter()
        # 这是调用核心：只改变 mode 名称，像素、尺寸、窗口半径完全相同。
        values = run(pixels, width, height, mode=name, radius=radius)
        timings[name] = (perf_counter()-start)*1000
        outputs[name] = np.frombuffer(values, np.float32).reshape(height, width)
    difference = np.abs(outputs['mean']-outputs['mean_naive'])
    # 验证整幅图，包括边界；不是只看预览是否相似。
    np.testing.assert_allclose(outputs['mean'], outputs['mean_naive'], rtol=0, atol=1e-6)
    return outputs, timings, {'max_abs_error': float(difference.max()),
                              'rmse': float(np.sqrt(np.mean(difference.astype(np.float64)**2)))}


def panel(gray, outputs, timings, errors, radius):
    canvas = np.full((830, 1000, 3), 24, np.uint8)
    delta = np.abs(outputs['mean']-outputs['mean_naive'])
    images = [('Original gray', gray), ('mean (integral table)', outputs['mean']),
              ('mean_naive (neighborhood loops)', outputs['mean_naive']), ('Absolute difference x1000', delta*1000)]
    for i, (label, image) in enumerate(images):
        x, y = 10+(i%2)*490, 55+(i//2)*335
        cv2.putText(canvas, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        display = cv2.cvtColor(np.rint(np.clip(image, 0, 1)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        canvas[y:y+300, x:x+480] = image_view(display, (480, 300), zoom=1.)
    texts = [f'radius={radius} | window={(2*radius+1)}x{(2*radius+1)} | center crop, original pixels',
             f'mean: {timings["mean"]:.2f} ms | naive: {timings["mean_naive"]:.2f} ms | ratio: {timings["mean_naive"]/max(timings["mean"], 1e-9):.2f}x',
             f'max error={errors["max_abs_error"]:.3g}; RMSE={errors["rmse"]:.3g}; tolerance=1e-6',
             'N/P: radius 1,3,7,15 | Q: quit | Timing is API cost, not pure kernel or camera FPS']
    for i, text in enumerate(texts):
        cv2.putText(canvas, text, (10, 730+i*25), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
    return canvas


def benchmark(frame, radii, repeats, output):
    # 接口读到帧不代表得到有效场景；拒绝近黑占位画面。
    if float(np.mean(frame)) < 2 and float(np.percentile(frame, 99)) < 4:
        raise RuntimeError('Camera frame is almost black; check iPhone preview/lock state and lighting, then retry')
    gray, pixels = prepare(frame)
    h, w = gray.shape
    records = []
    output.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output/'original.png'), frame): raise IOError('Cannot save experiment frame')
    for radius in radii:
        # 两种实现各预热一次，不把首次动态库加载计入样本。
        compare(pixels, w, h, radius)
        samples = {'mean': [], 'mean_naive': []}
        errors = []
        for index in range(repeats):
            values, timings, error = compare(pixels, w, h, radius, reverse=bool(index%2))
            for name in samples: samples[name].append(timings[name])
            errors.append(error)
        medians = {name: float(np.median(times)) for name, times in samples.items()}
        record = {'radius': radius, 'window': 2*radius+1, 'samples_ms': samples,
                  'median_ms': medians, 'speedup': medians['mean_naive']/medians['mean'],
                  'max_abs_error': max(e['max_abs_error'] for e in errors),
                  'max_rmse': max(e['rmse'] for e in errors)}
        records.append(record)
        for name, image in values.items():
            if not cv2.imwrite(str(output/f'{name}_r{radius}.png'), np.rint(image*255).astype(np.uint8)): raise IOError('Cannot save operator image')
        if not cv2.imwrite(str(output/f'comparison_r{radius}.png'), panel(gray, values, medians, errors[-1], radius)): raise IOError('Cannot save comparison')
        print(json.dumps(record), flush=True)
    report = {'frame_shape': list(frame.shape), 'repeats': repeats,
              'scope': 'Same captured frame for every sample/radius; median API cost includes validation, allocation and ctypes/native call. Excludes capture, grayscale/buffer preparation, error checks and display. Alternate execution order; one warmup per implementation/radius. Not end-to-end FPS.',
              'results': records}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# 实验 1：相同均值算法，不同实现', '',
             f'实际图像 {w}×{h}，每组 {repeats} 次，使用同一帧并交替调用顺序。', '',
             '| 半径 | 窗口边长 | mean 中位 ms | mean_naive 中位 ms | 加速比 | 最大绝对误差 |',
             '|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["radius"]} | {r["window"]} | {r["median_ms"]["mean"]:.3f} | {r["median_ms"]["mean_naive"]:.3f} | {r["speedup"]:.2f}× | {r["max_abs_error"]:.3g} |')
    lines += ['', '这是 API 调用耗时，包含输入检查、结果分配和 C++ 计算；不含采集/转换/显示，也不是视频 FPS。',
              '误差检查：整幅图 rtol=0、atol=1e-6，包括裁剪边界。两者使用同一方框平均算法；区别是逐像素遍历邻域与积分图查表。',
              'mean_naive 工作量随邻域面积增加；mean 先建积分图，再为每个矩形使用四个表项求和。实际耗时还受分配、缓存和调度影响。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=3, help='Current iPhone index; may change after reconnect')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--benchmark', action='store_true', help='Capture one scene, benchmark it, save results, exit')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--radii', type=int, nargs='+', default=[1, 3, 7, 15])
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/camera/experiment_mean')
    args = parser.parse_args()
    if args.camera < 0 or min(args.width, args.height) < 96 or args.repeats < 2 or args.frames < 0 or any(r < 0 or r > 31 for r in args.radii): parser.error('Invalid dimensions, count or radius (0..31)')
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
    title, index, count = 'Operator Lab | Experiment 1: mean', 0, 0
    try:
        if not cap.isOpened(): raise RuntimeError('Camera cannot open; check index and close other previews')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.benchmark:
            for _ in range(20):
                ok, frame = cap.read()
                if not ok or frame is None: raise RuntimeError('No camera frame')
            cap.release()  # 基准测试使用固定的同一帧，不占用相机继续采集。
            report = benchmark(frame, args.radii, args.repeats, args.output)
            print('REPORT: '+str(args.output/'RESULTS.md'), flush=True)
            return
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1000, 830)
        while not args.frames or count < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None: raise RuntimeError('No camera frame')
            gray, pixels = prepare(frame)
            h, w = gray.shape
            radius = args.radii[index]
            outputs, timings, errors = compare(pixels, w, h, radius, reverse=bool(count%2))
            cv2.imshow(title, panel(gray, outputs, timings, errors, radius))
            key = cv2.waitKey(1) & 255
            count += 1
            if count == 1: print(f'PREVIEW: visible={cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)}, captured={frame.shape}', flush=True)
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1: break
            if key in (ord('n'), ord('N'), ord('p'), ord('P')):
                index = (index+(-1 if key in (ord('p'), ord('P')) else 1))%len(args.radii)
    finally:
        cap.release()
        if not args.benchmark: cv2.destroyAllWindows()


if __name__ == '__main__': main()
