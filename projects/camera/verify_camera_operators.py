"""Run image operators on the same USB camera frames; report errors separately from no detections."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from array import array
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from operators import core, vision_ops as vision, measurement_ops as measure
from operators.defect_ops import local_defect_contrast, group_defect_fragments
from operators.fast_stats import compute
from operators.ridge_ops import ridge_response
from operators.feature_ops import create_orb_template, locate_planar_template
from operators.shape_ops import create_shape_template, match_shape
from operators.metrology_ops import measure_line, measure_rectangle


def buffer(image):
    # NumPy 二维图 -> 连续的一维 array('f')，每个像素是 4 字节 float。
    # core.run 的 ctypes 接口要求这种缓冲区。frombytes 是 array 的方法。
    values = array('f')
    values.frombytes(np.ascontiguousarray(image, dtype=np.float32).tobytes())
    return values


def cases(gray, previous, context=None):
    """Shared preparation is outside timing. Masks come from this camera image."""
    # 这是“算子登记表”的构造函数，不读取摄像头，也不创建显示窗口。
    # gray 为当前帧 float32 灰度图；previous 为上一帧；context 用于带出绘图辅助数据。
    h, w = gray.shape
    pixels = buffer(gray)
    mask = gray > .5
    mask_pixels = buffer(mask)
    # 共享输入准备：这里会实际执行动态阈值、区域筛选和 OpenCV 边缘提取。
    # 后面的 lambda 只登记调用，不会立即执行其中的算子。
    candidates = vision.dynamic_threshold(gray, radius=9, offset=.04, backend='native')
    selected = vision.select_regions(candidates, min_area=20)
    edge_map = cv2.Canny(np.rint(gray*255).astype(np.uint8), 40, 100)
    yy, xx = np.nonzero(edge_map)
    points = np.column_stack((xx, yy)).astype(np.float64)
    if len(points) > 256: points = points[np.linspace(0, len(points)-1, 256).astype(int)]
    # Template smoke test: select a textured patch from the SAME frame.
    size = min(96, h//3, w//3)
    patches = [(x, y, gray[y:y+size, x:x+size])
               for y in range(size, h-size, size) for x in range(size, w-size, size)]
    x, y, template = max(patches, key=lambda item: float(item[2].std()))
    start, end = (10., h/2), (w-11., h/2)
    jobs = []
    # 前 10 项统一走 core.run。core._MODES 是“名字 -> C++ mode 数字”的表。
    for name in core._MODES:
        radius, parameter = (32, 2.) if name == 'clahe' else (3, .08)
        if name == 'threshold': parameter = .5
        # lambda 是 Python 的匿名函数。n/r/p 的默认值固定住本轮的参数，
        # 避免循环结束后所有函数都错误地使用最后一轮的 name/radius/parameter。
        # 选 adaptive 时，相当于登记：
        # def operation(): return core.run(pixels, w, h, mode='adaptive', radius=3, parameter=.08)
        jobs.append((name, 'custom C++', lambda n=name, r=radius, p=parameter:
                     core.run(pixels, w, h, mode=n, radius=r, parameter=p)))
    # 其他算子直接登记各模块的函数调用，不全部经过 core.run 的 mode 分支。
    jobs += [
        ('canny', 'custom C++', lambda: core.canny(pixels, w, h, low=.06, high=.18)),
        ('distance_transform', 'custom C++', lambda: core.distance_transform(mask_pixels, w, h)),
        ('signed_distance', 'custom C++', lambda: core.signed_distance(mask_pixels, w, h)),
        ('feather', 'custom C++', lambda: core.feather(mask_pixels, w, h)),
        ('ridge_response', 'custom C++', lambda: ridge_response(gray, backend='native')),
        ('box_stats', 'custom C++', lambda: compute(gray, outer=9, backend='native')),
        ('ring_stats', 'custom C++', lambda: compute(gray, inner=3, outer=12, backend='native')),
        ('dynamic_threshold', 'custom C++ + NumPy', lambda: vision.dynamic_threshold(gray, radius=9, backend='native')),
        ('illumination_correct', 'custom C++ + NumPy', lambda: vision.illumination_correct(gray, radius=31, backend='native')),
        ('local_defect_contrast', 'custom C++', lambda: local_defect_contrast(gray, np.ones(gray.shape, bool), backend='native')),
        ('hysteresis_threshold', 'OpenCV + NumPy', lambda: vision.hysteresis_threshold(gray, .4, .7)),
        ('fill_holes', 'OpenCV + NumPy', lambda: vision.fill_holes(mask, max_area=1000)),
        ('select_regions', 'OpenCV + NumPy', lambda: vision.select_regions(candidates, min_area=20)),
        ('region_features', 'OpenCV + NumPy', lambda: vision.region_features(selected, min_area=20)),
        ('group_defect_fragments', 'OpenCV + NumPy', lambda: group_defect_fragments(selected, min_area=20)),
        ('measure_edges', 'OpenCV + NumPy', lambda: vision.measure_edges(gray, start, end)),
        ('measure_stripes', 'OpenCV + NumPy', lambda: measure.measure_stripes(gray, start, end)),
        ('measure_circle', 'OpenCV + NumPy', lambda: measure.measure_circle(gray, (w/2, h/2), min(w, h)/4, num_rays=64)),
        ('measure_line', 'OpenCV + NumPy', lambda: measure_line(gray, start, end)),
        ('measure_rectangle', 'OpenCV + NumPy', lambda: measure_rectangle(gray, (w/2, h/2), (w/2, h/2))),
        ('fit_line', 'NumPy RANSAC', lambda: vision.fit_line(points)),
        ('fit_circle', 'NumPy RANSAC', lambda: vision.fit_circle(points)),
        ('estimate_translation', 'OpenCV + NumPy', lambda: measure.estimate_translation(previous, gray)),
        ('match_template', 'OpenCV + NumPy', lambda: vision.match_template(gray, template, max_matches=1)),
    ]
    # 模板对象保存在 models 中；同一次 cases 创建的闭包共享这个字典。
    models = {}
    if context is not None:
        context.update(template=template, template_box=(x, y, size, size), models=models,
                       points=points, start=start, end=end)
    def make_shape():
        models['shape'] = create_shape_template(template, angles=(0.,), scales=(1.,))
        return {'ready': True}
    def find_shape():
        if 'shape' not in models: raise LookupError('Shape template could not be built from this scene')
        return match_shape(gray, models['shape'], roi=(x, y, size, size), candidates_per_pose=8)
    def make_orb():
        models['orb'] = create_orb_template(gray, max_features=500)
        return {'ready': True}
    def find_orb():
        if 'orb' not in models: raise LookupError('ORB template could not be built from this scene')
        return locate_planar_template(gray, models['orb'])
    jobs += [('create_shape_template', 'OpenCV + NumPy', make_shape),
             ('match_shape', 'OpenCV + NumPy', find_shape),
             ('create_orb_template', 'OpenCV + NumPy', make_orb),
             ('locate_planar_template', 'OpenCV + NumPy', find_orb)]
    # 返回元素格式：(名字, 实现方式说明, 待执行函数)。
    # 浏览器只挑当前选中的函数执行；批量验证程序则依次执行全部函数。
    return jobs


def summarize(value):
    if isinstance(value, array): value = np.asarray(value)
    if isinstance(value, np.ndarray):
        if np.isnan(value).any(): raise AssertionError('NaN in operator output')
        finite = value[np.isfinite(value)]
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'nonzero': int(np.count_nonzero(value)),
                'infinite': int(np.count_nonzero(np.isinf(value))),
                'min': float(finite.min()) if finite.size else None,
                'max': float(finite.max()) if finite.size else None}
    if isinstance(value, dict): return {k: summarize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {'count': len(value), 'first': summarize(value[0]) if value else None}
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): raise AssertionError('Nonfinite scalar output')
    return value


def preview(value, shape):
    if isinstance(value, dict):
        value = next((value[k] for k in ('image', 'score', 'background') if k in value), None)
    if isinstance(value, array): value = np.asarray(value).reshape(shape)
    if not isinstance(value, np.ndarray) or value.shape != shape: return None
    image = value.astype(np.float32)
    finite = np.isfinite(image)
    if not finite.any(): return np.zeros(shape, np.uint8)
    lo, hi = float(image[finite].min()), float(image[finite].max())
    image = np.nan_to_num(image, nan=0., posinf=hi, neginf=lo)
    if lo < 0 or hi > 1: image = (image-lo)/max(hi-lo, 1e-8)
    return np.rint(np.clip(image, 0, 1)*255).astype(np.uint8)


def verify_frames(frames, output, save_previews=False):
    """One warmup frame, then timings. Exceptions never become successful results."""
    records, tiles = {}, []
    if save_previews:
        tile = cv2.resize(frames[-1], (240, 180))
        cv2.rectangle(tile, (0, 0), (240, 24), (0, 0, 0), -1)
        cv2.putText(tile, 'Original camera frame', (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        tiles.append(tile)
    output.mkdir(parents=True, exist_ok=True)
    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        original = gray.copy()
        for name, backend, operation in cases(gray, previous):
            record = records.setdefault(name, {'backend': backend, 'samples_ms': [], 'calls': 0, 'errors': [], 'no_detection_calls': 0})
            begin = perf_counter()
            try:
                value = operation()
                elapsed = (perf_counter()-begin)*1000
                if not np.array_equal(gray, original): raise AssertionError('Operator mutated camera input')
                record['last_result'] = summarize(value)
                record['calls'] += 1
                if index > 0: record['samples_ms'].append(elapsed)
                if isinstance(value, dict) and value.get('success') is False:
                    record['no_detection_calls'] += 1
                if index == len(frames)-1 and save_previews:
                    image = preview(value, gray.shape)
                    if image is not None:
                        if not cv2.imwrite(str(output/(name+'.png')), image): raise IOError('Could not write preview')
                        tile = cv2.cvtColor(cv2.resize(image, (240, 180)), cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(tile, (0, 0), (240, 24), (0, 0, 0), -1)
                        cv2.putText(tile, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                        tiles.append(tile)
            except (ValueError, LookupError) as error:
                record['errors'].append({'frame': index, 'kind': 'scene_or_input_rejected', 'message': str(error)})
            except Exception as error:
                record['errors'].append({'frame': index, 'kind': 'execution_error', 'message': str(error)})
            print(f'frame {index+1}/{len(frames)} {name}: calls={record["calls"]}, errors={len(record["errors"])}', flush=True)
        previous = gray
    for record in records.values():
        record['median_ms'] = float(np.median(record['samples_ms'])) if record['samples_ms'] else None
        record['status'] = 'completed' if not record['errors'] else 'partial_or_rejected'
    if tiles:
        while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
        gallery = np.concatenate([np.concatenate(tiles[i:i+4], axis=1) for i in range(0, len(tiles), 4)], axis=0)
        if not cv2.imwrite(str(output/'gallery.png'), gallery): raise IOError('Could not write gallery')
    return records


def check_references(frame):
    """Actual-frame equality checks, separate from timings and detection accuracy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    h, w = gray.shape
    pixels = buffer(gray)
    pairs = [('mean_vs_naive', np.asarray(core.run(pixels, w, h, 'mean', 3)),
              np.asarray(core.run(pixels, w, h, 'mean_naive', 3)))]
    pairs.append(('ridge_native_vs_numpy', ridge_response(gray, backend='native'),
                  ridge_response(gray, backend='numpy')))
    for name, inner, outer in [('box', -1, 9), ('ring', 3, 12)]:
        a = compute(gray, inner=inner, outer=outer, backend='native')
        b = compute(gray, inner=inner, outer=outer, backend='opencv')
        pairs.extend((name+'_'+key, a[key], b[key]) for key in a)
    checks = {}
    for name, a, b in pairs:
        if a.dtype.kind in 'biu': np.testing.assert_array_equal(a, b)
        else: np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
        checks[name] = {'status': 'passed', 'max_abs_difference': float(np.max(np.abs(a.astype(float)-b.astype(float))))}
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--frames', type=int, default=6, help='Includes one warmup frame')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/camera/full_check')
    parser.add_argument('--save-previews', action='store_true', help='Save final-frame processed images and a gallery')
    args = parser.parse_args()
    if args.camera < 0 or args.frames < 2 or min(args.width, args.height) < 96:
        parser.error('camera >=0, frames >=2, dimensions >=96 required')
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
    frames = []
    try:
        if not cap.isOpened(): raise RuntimeError('Camera cannot open: check index, permission and other camera apps')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        for _ in range(args.frames):
            ok, frame = cap.read()
            if not ok or frame is None or min(frame.shape[:2]) < 96: raise RuntimeError('No usable camera frame')
            frames.append(frame)
    finally:
        cap.release()
    records = verify_frames(frames, args.output, args.save_previews)
    report = {'camera': args.camera, 'frames': len(frames), 'frame_shape': list(frames[0].shape),
              'scope': 'Operator API timing excluding capture, grayscale conversion, shared mask preparation, output validation and file writing; first frame warmup excluded. Not pure kernel timing.',
              'matching_scope': 'Template/ORB/shape match the same frame used to create the template; plumbing smoke test, not tracking or recognition accuracy. Shape: angle=0 scale=1 ROI around source patch. Circle/line/rectangle use central geometric priors, not supplied real targets; 2D fits use sampled scene edges and do not identify a semantic target.',
              'not_camera_validated': 'Stereo/depth/3D/robot/hand-eye/physical metrology require additional valid data or calibration; use their synthetic regression tests.',
              'reference_checks_last_frame': check_references(frames[-1]),
              'operators': records}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# 相机算子验证结果', '',
             f'相机索引：{args.camera}；实际帧：{report["frame_shape"]}；共 {len(frames)} 帧，第一帧预热不计时。', '',
             '计时包括算子 API 调用，不包括采集、灰度转换、共享输入准备、结果验证和写文件。',
             '模板使用同帧自匹配；形状仅测试角度 0、缩放 1 和源模板附近的 ROI。测量使用近似中心位置，未提供真实目标。',
             '成功调用不等于识别准确；未检出单独记录。拟合点来自场景边缘，长度单位为像素。', '',
             '| 算子 | 实现 | 成功调用 | 未检出 | 拒绝/错误 | 中位耗时 ms |',
             '|---|---|---:|---:|---:|---:|']
    for name, record in records.items():
        timing = f'{record["median_ms"]:.3f}' if record['median_ms'] is not None else '-'
        lines.append(f'| {name} | {record["backend"]} | {record["calls"]} | {record["no_detection_calls"]} | {len(record["errors"])} | {timing} |')
    lines += ['', f'真实帧数值对比：{len(report["reference_checks_last_frame"])} 项通过。均值优化版/朴素版、细线响应 C++/NumPy、方框及环形统计 C++/OpenCV。浮点容差 atol=rtol=2e-5，整数/掩码要求完全一致。', '',
              '双目、深度、三维、机器人、手眼和物理尺寸测量未用本相机验证：需要额外输入或标定。完整信息见 report.json。']
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Report: {args.output / "report.json"}', flush=True)
    if any(e['kind'] == 'execution_error' for r in records.values() for e in r['errors']): return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
