"""Reproducible CPU/API benchmarks; preserves historical macOS outputs."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from array import array
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import numpy as np
import cv2
from operators.build import compiler_command, library_name
from projects.examples.demo import scene
from operators import run, canny, distance_transform, signed_distance, feather
from operators.competition import grasp_depth, TargetEvidenceGate

ROOT = Path(__file__).resolve().parents[2]
from operators.image_io import read_image
from projects.metal.inspect_surface import inspect as v1, polygon_mask, Parameters
from projects.metal.inspect_surface_v2 import inspect as v2, ridge_response
from operators.defect_ops import local_defect_contrast, group_defect_fragments


def measure(cases, repeats):
    for fn in cases.values():
        fn(); fn()
    samples = {name: [] for name in cases}
    keys = list(cases)
    for repetition in range(repeats):
        for name in keys if repetition % 2 == 0 else keys[::-1]:
            start = time.perf_counter_ns()
            result = cases[name]()
            samples[name].append((time.perf_counter_ns()-start)/1e6)
            del result
    return {name: {'median_ms': statistics.median(values),
                   'p95_ms': sorted(values)[math.ceil(.95*len(values))-1],
                   'samples_ms': values} for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output'/'performance_current')
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error('Use at least 5 repeats')
    cpu = platform.processor()
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
    report = {'platform': platform.platform(), 'cpu': cpu, 'logical_cpus': os.cpu_count(),
              'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
              'opencv_threads': cv2.getNumThreads(), 'repeats': args.repeats, 'warmups': 2,
              'compiler': subprocess.check_output(compiler_command()+['--version'], text=True).splitlines()[0],
              'native_sha256': hashlib.sha256((ROOT/'operators'/library_name()).read_bytes()).hexdigest(),
              'scope': 'CPU API wall time, including validation/allocation; excludes input generation, decoding and output IO. Alternating case order. p95 is descriptive, not a real-time guarantee.',
              'core': {}, 'metal': []}
    for w, h in [(640, 480), (1280, 720), (1920, 1080)]:
        clean, pixels = scene(w, h)
        mask = array('f', (float(v > .5) for v in clean))
        cases = {}
        for mode in ['threshold', 'mean_naive', 'mean', 'adaptive', 'guided', 'clahe', 'erode', 'dilate', 'open', 'close']:
            radius = 64 if mode == 'clahe' else 7
            parameter = 3. if mode == 'clahe' else (.5 if mode == 'threshold' else .12)
            cases[mode] = lambda mode=mode, radius=radius, parameter=parameter: run(pixels, w, h, mode, radius, parameter)
        cases.update(canny=lambda: canny(pixels, w, h), distance=lambda: distance_transform(mask, w, h),
                     signed_distance=lambda: signed_distance(mask, w, h), feather=lambda: feather(mask, w, h, 12))
        error = max(abs(a-b) for a, b in zip(cases['mean'](), cases['mean_naive']()))
        assert error < 2e-6
        report['core'][f'{w}x{h}'] = {'timings': measure(cases, args.repeats), 'mean_max_abs_error': error,
            'parameters': {'radius': 7, 'clahe_tile': 64, 'clahe_clip': 3, 'scale': .12, 'feather_half_width': 12}}
        print('Measured core:', w, h, flush=True)
    gray = np.random.default_rng(2).random((960, 720), dtype=np.float32)
    np.testing.assert_allclose(ridge_response(gray, 'native'), ridge_response(gray, 'numpy'), atol=1e-7)
    report['ridge'] = measure({name: lambda name=name: ridge_response(gray, name) for name in ['native', 'numpy']}, args.repeats)
    for manifest in ['sample_rois.json', 'new_sample_rois.json']:
        for item in json.loads((ROOT/'projects/metal'/manifest).read_text(encoding='utf-8'))['images']:
            image = read_image(ROOT/'projects/metal/samples'/item['file'])
            assert list(image.shape[1::-1]) == item['size']
            roi = polygon_mask(image.shape, item['polygon'])
            reference = polygon_mask(image.shape, item['reference_polygon']) if 'reference_polygon' in item else None
            params = Parameters(**item.get('parameters', {}))
            result, maps = v2(image, roi, params, reference)
            w, h = result['working_size']
            gray = cv2.cvtColor(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            timings = measure({'v1': lambda: v1(image, roi, params, reference),
                               'v2': lambda: v2(image, roi, params, reference),
                               'local_contrast': lambda: local_defect_contrast(gray, maps['inspection_mask']),
                               'grouping': lambda: group_defect_fragments(maps['scratch_mask'], max_gap=12)}, args.repeats)
            report['metal'].append({'file': item['file'], 'manifest': manifest, 'working_size': [w, h],
                                    'timings': timings, 'v2_candidates': len(result['scratch_candidates']),
                                    'ridge_backend': result['ridge_backend']})
        print('Measured photos:', manifest, flush=True)
    w, h = 320, 240
    depth = array('f', [.3])*(w*h)
    mask = bytes(int(130 <= x < 190 and 90 <= y < 150) for y in range(h) for x in range(w))
    estimate = lambda: grasp_depth(depth, mask, w, h, fx=360, fy=360, cx=160, cy=120)
    good = estimate(); gate = TargetEvidenceGate(); counter = iter(range(1000000))
    def update_gate():
        stamp = next(counter)*.04
        return gate.update(good, stamp=stamp, now=stamp, target_key='benchmark')
    report['competition'] = measure({'grasp_depth': estimate, 'evidence_gate': update_gate}, args.repeats)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# CPU performance measurements', '', f"CPU: {cpu}; {report['platform']}",
             f"Python {report['python']}; NumPy {report['numpy']}; OpenCV {report['opencv']}; OpenCV threads: {report['opencv_threads']}",
             f"{args.repeats} calls per case after 2 warmups; alternating order. No image IO, GPU or camera acquisition.", '',
             '| Operator | 640x480 ms | 1280x720 ms | 1920x1080 ms |', '|---|---:|---:|---:|']
    for name in report['core']['640x480']['timings']:
        lines.append('| '+name+' | '+' | '.join(f"{v['timings'][name]['median_ms']:.3f}" for v in report['core'].values())+' |')
    lines += ['', '| Photo | v1 ms | v2 ms | Local contrast ms | Grouping ms |', '|---|---:|---:|---:|---:|']
    for row in report['metal']:
        lines.append('| '+row['file'][:8]+' | '+' | '.join(f"{row['timings'][name]['median_ms']:.3f}" for name in ['v1','v2','local_contrast','grouping'])+' |')
    lines += ['', 'Raw samples, p95, parameters, versions and DLL hash are in benchmark.json.',
              'Historical macOS measurements are a different machine/toolchain baseline, not a before/after optimization comparison.']
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Saved:', args.output/'RESULTS.md')


if __name__ == '__main__':
    main()
