"""Wall-clock API benchmarks for the new vision/robot operators, with raw samples."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from projects.vision_robot.demo_vision_robot import check_results, evaluate, make_inputs
from operators.feature_ops import create_orb_template, locate_planar_template
from operators.measurement_ops import estimate_translation, measure_circle, measure_stripes
from operators.perception_ops import elevation_grid, pointcloud_to_depth

ROOT = Path(__file__).resolve().parents[2]


def measure(cases, repeats):
    for fn in cases.values():
        for _ in range(3):
            fn()
    samples = {name: [] for name in cases}
    items = list(cases.items())
    for i in range(repeats):
        for name, fn in (items if i % 2 == 0 else items[::-1]):
            start = time.perf_counter_ns()
            result = fn()
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
            del result
    return {name: {'median_ms': float(np.median(values)),
                   'p95_ms': sorted(values)[math.ceil(.95 * len(values)) - 1],
                   'samples_ms': values} for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/vision_robot_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    parser.add_argument('--opencv-threads', type=int, default=None)
    args = parser.parse_args()
    if args.repeats < 5 or args.repeats > 10000:
        parser.error('repeats must be 5..10000')
    if args.opencv_threads is not None:
        if not 1 <= args.opencv_threads <= 256:
            parser.error('opencv-threads must be 1..256')
        cv2.setNumThreads(args.opencv_threads)
    data = make_inputs()
    model = create_orb_template(data['template'])
    checks = check_results(data, evaluate(data, model))
    for function, call_args in [(pointcloud_to_depth, (data['camera_points'], data['camera'])),
                                (elevation_grid, (data['world_points'], (-2, -1.5, 2, 1.5), .02))]:
        native = function(*call_args, backend='native')
        reference = function(*call_args, backend='numpy')
        for key in reference:
            np.testing.assert_allclose(native[key], reference[key], atol=1e-14, rtol=1e-14, equal_nan=True)
    checks['native_numpy_rasters_match'] = True
    cases = {
        'translation_640x480': lambda: estimate_translation(data['reference'], data['moving']),
        'stripe_250px_width9': lambda: measure_stripes(data['measurement'], (350, 240), (600, 240)),
        'circle_128rays_search12px': lambda: measure_circle(data['measurement'], (184, 242), 72, polarity='dark'),
        'orb_build_320x260': lambda: create_orb_template(data['template']),
        'orb_locate_cached_640x480': lambda: locate_planar_template(data['scene'], model),
        'pointcloud_depth_345600_native': lambda: pointcloud_to_depth(data['camera_points'], data['camera'], backend='native'),
        'pointcloud_depth_345600_numpy': lambda: pointcloud_to_depth(data['camera_points'], data['camera'], backend='numpy'),
        'elevation_200000_grid200x150_native': lambda: elevation_grid(data['world_points'], (-2, -1.5, 2, 1.5), .02, min_points=3, backend='native'),
        'elevation_200000_grid200x150_numpy': lambda: elevation_grid(data['world_points'], (-2, -1.5, 2, 1.5), .02, min_points=3, backend='numpy'),
    }
    cpu = platform.processor()
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
    report = {'platform': platform.platform(), 'cpu': cpu, 'python': platform.python_version(),
              'numpy': np.__version__, 'opencv': cv2.__version__, 'opencv_threads': cv2.getNumThreads(),
              'repeats': args.repeats, 'warmups': 3, 'checks': checks,
              'scope': 'CPU API time including validation/allocation; excludes input generation, image IO and one-time model creation except orb_build. Alternating order. Synthetic inputs; no real-time guarantee.',
              'hashes': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                         ['operators/measurement_ops.py', 'operators/feature_ops.py', 'operators/perception_ops.py', 'operators/perception_raster.hpp',
                          'operators/' + ('operators.dll' if os.name == 'nt' else ('liboperators.dylib' if platform.system() == 'Darwin' else 'liboperators.so')),
                          'projects/vision_robot/demo_vision_robot.py', 'projects/benchmarks/benchmark_vision_robot.py']},
              'timings': measure(cases, args.repeats)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Vision and robot operator timings', '',
             f"{cpu}; OpenCV {cv2.__version__}; threads={cv2.getNumThreads()}; {args.repeats} repetitions after 3 warmups.",
             report['scope'], '', '| Case | Median ms | P95 ms |', '|---|---:|---:|']
    for name, timing in report['timings'].items():
        lines.append(f"| {name} | {timing['median_ms']:.3f} | {timing['p95_ms']:.3f} |")
    lines += ['', 'Raw samples, versions, source hashes and numerical checks are in benchmark.json.']
    (args.output/'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
