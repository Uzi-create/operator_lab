"""Full-API timings after independent known-truth gates; no field-FPS claim."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np

from projects.benchmarks.benchmark_vision_robot import measure as time_calls
from projects.vision_robot.demo_correspondence_cylinder import make_scene, locate, measure, evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'projects/output/correspondence_cylinder_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    parser.add_argument('--baseline', type=Path, help='Optional earlier correspondence_ops.py for alternating, accuracy-gated comparison')
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(make_scene(seed))[2]) for seed in (61010, 47, 903, 272)]
    small, large = make_scene(), make_scene(count=8000, clutter=2000)
    evaluate(large)
    cases = {
        'pose_1500_30percent_mismatches': lambda: locate(small),
        'pose_10000_30percent_mismatches': lambda: locate(large),
        'cylinder_1500_20percent_clutter': lambda: measure(small),
        'cylinder_10000_20percent_clutter': lambda: measure(large),
    }
    comparison = None
    if args.baseline is not None:
        spec = importlib.util.spec_from_file_location('correspondence_baseline', args.baseline.resolve())
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)

        def before(data):
            return old.estimate_rigid_transform_robust(data['source'], data['matched_target'],
                         threshold=.003, min_inlier_ratio=.6, seed=37)

        comparison = dict(baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(), cases=[])
        for label, data in [('1500', small), ('10000', large)]:
            previous, current = before(data), locate(data)
            assert previous['success'] and current['success']
            np.testing.assert_array_equal(previous['inliers'], current['inliers'])
            np.testing.assert_allclose(previous['transform'], current['transform'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(previous['residuals'], current['residuals'], rtol=0, atol=1e-12)
            comparison['cases'].append(dict(points=len(data['source']), masks_equal=True,
                max_transform_difference=float(np.max(np.abs(previous['transform']-current['transform']))),
                previous_iterations=previous['iterations'], current_iterations=current['iterations']))
        cases['pose_before_1500'] = lambda: before(small)
        cases['pose_before_10000'] = lambda: before(large)
    timings = time_calls(cases, args.repeats)
    if comparison is not None:
        comparison['median_speedup'] = {
            size: timings[f'pose_before_{size}']['median_ms'] /
                  timings[f'pose_{size}_30percent_mismatches']['median_ms'] for size in ('1500', '10000')}
    files = ['operators/correspondence_ops.py', 'operators/cylinder_ops.py',
             'projects/vision_robot/demo_correspondence_cylinder.py',
             'projects/benchmarks/benchmark_correspondence_cylinder.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings, comparison=comparison,
                  scope='Full API including input validation, RANSAC, refinement and final residuals. Synthetic data; no camera capture or display.',
                  implementation='NumPy matrix kernels and bounded Python RANSAC orchestration; not a new custom C++ kernel.',
                  environment=dict(python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__,
                                   system=platform.system(), machine=platform.machine(), processor=platform.processor()),
                  hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: known-truth pose and cylinder gates before timing.')


if __name__ == '__main__':
    main()
