"""Full robust sphere fitting timings and independent known-truth checks."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

from operators.sphere_ops import fit_sphere
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_sphere import make_points, evaluate

ROOT = Path(__file__).resolve().parents[2]


def fit(data):
    return fit_sphere(data['points'], threshold=.015,
                      min_inliers=max(12, int(len(data['points'])*.7)),
                      min_inlier_ratio=.7, seed=41)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/sphere_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(make_points(seed))[1]) for seed in (827, 91, 541, 713)]
    small = make_points()
    large = make_points(count=8000, outliers=2000)
    timings = measure({'sphere_1200_17percent_outliers': lambda: fit(small),
                       'sphere_10000_20percent_outliers': lambda: fit(large)}, args.repeats)
    files = ['operators/sphere_ops.py', 'projects/vision_robot/demo_sphere.py',
             'projects/benchmarks/benchmark_sphere.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='Full RANSAC plus geometric refinement and final-pose residual reclassification. Iterations adapt to support. Synthetic full-sphere geometry.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four known-truth robust sphere scenes.')


if __name__ == '__main__':
    main()
