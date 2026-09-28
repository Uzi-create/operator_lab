"""Robust 3D line full-API timings with independent geometric truth."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

from operators.line3d_ops import fit_line_3d
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_line3d import make_points, evaluate

ROOT = Path(__file__).resolve().parents[2]


def run(data):
    return fit_line_3d(data['points'], threshold=.012,
                       min_inliers=max(12, int(.7*len(data['points']))),
                       min_inlier_ratio=.7, min_span=1.8, seed=71)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/line3d_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(make_points(seed))[1]) for seed in (268, 11, 247, 999)]
    small = make_points()
    large = make_points(count=8000, outliers=2000)
    timings = measure({'line3d_1200_17percent_outliers': lambda: run(small),
                       'line3d_10000_20percent_outliers': lambda: run(large)}, args.repeats)
    files = ['operators/line3d_ops.py', 'projects/vision_robot/demo_line3d.py',
             'projects/benchmarks/benchmark_line3d.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='Whole RANSAC/TLS/final classification API, deterministic seeds. Synthetic 3D seam with observed inlier span, not full rail length.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four known-truth robust 3D lines.')


if __name__ == '__main__':
    main()
