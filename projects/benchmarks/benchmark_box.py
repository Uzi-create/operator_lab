"""Gravity-aligned 3D box API timings with known cuboid truth."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

from operators.box_ops import gravity_aligned_box
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_box import make_points, evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/box_benchmark')
    parser.add_argument('--repeats', type=int, default=31)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(make_points(seed))[1]) for seed in (341, 19, 481, 719)]
    small = make_points(count=2000)
    large = make_points(count=100_000)
    timings = measure({'box_2008_points': lambda: gravity_aligned_box(small['points']),
                       'box_100008_points': lambda: gravity_aligned_box(large['points'])}, args.repeats)
    sources = ['operators/box_ops.py', 'projects/vision_robot/demo_box.py',
               'projects/benchmarks/benchmark_box.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='Full API incl. validation, recentering, OpenCV minimum-area footprint and float64 extents. Synthetic full cuboid includes all eight vertices.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four known-truth rotated boxes.')


if __name__ == '__main__':
    main()
