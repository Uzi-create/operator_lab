"""Timing and known-truth checks for planar camera calibration."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

from operators.calibration_ops import calibrate_planar_camera
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_calibration import make_scene, evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/calibration_benchmark')
    parser.add_argument('--repeats', type=int, default=11)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 1000:
        parser.error('repeats must be 5..1000')
    checks = [evaluate(seed)[1] for seed in (702, 31, 402, 918)]
    board, pixels, _, _ = make_scene()
    timings = measure({'calibrate_24_views_54_points': lambda: calibrate_planar_camera(
        board, pixels, (1280, 960))}, args.repeats)
    files = ['operators/calibration_ops.py', 'projects/vision_robot/demo_calibration.py',
             'projects/benchmarks/benchmark_calibration.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='Full OpenCV C++ calibration plus Python final reprojection checks, 24 views x 54 points; synthetic known camera only. Checkerboard detection is separate and excluded from timing.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four known-truth calibration scenes.')


if __name__ == '__main__':
    main()
