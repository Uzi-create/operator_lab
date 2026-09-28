"""Known-truth hand-eye accuracy and full-call timings."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

from operators.handeye_ops import calibrate_eye_in_hand, calibrate_eye_in_hand_robust
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_handeye import make_scene, evaluate, evaluate_robust, corrupted_observations

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/handeye_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, clean=evaluate(make_scene(seed))[1],
                   robust=evaluate_robust(make_scene(seed))[1]) for seed in (23, 56, 802, 134)]
    small = make_scene(23, 20)
    large = make_scene(23, 100)
    damaged, _ = corrupted_observations(small)
    cases = {
        'handeye_20_views': lambda: calibrate_eye_in_hand(small['robot'], small['observations']),
        'handeye_100_views': lambda: calibrate_eye_in_hand(large['robot'], large['observations']),
        'handeye_robust_20_views_four_outliers': lambda: calibrate_eye_in_hand_robust(
            small['robot'], damaged, min_inliers=12, seed=42),
        'handeye_robust_fixed_128_trials': lambda: calibrate_eye_in_hand_robust(
            small['robot'], damaged, min_inliers=12, seed=42, adaptive=False),
    }
    timings = measure(cases, args.repeats)
    files = ['operators/handeye_ops.py', 'projects/vision_robot/demo_handeye.py',
             'projects/benchmarks/benchmark_handeye.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='Full eye-in-hand API including rigid pose validation, robot motion checks, OpenCV C++ solver and independent fixed-target residuals. Synthetic poses only.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four known-truth hand-eye scenes.')


if __name__ == '__main__':
    main()
