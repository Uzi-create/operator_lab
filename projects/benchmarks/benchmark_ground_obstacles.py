"""Gravity-constrained ground fitting: truth checks and full-API timing."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

from operators.ground_ops import segment_ground_obstacles
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_ground_obstacles import make_scene, evaluate

ROOT = Path(__file__).resolve().parents[2]


def call(data):
    return segment_ground_obstacles(data['points'], expected_ground_height=0.,
                                    height_tolerance=.08, max_tilt_deg=8.,
                                    distance_threshold=.012, min_ground_points=100,
                                    min_ground_ratio=.1, seed=31)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/ground_obstacles_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(make_scene(seed))[1]) for seed in (211, 19, 308, 601)]
    wall_scene = make_scene()
    floor_scene = make_scene(wall_count=700)
    cases = {
        'ground_10000_wall_dominant': lambda: call(wall_scene),
        'ground_4700_floor_dominant': lambda: call(floor_scene),
    }
    timings = measure(cases, args.repeats)
    files = ['operators/ground_ops.py', 'projects/vision_robot/demo_ground_obstacles.py',
             'projects/benchmarks/benchmark_ground_obstacles.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='Full NumPy API including RANSAC, gravity/height prior, SVD refinement, final masks. Deterministic fixed seed. Iterations adapt to measured support; scene mixtures differ.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four known-truth gravity-constrained scenes.')


if __name__ == '__main__':
    main()
