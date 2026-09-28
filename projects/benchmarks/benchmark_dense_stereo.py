"""Known-shift dense stereo accuracy and full-call timing."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

from operators.dense_stereo_ops import dense_stereo_depth
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_dense_stereo import make_scene, evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/dense_stereo_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(make_scene(seed))[1]) for seed in (414, 27, 919, 611)]
    checks.append(dict(seed=414, **evaluate(make_scene(414, 640, 480))[1]))
    cases = {}
    for width, height in ((320, 240), (640, 480)):
        data = make_scene(414, width, height)
        for reverse in (True, False):
            name = f'dense_stereo_{width}x{height}_reverse_{reverse}'
            cases[name] = lambda data=data, reverse=reverse: dense_stereo_depth(
                data['left'], data['right'], data['camera'], data['baseline'],
                num_disparities=64*data['camera'].width//320, left_right_check=reverse)
    timings = measure(cases, args.repeats)
    files = ['operators/dense_stereo_ops.py', 'projects/vision_robot/demo_dense_stereo.py',
             'projects/benchmarks/benchmark_dense_stereo.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='Full rectified stereo API; two SGBM passes when reverse check is enabled. Synthetic translated random texture only, excludes rectification and camera capture.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: five known-shift occlusion scenes.')


if __name__ == '__main__':
    main()
