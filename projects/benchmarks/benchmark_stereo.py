"""Full stereo triangulation timing with seeded truth and wrong matches."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

from operators.stereo_ops import triangulate_stereo
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_stereo import make_scene, evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/stereo_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = []
    for seed in (58, 418, 921, 274):
        data = make_scene(seed)
        checks.append(dict(seed=seed, **evaluate(data)[1]))
    cases = {}
    for count in (2000, 20000):
        data = make_scene(58, count, 120)
        cases[f'stereo_midpoint_{count}'] = lambda data=data: triangulate_stereo(
            data['left'], data['right'], data['camera'], data['camera'], data['pose'],
            distortion_left=data['distortion'], distortion_right=data['distortion'])
        cases[f'stereo_dlt_{count}'] = lambda data=data: triangulate_stereo(
            data['left'], data['right'], data['camera'], data['camera'], data['pose'],
            distortion_left=data['distortion'], distortion_right=data['distortion'], method='dlt')
    timings = measure(cases, args.repeats)
    files = ['operators/stereo_ops.py', 'projects/vision_robot/demo_stereo.py',
             'projects/benchmarks/benchmark_stereo.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='Full stereo API with two distortion inversions, C++ DLT and final two-view reprojection; excludes image matching and calibration. Synthetic known truth only.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four known-truth stereo scenes.')


if __name__ == '__main__':
    main()
