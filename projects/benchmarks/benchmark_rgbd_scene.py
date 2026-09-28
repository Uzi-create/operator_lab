"""Full organized-depth to 3D ground/obstacle pipeline timing."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_rgbd_scene import make_scene, evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/rgbd_scene_benchmark')
    parser.add_argument('--repeats', type=int, default=31)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(make_scene(seed))[1]) for seed in (328, 19, 117, 241)]
    data = make_scene()
    timings = measure({'rgbd_320x240_components_3d_ground_boxes': lambda: evaluate(data)}, args.repeats)
    files = ['projects/vision_robot/demo_rgbd_scene.py', 'projects/benchmarks/benchmark_rgbd_scene.py',
             'operators/depth_components_ops.py', 'operators/ground_ops.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='320x240 depth image, stride-2 point backprojection, full component/ground/obstacle/visible-box APIs and final truth checks. Synthetic aligned scene only.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: 4 known-truth depth-to-3D scenes.')


if __name__ == '__main__':
    main()
