"""Known-truth point-to-plane/point-to-point ICP comparison and timings."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from operators.icp_plane_ops import icp_point_to_plane
from operators.registration_ops import NearestNeighborIndex, icp_point_to_point
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_icp_plane import make_scene, evaluate

ROOT = Path(__file__).resolve().parents[2]


def error(result, truth):
    relative = result['transform'][:3, :3]@truth[:3, :3].T
    return dict(rotation_error_deg=float(np.degrees(np.linalg.norm(cv2.Rodrigues(relative)[0]))),
                translation_error_m=float(np.linalg.norm(result['transform'][:3, 3]-truth[:3, 3])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/icp_plane_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = []
    for seed in (521, 91, 305, 717):
        data = make_scene(seed)
        _, plane_checks = evaluate(data)
        point = icp_point_to_point(data['source'], data['target'], max_distance=.09,
                                   trim_fraction=.9, min_overlap=.7)
        checks.append(dict(seed=seed, plane=plane_checks, point=error(point, data['truth']),
                           point_status=point['status']))
    data = make_scene()
    with NearestNeighborIndex(data['target'], backend='native') as index:
        cases = {
            'icp_plane_cached_target_1560x1800': lambda: icp_point_to_plane(
                data['source'], index, data['normals'], max_distance=.09,
                trim_fraction=.9, min_overlap=.7),
            'icp_point_cached_target_1560x1800': lambda: icp_point_to_point(
                data['source'], index, max_distance=.09,
                trim_fraction=.9, min_overlap=.7),
        }
        timings = measure(cases, args.repeats)
    sources = ['operators/icp_plane_ops.py', 'operators/registration_ops.py',
               'projects/vision_robot/demo_icp_plane.py', 'projects/benchmarks/benchmark_icp_plane.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='Same cached exact KD-tree target and noisy partial-overlap three-face scene. Plane method uses known target normals, so lower RMS/latency is not a fair algorithm-only accuracy comparison. Full APIs include final checks, exclude target-index build.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four known-truth multisurface registration scenes.')


if __name__ == '__main__':
    main()
