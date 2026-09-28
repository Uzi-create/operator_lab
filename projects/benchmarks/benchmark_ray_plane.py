"""Public API timings and known-truth reprojection for calibrated plane geometry."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from operators.ray_plane_ops import camera_rays, intersect_image_plane
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_ray_plane import make_inputs, evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/ray_plane_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(make_inputs(seed))[1]) for seed in (745, 19, 81, 143)]
    data = make_inputs(count=20_000)
    solution = intersect_image_plane(data['clean_pixels'], data['camera'], [0, 0, 1], -2.,
                                     distortion=data['distortion'],
                                     T_frame_from_camera=data['transform'])
    assert solution['valid'].all()
    max_clean_error = float(np.max(np.linalg.norm(solution['points_frame']-data['points_frame'], axis=1)))
    assert max_clean_error < 1e-10, max_clean_error
    cases = {
        'rays_20000_pinhole': lambda: camera_rays(data['clean_pixels'], data['camera']),
        'rays_20000_standard_distortion': lambda: camera_rays(data['clean_pixels'], data['camera'], distortion=data['distortion']),
        'plane_20000_pinhole': lambda: intersect_image_plane(data['clean_pixels'], data['camera'], [0, 0, 1], -2., T_frame_from_camera=data['transform']),
        'plane_20000_standard_distortion': lambda: intersect_image_plane(data['clean_pixels'], data['camera'], [0, 0, 1], -2.,
                                      distortion=data['distortion'], T_frame_from_camera=data['transform']),
    }
    timings = measure(cases, args.repeats)
    files = ['operators/ray_plane_ops.py', 'projects/vision_robot/demo_ray_plane.py',
             'projects/benchmarks/benchmark_ray_plane.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks,
                  max_clean_geometry_error_m=max_clean_error,
                  scope='20,000 pixels, whole NumPy/OpenCV public API, calibration/inverse distortion/validation included. Pinhole timings use distorted observations as ordinary coordinates solely for timing, not accuracy.',
                  timings=timings,
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four noisy truth scenes and 20,000 clean distorted-pixel round trips.')


if __name__ == '__main__':
    main()
