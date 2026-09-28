"""Cached stereo rectification accuracy and full-image remap timings."""
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

from operators.stereo_rectify_ops import StereoRectifier
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_stereo_rectify import make_scene, make_strong_distortion_scene, evaluate, dot_images

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/stereo_rectify_benchmark')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(make_scene(seed))[2]) for seed in (58, 418, 921, 274)]
    strong = make_strong_distortion_scene()
    strong_rectifier, _, strong_check = evaluate(strong)
    old_left = cv2.undistortPoints(strong['left'].reshape(-1, 1, 2),
                                   strong_rectifier.left_matrix, strong['distortion'],
                                   R=strong_rectifier.R_left, P=strong_rectifier.P_left).reshape(-1, 2)
    old_right = cv2.undistortPoints(strong['right'].reshape(-1, 1, 2),
                                    strong_rectifier.right_matrix, strong['distortion'],
                                    R=strong_rectifier.R_right, P=strong_rectifier.P_right).reshape(-1, 2)
    checks.append(dict(scene='strong_distortion_wide_field', **strong_check,
                       opencv_default_vertical_error_max_px=float(
                           np.max(np.abs(old_left[:, 1]-old_right[:, 1])))))
    data = make_scene()
    rectifier, _, _ = evaluate(data)
    left, right = dot_images(data)
    cases = {
        'rectifier_build_1280x960': lambda: StereoRectifier(
            data['camera'], data['camera'], data['pose'],
            distortion_left=data['distortion'], distortion_right=data['distortion']),
        'cached_remap_pair_1280x960': lambda: rectifier.rectify_images(left, right),
        'rectify_2000_correspondences': lambda: rectifier.rectify_points(data['left'], data['right']),
        'rectify_1500_strong_distortion': lambda: strong_rectifier.rectify_points(
            strong['left'], strong['right']),
    }
    timings = measure(cases, args.repeats)
    files = ['operators/stereo_rectify_ops.py', 'projects/vision_robot/demo_stereo_rectify.py',
             'projects/benchmarks/benchmark_stereo_rectify.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  scope='OpenCV C++ stereoRectify/map initialization separated from cached two-image remap and point rectification. Synthetic calibrated geometry only.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: five known-truth rectification scenes, including strong distortion.')


if __name__ == '__main__':
    main()
