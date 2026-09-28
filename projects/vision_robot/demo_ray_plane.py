"""Calibrated pixel-to-plane metric measurements with known camera geometry."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.image_io import write_image
from operators.ray_plane_ops import intersect_image_plane
from operators.robot_ops import Intrinsics


def make_inputs(seed=745, count=300):
    rng = np.random.default_rng(seed)
    camera = Intrinsics(960, 720, 760., 770., 480., 360.)
    matrix = np.array([[760., 0, 480.], [0, 770., 360.], [0, 0, 1.]])
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.array([.12, -.16, .04]))[0]
    transform[:3, 3] = [.08, -.04, .13]
    points_frame = np.column_stack((rng.uniform(-.45, .45, count),
                                     rng.uniform(-.32, .32, count), np.full(count, 2.)))
    points_camera = (points_frame-transform[:3, 3]) @ transform[:3, :3]
    distortion = np.array([-.13, .035, .001, -.002, .004])
    clean = cv2.projectPoints(points_camera, np.zeros(3), np.zeros(3), matrix,
                              distortion)[0].reshape(-1, 2)
    pixels = clean+rng.normal(0, .15, clean.shape)
    return dict(camera=camera, matrix=matrix, transform=transform, points_frame=points_frame,
                distortion=distortion, pixels=pixels, clean_pixels=clean)


def evaluate(data):
    result = intersect_image_plane(data['pixels'], data['camera'], [0, 0, 1], -2.,
                                   distortion=data['distortion'],
                                   T_frame_from_camera=data['transform'])
    assert result['valid'].all()
    errors = np.linalg.norm(result['points_frame']-data['points_frame'], axis=1)
    metrics = dict(points=len(errors), mean_error_mm=float(errors.mean()*1000),
                   p95_error_mm=float(np.percentile(errors, 95)*1000),
                   max_error_mm=float(errors.max()*1000),
                   max_inverse_reprojection_error_px=float(result['reprojection_errors_px'].max()))
    assert metrics['p95_error_mm'] < 1.5, metrics
    return result, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/ray_plane')
    args = parser.parse_args()
    data = make_inputs()
    result, checks = evaluate(data)
    args.output.mkdir(parents=True, exist_ok=True)
    canvas = np.full((720, 960, 3), 25, np.uint8)
    for i, (true, measured) in enumerate(zip(data['clean_pixels'], data['pixels'])):
        cv2.circle(canvas, tuple(np.rint(true).astype(int)), 2, (60, 215, 70), -1)
        cv2.drawMarker(canvas, tuple(np.rint(measured).astype(int)), (255, 170, 55),
                       cv2.MARKER_CROSS, 4, 1)
    cv2.putText(canvas, 'Known plane: green truth pixels / blue noisy observations',
                (25, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 235, 235), 1)
    write_image(args.output/'observations.png', canvas)
    report = dict(scope='Synthetic calibrated camera and supplied plane; metric errors reflect 0.15 px pixel noise.',
                  checks=checks, plane_frame='z=2 metres',
                  T_frame_from_camera=data['transform'].tolist())
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Pixel-to-plane geometry</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Calibrated pixel-to-plane measurement</h1><p>Plane z=2 m in the frame. Distorted observed pixels carry 0.15 px noise; metric output and error are in report.json.</p><img src="observations.png">',
        encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
