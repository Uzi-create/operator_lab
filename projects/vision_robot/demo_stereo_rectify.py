"""Calibrated raw stereo projection, cached rectification and triangulation."""
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
from operators.stereo_ops import triangulate_stereo
from operators.stereo_rectify_ops import StereoRectifier
from projects.vision_robot.demo_stereo import make_scene as make_stereo_scene


def make_scene(seed=58, count=2000):
    data = make_stereo_scene(seed, count, mismatches=0)
    camera = data['camera']
    matrix = np.array([[camera.fx, 0., camera.cx], [0., camera.fy, camera.cy], [0., 0., 1.]])
    truth = data['truth']
    right_truth = truth @ data['pose'][:3, :3].T + data['pose'][:3, 3]
    data['left'] = cv2.projectPoints(truth, np.zeros(3), np.zeros(3), matrix,
                                     data['distortion'])[0].reshape(-1, 2)
    data['right'] = cv2.projectPoints(right_truth, np.zeros(3), np.zeros(3), matrix,
                                      data['distortion'])[0].reshape(-1, 2)
    return data


def make_strong_distortion_scene(seed=18, count=1500):
    data = make_scene(seed, count)
    rng = np.random.default_rng(seed)
    camera = data['camera']
    matrix = np.array([[camera.fx, 0., camera.cx], [0., camera.fy, camera.cy], [0., 0., 1.]])
    distortion = np.array([-.3, .12, .002, -.002, -.03])
    points = rng.uniform([-.9, -.65, 1.1], [.9, .65, 3.], (count, 3))
    right_points = points @ data['pose'][:3, :3].T + data['pose'][:3, 3]
    left = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
    right = cv2.projectPoints(right_points, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
    inside = ((left[:, 0] > 5) & (left[:, 0] < camera.width-5) &
              (left[:, 1] > 5) & (left[:, 1] < camera.height-5) &
              (right[:, 0] > 5) & (right[:, 0] < camera.width-5) &
              (right[:, 1] > 5) & (right[:, 1] < camera.height-5))
    data.update(truth=points[inside], left=left[inside], right=right[inside],
                distortion=distortion)
    return data


def evaluate(data):
    rectifier = StereoRectifier(data['camera'], data['camera'], data['pose'],
                                distortion_left=data['distortion'],
                                distortion_right=data['distortion'])
    pixels = rectifier.rectify_points(data['left'], data['right'])
    result = triangulate_stereo(pixels['left'], pixels['right'],
                                rectifier.rectified_camera, rectifier.rectified_camera,
                                rectifier.T_right_from_left_rectified)
    truth_rectified = data['truth'] @ rectifier.R_left.T
    errors = np.linalg.norm(result['points_left'][result['valid']]-
                            truth_rectified[result['valid']], axis=1)
    assert pixels['valid'].all()
    assert result['valid_count'] >= .98*len(truth_rectified)
    assert pixels['vertical_errors_px'].max() < 1e-5
    assert errors.max() < 1e-6
    checks = dict(correspondences=len(data['left']), rectified_vertical_error_max_px=float(
        pixels['vertical_errors_px'].max()), triangulated_3d_error_max_m=float(errors.max()),
        baseline_m=rectifier.baseline, valid_count=result['valid_count'],
        points_outside_rectified_image=int(len(truth_rectified)-result['valid_count']),
        left_valid_roi=list(rectifier.roi_left), right_valid_roi=list(rectifier.roi_right))
    return rectifier, pixels, checks


def dot_images(data):
    size = (data['camera'].height, data['camera'].width, 3)
    left, right = np.zeros(size, np.uint8), np.zeros(size, np.uint8)
    for image, pixels in ((left, data['left']), (right, data['right'])):
        for x, y in pixels:
            cv2.circle(image, (round(x), round(y)), 2, (70, 220, 100), -1)
    return left, right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1]/'output/stereo_rectify')
    args = parser.parse_args()
    data = make_scene()
    rectifier, _, checks = evaluate(data)
    raw_left, raw_right = dot_images(data)
    images = rectifier.rectify_images(raw_left, raw_right)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'rectified_left.png', images['left'])
    write_image(args.output/'rectified_right.png', images['right'])
    (args.output/'report.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Stereo rectification</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{width:100%}</style>'
        '<h1>Stereo rectification</h1><p>Projected 3D dots with lens distortion and rotated stereo rig.'
        ' Corresponding rectified dots share the same image row.</p><img src="rectified_left.png">'
        '<img src="rectified_right.png">', encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
