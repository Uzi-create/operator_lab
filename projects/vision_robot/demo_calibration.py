"""Synthetic planar camera calibration with independently known intrinsics."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.calibration_ops import calibrate_planar_camera, detect_chessboard_corners
from operators.image_io import write_image


def make_scene(seed=702, views=24, noise_px=.15):
    rng = np.random.default_rng(seed)
    board = np.array([(x*.05, y*.05, 0.) for y in range(6) for x in range(9)])
    matrix = np.array([[800., 0., 640.], [0., 820., 480.], [0., 0., 1.]])
    distortion = np.array([-.12, .025, .001, -.002, .004])
    pixels = []
    attempts = 0
    while len(pixels) < views:
        attempts += 1
        if attempts > views*100:
            raise RuntimeError('Could not generate enough fully visible checkerboards')
        rotation = rng.uniform([-.5, -.55, -.2], [.5, .55, .2])
        translation = rng.uniform([-.4, -.33, .65], [.28, .25, 1.25])
        projected = cv2.projectPoints(board, rotation, translation, matrix, distortion)[0].reshape(-1, 2)
        if (projected[:, 0].min() < 20 or projected[:, 0].max() > 1259 or
                projected[:, 1].min() < 20 or projected[:, 1].max() > 939):
            continue
        pixels.append(projected+rng.normal(0, noise_px, projected.shape))
    return board, pixels, matrix, distortion


def synthetic_checkerboard():
    gray = np.zeros((560, 740), np.uint8)
    for row in range(7):
        for col in range(10):
            if (row+col) % 2 == 0:
                gray[50+row*60:50+(row+1)*60, 50+col*60:50+(col+1)*60] = 255
    return gray


def evaluate(seed=702):
    board, observations, truth, distortion = make_scene(seed)
    result = calibrate_planar_camera(board, observations, (1280, 960))
    assert result['success'], result['reason']
    camera = result['camera']
    estimated = np.array([camera.fx, camera.fy, camera.cx, camera.cy])
    actual = np.array([truth[0, 0], truth[1, 1], truth[0, 2], truth[1, 2]])
    absolute_error = np.abs(estimated-actual)
    assert absolute_error.max() < 3. and result['rms_px'] < .25
    assert np.max(np.abs(result['distortion']-distortion)) < .02
    found = detect_chessboard_corners(synthetic_checkerboard(), (9, 6))
    assert found['success'] and len(found['corners_xy']) == 54
    checks = dict(seed=seed, view_count=len(observations), points_per_view=len(board),
                  image_size=[1280, 960], noise_sigma_px=.15,
                  truth_intrinsics=actual.tolist(), estimated_intrinsics=estimated.tolist(),
                  intrinsic_absolute_error_px=absolute_error.tolist(),
                  truth_distortion=distortion.tolist(), estimated_distortion=result['distortion'].tolist(),
                  reprojection_rms_px=result['rms_px'],
                  per_view_rms_px=result['per_view_rms_px'].tolist(),
                  normal_spread_deg=result['normal_spread_deg'],
                  checkerboard_corners_detected=len(found['corners_xy']))
    return found, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1]/'output/calibration')
    args = parser.parse_args()
    found, checks = evaluate()
    image = cv2.cvtColor(synthetic_checkerboard(), cv2.COLOR_GRAY2BGR)
    for x, y in found['corners_xy']:
        cv2.circle(image, (round(x), round(y)), 3, (0, 0, 255), -1)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'checkerboard_corners.png', image)
    (args.output/'report.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Camera calibration</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Planar camera calibration</h1><p>24 synthetic tilted views; the displayed checkerboard'
        ' is a separate corner detector check. View observations are generated from known geometry.</p>'
        '<img src="checkerboard_corners.png">', encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
