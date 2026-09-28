"""Create a synthetic calibrated stereo pair and run the file-based workflow."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np

from operators.image_io import write_image
from projects.vision_robot.demo_dense_stereo import make_scene
from projects.vision_robot.run_stereo_pair import run_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1]/'output/stereo_file_workflow')
    args = parser.parse_args()
    scene = make_scene()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = args.output/'inputs'
    inputs.mkdir(exist_ok=True)
    left_path, right_path = inputs/'left.png', inputs/'right.png'
    write_image(left_path, scene['left'])
    write_image(right_path, scene['right'])
    camera = scene['camera']
    intrinsics = dict(width=camera.width, height=camera.height, fx=camera.fx,
                      fy=camera.fy, cx=camera.cx, cy=camera.cy)
    transform = np.eye(4)
    transform[0, 3] = -scene['baseline']
    calibration = inputs/'calibration.json'
    calibration.write_text(json.dumps(dict(left_camera=intrinsics, right_camera=intrinsics,
                                           T_right_from_left=transform.tolist()), indent=2),
                           encoding='utf-8')
    report = run_pair(left_path, right_path, calibration, args.output/'result')
    assert report['success'] and abs(report['median_depth']-5.) < .06
    assert report['point_cloud_count'] == report['valid_pixels']
    saved_depth = np.load(args.output/'result/depth.npy')
    saved_cloud = np.load(args.output/'result/point_cloud_xyz.npy')
    assert saved_depth.shape == scene['left'].shape
    assert len(saved_cloud) == report['point_cloud_count']
    (args.output/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Stereo file workflow</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Calibrated stereo file workflow</h1><p>Two input PNGs and calibration JSON produce'
        ' rectified images, disparity, depth and a metric XYZ cloud.</p>'
        '<img src="result/depth_preview.png">', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
