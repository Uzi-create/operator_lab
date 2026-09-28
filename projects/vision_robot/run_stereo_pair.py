"""Run cached stereo rectification, dense depth and 3D backprojection on files."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.dense_stereo_ops import dense_stereo_depth
from operators.image_io import read_image, write_image
from operators.robot_ops import Intrinsics, depth_to_points
from operators.stereo_rectify_ops import StereoRectifier


def _camera(values, name):
    if not isinstance(values, dict):
        raise ValueError(f'{name} must be an object with width,height,fx,fy,cx,cy')
    try:
        return Intrinsics(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid {name}: {error}') from error


def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Calibration JSON must be an object')
    for name in ('left_camera', 'right_camera', 'T_right_from_left'):
        if name not in data:
            raise ValueError(f'Missing calibration field: {name}')
    left = _camera(data['left_camera'], 'left_camera')
    right = _camera(data['right_camera'], 'right_camera')
    rectifier = StereoRectifier(left, right, data['T_right_from_left'],
                                distortion_left=data.get('distortion_left'),
                                distortion_right=data.get('distortion_right'))
    return rectifier


def run_pair(left_path, right_path, calibration_path, output, *,
             num_disparities=64, min_depth=.05, max_depth=50.):
    rectifier = load_calibration(calibration_path)
    left = read_image(left_path)
    right = read_image(right_path)
    rectified = rectifier.rectify_images(left, right)
    depth_result = dense_stereo_depth(rectified['left'], rectified['right'],
                                      rectified['camera'], rectified['baseline'],
                                      num_disparities=num_disparities,
                                      min_depth=min_depth, max_depth=max_depth)
    cloud = depth_to_points(depth_result['depth'], rectified['camera'],
                            min_depth=min_depth, max_depth=max_depth)
    assert len(cloud['points']) == depth_result['valid_count']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/'depth.npy', depth_result['depth'])
    np.save(output/'disparity_px.npy', depth_result['disparity_px'])
    np.save(output/'point_cloud_xyz.npy', cloud['points'].astype(np.float32))
    write_image(output/'rectified_left.png', rectified['left'])
    write_image(output/'rectified_right.png', rectified['right'])
    preview = np.zeros(depth_result['depth'].shape, np.uint8)
    if depth_result['valid_count']:
        values = depth_result['depth'][depth_result['valid']]
        near, far = np.percentile(values, [2, 98])
        preview[depth_result['valid']] = np.rint(np.clip(
            (far-values)/max(float(far-near), 1e-6), 0, 1)*255).astype(np.uint8)
    colored = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    colored[~depth_result['valid']] = 0
    write_image(output/'depth_preview.png', colored)
    report = dict(success=bool(depth_result['valid_count']),
                  input_images=[str(left_path), str(right_path)],
                  calibration_file=str(calibration_path),
                  size=[rectified['camera'].width, rectified['camera'].height],
                  baseline=rectifier.baseline, valid_pixels=depth_result['valid_count'],
                  valid_fraction=float(depth_result['valid'].mean()),
                  point_cloud_count=len(cloud['points']),
                  median_depth=(float(np.median(values)) if depth_result['valid_count'] else None),
                  left_valid_roi=list(rectifier.roi_left), right_valid_roi=list(rectifier.roi_right),
                  rectified_intrinsics=dict(fx=rectified['camera'].fx, fy=rectified['camera'].fy,
                                            cx=rectified['camera'].cx, cy=rectified['camera'].cy),
                  outputs=['depth.npy', 'disparity_px.npy', 'point_cloud_xyz.npy',
                           'rectified_left.png', 'rectified_right.png', 'depth_preview.png'])
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-disparities', type=int, default=64)
    parser.add_argument('--min-depth', type=float, default=.05)
    parser.add_argument('--max-depth', type=float, default=50.)
    args = parser.parse_args()
    print(json.dumps(run_pair(args.left, args.right, args.calibration, args.output,
                              num_disparities=args.num_disparities,
                              min_depth=args.min_depth, max_depth=args.max_depth), indent=2))


if __name__ == '__main__':
    main()
