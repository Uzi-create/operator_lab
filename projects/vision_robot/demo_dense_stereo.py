"""Known-disparity rectified image pair and dense metric stereo depth."""
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
from operators.image_io import write_image
from operators.robot_ops import Intrinsics, depth_to_points


def make_scene(seed=414, width=320, height=240, occlude=True):
    rng = np.random.default_rng(seed)
    shift = 12*(width//320)
    left = cv2.GaussianBlur(rng.integers(0, 256, (height, width), dtype=np.uint8), (3, 3), 0)
    right = np.zeros_like(left)
    right[:, :-shift] = left[:, shift:]
    if occlude:
        start = int(width*.375)
        end = start+int(width*.09375)
        right[:, start:end] = rng.integers(0, 256, (height, end-start), dtype=np.uint8)
    camera = Intrinsics(width, height, 500.*width/320, 500.*width/320, width/2, height/2)
    return dict(left=left, right=right, camera=camera, baseline=.12,
                truth_disparity_px=shift, truth_depth_m=5.,
                occluded_right_columns=[int(width*.375), int(width*.46875)])


def evaluate(data):
    width = data['camera'].width
    shift = data['truth_disparity_px']
    result = dense_stereo_depth(data['left'], data['right'], data['camera'], data['baseline'],
                                num_disparities=64*width//320)
    clear = np.s_[:, int(width*.56):int(width*.72)]
    occluded = np.s_[:, int(width*.375)+shift:int(width*.46875)+shift]
    clear_valid = result['valid'][clear]
    occluded_valid = result['valid'][occluded]
    median_disparity = float(np.nanmedian(result['disparity_px'][clear]))
    median_depth = float(np.nanmedian(result['depth'][clear]))
    assert clear_valid.mean() > .95
    assert occluded_valid.mean() < .2
    assert abs(median_disparity-shift) <= .125
    assert abs(median_depth-data['truth_depth_m']) < .06
    cloud = depth_to_points(result['depth'], data['camera'], min_depth=.05, max_depth=100.)
    assert len(cloud['points']) == result['valid_count']
    cloud_depth = float(np.median(cloud['points'][:, 2]))
    assert abs(cloud_depth-data['truth_depth_m']) < .06
    u, v = cloud['pixels'][0]
    assert np.isclose(cloud['points'][0, 0],
                      (u-data['camera'].cx)*result['depth'][v, u]/data['camera'].fx)
    checks = dict(image_size=[width, data['camera'].height], truth_disparity_px=shift,
                  truth_depth_m=data['truth_depth_m'], median_clear_disparity_px=median_disparity,
                  median_clear_depth_m=median_depth, clear_valid_fraction=float(clear_valid.mean()),
                  occluded_valid_fraction=float(occluded_valid.mean()),
                  valid_count=result['valid_count'], point_cloud_count=len(cloud['points']),
                  point_cloud_median_depth_m=cloud_depth)
    return result, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1]/'output/dense_stereo')
    args = parser.parse_args()
    data = make_scene()
    result, checks = evaluate(data)
    disparity = np.zeros(data['left'].shape, np.uint8)
    disparity[result['valid']] = np.clip(result['disparity_px'][result['valid']]/20*255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(disparity, cv2.COLORMAP_TURBO)
    colored[~result['valid']] = 0
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'left.png', data['left'])
    write_image(args.output/'right.png', data['right'])
    write_image(args.output/'disparity.png', colored)
    (args.output/'report.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Dense stereo</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Rectified stereo depth</h1><p>Synthetic 12 px disparity and occluded right-image strip.'
        ' Black pixels failed stereo checks.</p><img src="left.png"><img src="right.png">'
        '<img src="disparity.png">', encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
