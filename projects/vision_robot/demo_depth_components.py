"""Known-truth organized depth components with holes, noise and small clutter."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.depth_components_ops import depth_components
from operators.image_io import write_image


def make_depth(seed=982, shape=(480, 640)):
    height, width = shape
    rng = np.random.default_rng(seed)
    y, x = np.indices(shape)
    truth = np.zeros(shape, np.uint8)
    truth[:, :] = 1
    first = (x >= int(.12*width)) & (x < int(.35*width)) & (y >= int(.16*height)) & (y < int(.45*height))
    second = (x >= int(.56*width)) & (x < int(.86*width)) & (y >= int(.54*height)) & (y < int(.88*height))
    truth[first] = 2
    truth[second] = 3
    metres = np.where(first, 1., np.where(second, 1.3, 2.)).astype(np.float64)
    metres += rng.normal(0, .0015, shape)
    depth = np.rint(metres*1000).astype(np.uint16)
    holes = rng.random(shape) < .01
    depth[holes] = 0
    truth[holes] = 0
    # Isolated near-depth specks are deliberately too small to keep.
    specks = ((y%31 == 8) & (x%37 == 7) & (truth == 1))
    depth[specks] = 600
    truth[specks] = 0
    return depth, truth


def evaluate(depth, truth, backend='native'):
    result = depth_components(depth, depth_scale=.001, max_depth=3.,
                              absolute_jump=.015, relative_jump=0.,
                              min_area=100, backend=backend)
    assert result['region_count'] == 3, result['regions']
    labels = result['labels']
    np.testing.assert_array_equal(labels, truth.astype(np.int32))
    errors = int(np.count_nonzero(labels != truth))
    checks = dict(region_count=3, mislabeled_pixels=errors,
                  labeled_pixels=result['labeled_pixel_count'],
                  region_areas=[region['area'] for region in result['regions']],
                  mean_depth_m=[region['mean_depth'] for region in result['regions']])
    return result, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/depth_components')
    parser.add_argument('--backend', choices=('native', 'numpy', 'auto'), default='native')
    args = parser.parse_args()
    depth, truth = make_depth()
    result, checks = evaluate(depth, truth, backend=args.backend)
    args.output.mkdir(parents=True, exist_ok=True)
    levels = np.clip(depth.astype(np.float32)/2500, 0, 1)
    raw = cv2.applyColorMap(np.rint(levels*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    raw[depth == 0] = 0
    colors = np.array([[0, 0, 0], [160, 160, 160], [65, 210, 65], [60, 180, 245]], np.uint8)
    segments = colors[result['labels']]
    write_image(args.output/'depth.png', raw)
    write_image(args.output/'regions.png', segments)
    write_image(args.output/'overview.png', cv2.hconcat([raw, segments]))
    report = dict(scope='Synthetic three-depth scene. Exact label truth is known; not semantic or field validation.',
                  backend=result['backend'], checks=checks, regions=result['regions'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Organized depth components</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Depth components</h1><p>Left: metric depth; right: jump-connected labels. Holes and small specks are black.</p><img src="overview.png">',
        encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
