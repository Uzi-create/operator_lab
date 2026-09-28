"""Robust 3D sphere measurement with known radius and deliberate outliers."""
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
from operators.sphere_ops import fit_sphere


def make_points(seed=827, count=1000, outliers=200):
    rng = np.random.default_rng(seed)
    center = np.array([.2, -.3, .5])
    radius = .7
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    points = center+directions*(radius+rng.normal(0, .003, count))[:, None]
    points = np.vstack((points, rng.uniform(-2., 2., (outliers, 3))))
    return dict(points=points, center=center, radius=radius,
                surface_count=count, outlier_count=outliers)


def evaluate(data):
    required = max(12, int(np.ceil(len(data['points'])*.7)))
    result = fit_sphere(data['points'], threshold=.015, min_inliers=required,
                        min_inlier_ratio=.7, seed=41)
    assert result['success'], result['reason']
    count = data['surface_count']
    true_accepted = int(result['inliers'][:count].sum())
    false_accepted = int(result['inliers'][count:].sum())
    center_error = float(np.linalg.norm(result['center']-data['center']))
    radius_error = float(result['radius']-data['radius'])
    # Uniform clutter can land inside the radial tolerance by chance; it is
    # geometrically indistinguishable from a surface point by this operator.
    assert true_accepted/count > .99 and false_accepted <= max(5, int(np.ceil(.03*data['outlier_count'])))
    assert center_error < .001 and abs(radius_error) < .001
    checks = dict(center_error_m=center_error, radius_error_m=radius_error,
                  radial_rms_m=result['rms'], true_surface_accepted=true_accepted,
                  deliberately_wrong_accepted=false_accepted, iterations=result['iterations'])
    return result, checks


def visualize(data, result):
    canvas = np.full((480, 960, 3), 22, np.uint8)
    for panel, axes in enumerate(((0, 1), (0, 2))):
        origin = np.array([240+panel*480, 240])
        projected = np.rint(data['points'][:, axes]*95+origin).astype(int)
        center = tuple(np.rint(result['center'][list(axes)]*95+origin).astype(int))
        cv2.circle(canvas, center, int(round(result['radius']*95)), (60, 210, 70), 1)
        for xy, accepted in zip(projected, result['inliers']):
            x, y = map(int, xy)
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.circle(canvas, (x, y), 1, (50, 210, 70) if accepted else (60, 80, 235), -1)
        cv2.putText(canvas, 'XY' if panel == 0 else 'XZ', (20+panel*480, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (230, 230, 230), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/sphere')
    args = parser.parse_args()
    data = make_points()
    result, checks = evaluate(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'sphere.png', visualize(data, result))
    report = dict(scope='Synthetic fully observed sphere with radial noise and 200 deliberate 3D outliers; not real scanner performance.',
                  checks=checks, center=result['center'].tolist(), radius_m=result['radius'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Robust sphere measurement</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Robust 3D sphere</h1><p>Green: radial inliers and fitted circle projection. Red: rejected points.</p><img src="sphere.png">',
        encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
