"""Known-truth gravity-aligned 3D box from a rotated point cluster."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.box_ops import gravity_aligned_box
from operators.image_io import write_image


def make_points(seed=341, count=2000):
    rng = np.random.default_rng(seed)
    center = np.array([.3, -.2, .5])
    size = np.array([.8, .35, .4])
    angle = 23.
    rotation = cv2.Rodrigues(np.array([0., 0., np.deg2rad(angle)]))[0]
    extrema = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    local = np.vstack((extrema, rng.uniform(-.5, .5, (count, 3))))
    points = (local*size)@rotation.T+center
    return dict(points=points, center=center, size=size, yaw_deg=angle)


def evaluate(data):
    result = gravity_aligned_box(data['points'])
    assert result['success'], result['reason']
    error = result['size']-data['size']
    center_error = float(np.linalg.norm(result['center']-data['center']))
    assert np.max(abs(error)) < 1e-6 and center_error < 1e-6
    local = (data['points']-result['center'])@result['axes']
    assert (abs(local) <= result['size']/2+1e-12).all()
    return result, dict(center_error_m=center_error, size_errors_m=error.tolist(),
                        footprint_area_m2=result['footprint_area'], volume_m3=result['volume'],
                        all_points_enclosed=True)


def visualize(data, result):
    canvas = np.full((500, 1000, 3), 24, np.uint8)
    for panel, axes in enumerate(((0, 1), (0, 2))):
        projected = np.rint(data['points'][:, axes]*300+[250+500*panel, 250]).astype(int)
        for x, y in projected:
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.circle(canvas, (int(x), int(y)), 1, (160, 160, 160), -1)
        corners = np.rint(result['corners'][:, axes]*300+[250+500*panel, 250]).astype(int)
        for layer in (0, 4):
            cv2.polylines(canvas, [corners[layer:layer+4]], True, (50, 210, 70), 2)
        for i in range(4):
            cv2.line(canvas, tuple(corners[i]), tuple(corners[i+4]), (50, 210, 70), 2)
        cv2.putText(canvas, 'XY' if panel == 0 else 'XZ', (20+500*panel, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (230, 230, 230), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/oriented_box')
    args = parser.parse_args()
    data = make_points()
    result, checks = evaluate(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'box.png', visualize(data, result))
    report = dict(scope='Synthetic full cuboid incl. all eight extreme vertices; actual partial camera views underestimate unseen volume.',
                  checks=checks, center=result['center'].tolist(), size=result['size'].tolist(),
                  axes=result['axes'].tolist())
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Gravity-aligned box</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Minimum-area horizontal box</h1><p>Green box encloses all supplied 3D points. Synthetic complete cuboid with known extent.</p><img src="box.png">',
        encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
