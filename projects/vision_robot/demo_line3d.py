"""Known-truth 3D seam/rail axis and observed segment with outliers."""
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
from operators.line3d_ops import fit_line_3d


def make_points(seed=268, count=1000, outliers=200):
    rng = np.random.default_rng(seed)
    point = np.array([.2, -.3, .5])
    direction = np.array([1., .4, -.2])
    direction /= np.linalg.norm(direction)
    distance = rng.uniform(-1., 1., count)
    noise = rng.normal(0., .003, (count, 3))
    noise -= (noise@direction)[:, None]*direction
    points = point+distance[:, None]*direction+noise
    points = np.vstack((points, rng.uniform(-2., 2., (outliers, 3))))
    return dict(points=points, point=point, direction=direction,
                surface_count=count, outlier_count=outliers)


def evaluate(data):
    result = fit_line_3d(data['points'], threshold=.012,
                         min_inliers=max(12, int(.7*len(data['points']))),
                         min_inlier_ratio=.7, min_span=1.8, seed=71)
    assert result['success'], result['reason']
    angle = float(np.degrees(np.arccos(np.clip(abs(result['direction']@data['direction']), -1, 1))))
    offset = float(np.linalg.norm(np.cross(result['point']-data['point'], data['direction'])))
    retained = int(result['inliers'][:data['surface_count']].sum())
    false = int(result['inliers'][data['surface_count']:].sum())
    assert angle < .1 and offset < .001
    assert retained/data['surface_count'] > .99 and false <= max(3, int(.02*data['outlier_count']))
    checks = dict(axis_angle_error_deg=angle, perpendicular_offset_error_m=offset,
                  observed_span_m=result['span'], orthogonal_rms_m=result['rms'],
                  true_line_accepted=retained, deliberate_outliers_accepted=false,
                  iterations=result['iterations'])
    return result, checks


def visualize(data, result):
    canvas = np.full((480, 960, 3), 24, np.uint8)
    for panel, axes in enumerate(((0, 1), (0, 2))):
        origin = np.array([240+panel*480, 240])
        xy = np.rint(data['points'][:, axes]*95+origin).astype(int)
        for (x, y), good in zip(xy, result['inliers']):
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.circle(canvas, (int(x), int(y)), 1, (60, 210, 70) if good else (60, 80, 235), -1)
        ends = np.rint(result['segment'][:, axes]*95+origin).astype(int)
        cv2.line(canvas, tuple(ends[0]), tuple(ends[1]), (0, 220, 245), 2)
        cv2.putText(canvas, 'XY' if panel == 0 else 'XZ', (20+panel*480, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (230, 230, 230), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/line3d')
    args = parser.parse_args()
    data = make_points()
    result, checks = evaluate(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'line3d.png', visualize(data, result))
    report = dict(scope='Synthetic 3D line with transverse noise and 200 deliberate spatial outliers. Segment spans observed inliers only.',
                  checks=checks, point=result['point'].tolist(), direction=result['direction'].tolist(),
                  segment=result['segment'].tolist())
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Robust 3D line</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>3D line and observed segment</h1><p>Green inliers, red rejected points, yellow fitted segment.</p><img src="line3d.png">',
        encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
