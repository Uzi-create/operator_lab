"""Synthetic multi-caliper line and rotated-rectangle measurement examples."""
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
from operators.metrology_ops import measure_line, measure_rectangle


def make_inputs(seed=43, noise=.008, angle=23.):
    rng = np.random.default_rng(seed)
    y, x = np.indices((480, 640), dtype=np.float64)
    line_angle = np.radians(13.)
    direction = np.array([np.cos(line_angle), np.sin(line_angle)])
    normal = np.array([-direction[1], direction[0]])
    point = np.array([280.3, 220.6])
    distance = (x-point[0])*normal[0]+(y-point[1])*normal[1]
    line_image = (.5+.4*np.tanh(distance/1.3)).astype(np.float32)
    line_image = np.clip(line_image+rng.normal(0, noise, line_image.shape), 0, 1).astype(np.float32)
    # Remove a section rather than silently filling unsupported measurements.
    line_image[:, 320:350] = .5
    start, end = point-210*direction+[0, 1], point+240*direction+[0, 1]
    center, size = np.array([320.3, 240.6]), np.array([230.4, 148.7])
    rad = np.radians(angle)
    u = np.array([np.cos(rad), np.sin(rad)])
    v = np.array([-u[1], u[0]])
    shifted = np.stack((x-center[0], y-center[1]), axis=-1)
    distance = np.minimum(size[0]/2-abs(shifted@u), size[1]/2-abs(shifted@v))
    rectangle = (.5+.4*np.tanh(distance/1.2)).astype(np.float32)
    rectangle = np.clip(rectangle+rng.normal(0, noise, rectangle.shape), 0, 1).astype(np.float32)
    return {'line_image': line_image, 'line_start': start, 'line_end': end,
            'line_point': point, 'line_direction': direction, 'line_normal': normal,
            'rectangle_image': rectangle, 'rectangle_center': center,
            'rectangle_size': size, 'rectangle_angle': angle}


def evaluate(data):
    line = measure_line(data['line_image'], data['line_start'], data['line_end'], num_calipers=48)
    rectangle = measure_rectangle(data['rectangle_image'], data['rectangle_center']+[1, -1],
                                  data['rectangle_size']+[1, -1], data['rectangle_angle']+1)
    assert line['success'], line['reason']
    assert rectangle['success'], rectangle['reason']
    line_error = float(abs((line['point']-data['line_point'])@data['line_normal']))
    center_error = float(np.linalg.norm(rectangle['center_xy']-data['rectangle_center']))
    size_error = np.array([rectangle['width_px'], rectangle['height_px']])-data['rectangle_size']
    assert line_error < .1, line_error
    assert center_error < .15, center_error
    assert np.max(np.abs(size_error)) < .2, size_error
    checks = {'line_offset_error_px': line_error, 'line_rms_px': line['rms'],
              'rectangle_center_error_px': center_error, 'rectangle_size_error_px': size_error.tolist(),
              'rectangle_angle_error_deg': rectangle['angle_deg']-data['rectangle_angle']}
    return line, rectangle, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/metrology')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = make_inputs()
    line, rectangle, checks = evaluate(data)
    pictures = []
    for key, result in [('line', line), ('rectangle', rectangle)]:
        image = cv2.cvtColor(np.rint(data[key+'_image']*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        lines = [result] if key == 'line' else result['lines']
        for fitted in lines:
            for point, accepted in zip(fitted['edge_points'], fitted['inliers']):
                cv2.circle(image, tuple(np.rint(point).astype(int)), 3,
                           (0, 230, 0) if accepted else (0, 0, 230), -1)
            ends = np.rint(fitted['segment_xy']).astype(int)
            cv2.line(image, tuple(ends[0]), tuple(ends[1]), (0, 220, 220), 1)
        if key == 'rectangle':
            cv2.polylines(image, [np.rint(result['corners_xy']).astype(np.int32)], True, (255, 160, 20), 1)
            label = f"{result['width_px']:.3f} x {result['height_px']:.3f} px"
        else:
            label = f"Line RMS {result['rms']:.4f} px"
        cv2.putText(image, label, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 220, 0), 1)
        write_image(args.output/(key+'.png'), image)
        pictures.append(image)
    write_image(args.output/'overview.png', cv2.hconcat(pictures))
    report = {'scope': 'Synthetic known geometry with noise/occlusion; not physical measurement accuracy.',
              'checks': checks, 'line_inliers': int(line['inliers'].sum()),
              'line_coverage': line['coverage'], 'rectangle_orthogonality_deg': rectangle['orthogonality_error_deg']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Multi-caliper metrology</title>'
        '<style>body{background:#17202b;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Multi-caliper metrology</h1><p>Synthetic inputs. Yellow: fitted support; green: inlier edge points; blue: rectangle intersections. Units: pixels.</p>'
        '<img src="overview.png">', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
