"""Reproducible synthetic examples of metrology, planar matching and 3D rasters."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.feature_ops import create_orb_template, locate_planar_template
from operators.measurement_ops import estimate_translation, measure_circle, measure_stripes
from operators.perception_ops import elevation_grid, pointcloud_to_depth
from operators.robot_ops import Intrinsics, depth_to_points
from operators.image_io import write_image


def make_inputs():
    rng = np.random.default_rng(20260925)
    reference = cv2.GaussianBlur(rng.random((480, 640), dtype=np.float32), (0, 0), 1.3)
    shift = np.array([7.3, -4.6])
    moving = cv2.warpAffine(reference, np.array([[1., 0., shift[0]], [0., 1., shift[1]]]),
                            (640, 480), borderMode=cv2.BORDER_REFLECT101)

    template_u8 = np.full((260, 320), 80, np.uint8)
    for _ in range(180):
        xy = tuple(int(v) for v in rng.integers([16, 16], [304, 244]))
        cv2.circle(template_u8, xy, int(rng.integers(2, 10)), int(rng.integers(10, 245)), -1)
    for _ in range(30):
        p, q = rng.integers([15, 15], [305, 245], (2, 2))
        cv2.line(template_u8, tuple(p), tuple(q), int(rng.integers(20, 235)), 1)
    cv2.putText(template_u8, 'PART 2026', (25, 138), cv2.FONT_HERSHEY_SIMPLEX, 1, 245, 2)
    template = template_u8.astype(np.float32) / 255
    source_corners = np.array([[0, 0], [319, 0], [319, 259], [0, 259]], np.float32)
    target_corners = np.array([[173, 62], [497, 99], [470, 390], [133, 340]], np.float32)
    homography = cv2.getPerspectiveTransform(source_corners, target_corners)
    scene = cv2.warpPerspective(template, homography, (640, 480))
    scene[240:277, 373:423] = .25  # A modest occlusion.

    yy, xx = np.indices((480, 640), dtype=np.float32)
    true_center, true_radius = np.array([185.3, 240.6]), 72.4
    disc = .5 + .4 * np.tanh((true_radius - np.hypot(xx - true_center[0], yy - true_center[1])) / 1.3)
    bars = .1 + .4 * (np.tanh((xx - 410.3) / 1.5) - np.tanh((xx - 442.7) / 1.5))
    bars += .4 * (np.tanh((xx - 501.2) / 1.5) - np.tanh((xx - 526.6) / 1.5))
    bars[(yy < 100) | (yy > 380)] = .1
    measurement = np.maximum(disc, bars).astype(np.float32)

    camera = Intrinsics(640, 480, 580., 580., 319.5, 239.5)
    depth = 1.2 + .15 * xx / 640
    depth[150:310, 250:390] = .7
    points = depth_to_points(depth, camera)['points']
    # A farther layer projects to the same pixels and must lose the z-buffer.
    camera_points = np.vstack((points, points[::8] * 1.4))
    world_xy = rng.uniform([-2, -1.5], [2, 1.5], (200000, 2))
    world_z = .02 * np.sin(world_xy[:, 0] * 3) + rng.normal(0, .002, len(world_xy))
    world_z[(abs(world_xy[:, 0] - .5) < .3) & (abs(world_xy[:, 1]) < .4)] += .25
    world_points = np.column_stack((world_xy, world_z))
    return dict(reference=reference, moving=moving, shift=shift, template=template,
                scene=scene, target_corners=target_corners, measurement=measurement,
                true_center=true_center, true_radius=true_radius, camera=camera,
                depth=depth, camera_points=camera_points, world_points=world_points)


def evaluate(data, model=None):
    if model is None:
        model = create_orb_template(data['template'])
    return {
        'translation': estimate_translation(data['reference'], data['moving']),
        'planar': locate_planar_template(data['scene'], model),
        'stripes': measure_stripes(data['measurement'], (350, 240), (600, 240)),
        'circle': measure_circle(data['measurement'], (184, 242), 72, polarity='dark'),
        'depth': pointcloud_to_depth(data['camera_points'], data['camera']),
        'elevation': elevation_grid(data['world_points'], (-2, -1.5, 2, 1.5), .02, min_points=3),
    }


def check_results(data, out):
    assert out['translation']['success'], out['translation']['reason']
    assert out['planar']['success'], out['planar']['reason']
    assert out['circle']['success'], out['circle']['reason']
    np.testing.assert_allclose(out['translation']['shift_xy'], data['shift'], atol=.35)
    np.testing.assert_allclose(out['planar']['corners_xy'], data['target_corners'], atol=4)
    np.testing.assert_allclose(out['circle']['center'], data['true_center'], atol=.2)
    assert abs(out['circle']['radius'] - data['true_radius']) < .2
    widths = [stripe['width_px'] for stripe in out['stripes']['stripes']]
    np.testing.assert_allclose(widths, [32.4, 25.4], atol=.2)
    np.testing.assert_array_equal(out['depth']['depth'], data['depth'].astype(np.float64))
    assert out['depth']['valid'].all()
    assert int(out['elevation']['count'].sum()) == len(data['world_points'])
    return {'translation_error_px': (out['translation']['shift_xy'] - data['shift']).tolist(),
            'planar_max_corner_error_px': float(np.linalg.norm(out['planar']['corners_xy'] - data['target_corners'], axis=1).max()),
            'planar_inliers': out['planar']['inlier_count'],
            'circle_center_error_px': float(np.linalg.norm(out['circle']['center'] - data['true_center'])),
            'circle_radius_error_px': float(out['circle']['radius'] - data['true_radius']),
            'stripe_widths_px': widths, 'depth_exact_roundtrip': True,
            'grid_count_conserved': True}


def color_values(values, valid):
    values = np.where(valid, values, 0)
    output = np.zeros(values.shape, np.uint8)
    if valid.any():
        low, high = values[valid].min(), values[valid].max()
        if high > low:
            output[valid] = np.rint(255 * (values[valid] - low) / (high - low)).astype(np.uint8)
    colored = cv2.applyColorMap(output, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/vision_robot_new')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = make_inputs()
    out = evaluate(data)
    report = {'scope': 'Synthetic geometry checks; not field accuracy or calibration.',
              'checks': check_results(data, out)}
    gray8 = lambda a: np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)
    measurement = cv2.cvtColor(gray8(data['measurement']), cv2.COLOR_GRAY2BGR)
    circle = out['circle']
    for point in circle['edge_points'][circle['inliers']]:
        cv2.circle(measurement, tuple(np.rint(point).astype(int)), 2, (0, 220, 0), -1)
    cv2.circle(measurement, tuple(np.rint(circle['center']).astype(int)), 3, (0, 0, 255), -1)
    cv2.putText(measurement, f"radius {circle['radius']:.2f} px", (70, 345), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 0), 1)
    for stripe in out['stripes']['stripes']:
        p, q = (tuple(np.rint(stripe[key]).astype(int)) for key in ('start_xy', 'end_xy'))
        cv2.line(measurement, p, q, (0, 0, 255), 2)
        cv2.putText(measurement, f"{stripe['width_px']:.2f}", (p[0] - 10, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 220), 1)
    located = cv2.cvtColor(gray8(data['scene']), cv2.COLOR_GRAY2BGR)
    corners = np.rint(out['planar']['corners_xy']).astype(np.int32)
    cv2.polylines(located, [corners], True, (0, 230, 0), 2)
    for point in out['planar']['scene_points_xy'][out['planar']['inliers']]:
        cv2.circle(located, tuple(np.rint(point).astype(int)), 2, (0, 200, 255), -1)
    aligned = cv2.warpAffine(data['moving'], out['translation']['alignment_matrix'], (640, 480))
    images = {'measurements.png': measurement, 'planar_match.png': located,
              'planar_template.png': gray8(data['template']),
              'translation_reference.png': gray8(data['reference']),
              'translation_moving.png': gray8(data['moving']), 'translation_aligned.png': gray8(aligned),
              'depth.png': color_values(out['depth']['depth'], out['depth']['valid']),
              'elevation.png': color_values(out['elevation']['max_z'], out['elevation']['valid'])}
    for name, image in images.items():
        write_image(args.output/name, image)
    # Keep numerical data separate from normalized display colors.
    np.savez_compressed(args.output/'geometry.npz', depth=out['depth']['depth'],
                        source_index=out['depth']['source_index'], grid_max_z=out['elevation']['max_z'],
                        grid_mean_z=out['elevation']['mean_z'], grid_count=out['elevation']['count'],
                        grid_valid=out['elevation']['valid'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    tiles = ''.join(f'<figure><img src="{name}"><figcaption>{name}</figcaption></figure>' for name in images)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Vision and robot operators</title>'
        '<style>body{font:16px system-ui;background:#14202b;color:#e9f0f6;margin:28px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}figure{margin:0;padding:14px;background:#223343;border-radius:8px}img{width:100%;height:310px;object-fit:contain}figcaption{padding-top:8px}</style>'
        '<h1>Vision and robot operators</h1><p>Synthetic examples. Measurements in pixels; depth and elevation in meters. Color maps are for display.</p><main>'
        + tiles + '</main>', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved', args.output/'index.html')


if __name__ == '__main__':
    main()
