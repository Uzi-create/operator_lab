"""Known-correspondence 3D localization followed by known-axis cylinder metrology."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.correspondence_ops import estimate_rigid_transform_robust
from operators.cylinder_ops import fit_cylinder_known_axis
from operators.image_io import write_image


def make_scene(seed=61010, count=1200, clutter=300):
    rng = np.random.default_rng(seed)
    axis = np.array([.2, -.4, .9])
    axis /= np.linalg.norm(axis)
    first = np.cross(axis, [1., 0., 0.])
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    center, radius, length = np.array([.10, -.08, .25]), .075, .32
    angles = rng.uniform(np.deg2rad(-150), np.deg2rad(150), count)
    axial = rng.uniform(-length/2, length/2, count)
    source = (center + axial[:, None] * axis + radius *
              (np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second))
    source = np.vstack((source, center + rng.uniform(-.22, .22, (clutter, 3))))
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.array([.6, -.25, .45]))[0]
    transform[:3, 3] = [.32, -.19, 1.4]
    target = source @ transform[:3, :3].T + transform[:3, 3]
    target += rng.normal(0, .0005, target.shape)
    matched_target = target.copy()
    wrong = rng.choice(len(source), int(.3 * len(source)), replace=False)
    # Wrong target indices model feature-matching mistakes, not an artificial
    # different coordinate unit or a second hidden motion transform.
    matched_target[wrong] = target[np.roll(wrong, 1)]
    return dict(source=source, target=target, matched_target=matched_target,
                wrong_indices=wrong, transform=transform, radius=radius, length=length,
                axis_source=axis, axis_target=transform[:3, :3] @ axis,
                center_target=transform[:3, :3] @ center + transform[:3, 3],
                surface_count=count)


def locate(data):
    return estimate_rigid_transform_robust(data['source'], data['matched_target'],
                                           threshold=.003, min_inlier_ratio=.6, seed=37)


def measure(data, axis=None):
    return fit_cylinder_known_axis(data['target'], data['axis_target'] if axis is None else axis,
                                   threshold=.003, min_radius=.03, max_radius=.12,
                                   min_inlier_ratio=.7, min_coverage_deg=200.,
                                   min_axial_span=.2, seed=41)


def evaluate(data):
    pose = locate(data)
    assert pose['success'], pose['reason']
    transform = pose['transform']
    relative = transform[:3, :3] @ data['transform'][:3, :3].T
    rotation_error = float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2, -1., 1.))))
    translation_error = float(np.linalg.norm(transform[:3, 3] - data['transform'][:3, 3]))
    residuals = np.linalg.norm(data['source'] @ transform[:3, :3].T +
                               transform[:3, 3] - data['matched_target'], axis=1)
    np.testing.assert_allclose(pose['residuals'], residuals, atol=1e-12)
    np.testing.assert_array_equal(pose['inliers'], residuals <= .003)
    assert rotation_error < .08 and translation_error < .0005
    # Known fixture axis in the source frame, rotated by the recovered pose.
    cylinder = measure(data, transform[:3, :3] @ data['axis_source'])
    assert cylinder['success'], cylinder['reason']
    delta = cylinder['axis_point'] - data['center_target']
    perpendicular = delta - (delta @ data['axis_target']) * data['axis_target']
    axis_distance = float(np.linalg.norm(perpendicular))
    radius_error = float(cylinder['radius'] - data['radius'])
    assert axis_distance < .001 and abs(radius_error) < .001
    # Visible point extrema are not physical end-cap measurements.
    measured_span = float(np.ptp(data['target'][cylinder['inliers']] @ cylinder['axis_direction']))
    np.testing.assert_allclose(cylinder['length_observed'], measured_span, atol=1e-12)
    checks = dict(rotation_error_deg=rotation_error, translation_error_m=translation_error,
                  pose_rms_m=pose['rms'], correspondence_inliers=pose['inlier_count'],
                  wrong_correspondences_accepted=int(pose['inliers'][data['wrong_indices']].sum()),
                  cylinder_axis_distance_m=axis_distance, radius_error_m=radius_error,
                  cylinder_rms_m=cylinder['rms'], observed_length_m=cylinder['length_observed'],
                  coverage_deg=cylinder['coverage_deg'])
    return pose, cylinder, checks


def visualize(data, cylinder):
    canvas = np.full((530, 1100, 3), 25, np.uint8)
    center = data['center_target']
    for panel, axes in enumerate(((0, 1), (0, 2))):
        origin = np.array([275 + panel*550, 265])
        pixels = np.rint((data['target'][:, axes] - center[list(axes)]) * 900 + origin).astype(int)
        for (x, y), accepted in zip(pixels, cylinder['inliers']):
            if panel*550 <= x < (panel+1)*550 and 50 < y < 465:
                cv2.circle(canvas, (int(x), int(y)), 1,
                           (65, 205, 65) if accepted else (85, 80, 210), -1)
        ends = np.rint((cylinder['segment_xyz'][:, axes] - center[list(axes)]) * 900 + origin).astype(int)
        cv2.line(canvas, tuple(ends[0]), tuple(ends[1]), (250, 210, 50), 3, cv2.LINE_AA)
        cv2.putText(canvas, ('XY' if panel == 0 else 'XZ') + ' projection / metres',
                    (20+panel*550, 32), cv2.FONT_HERSHEY_SIMPLEX, .7, (230, 230, 230), 1)
    cv2.putText(canvas, 'Green: radial support | Red: rejected clutter | Cyan: observed axis segment',
                (20, 488), cv2.FONT_HERSHEY_SIMPLEX, .62, (230, 230, 230), 1)
    cv2.putText(canvas, 'Known axis required; visible length does not locate physical end caps.',
                (20, 515), cv2.FONT_HERSHEY_SIMPLEX, .58, (230, 230, 230), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'output/correspondence_cylinder')
    args = parser.parse_args()
    data = make_scene()
    pose, cylinder, checks = evaluate(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output / 'localization_measurement.png', visualize(data, cylinder))
    report = dict(scope='Synthetic known correspondences with 30% mismatches, 20% cylinder clutter; not hardware validation.',
                  checks=checks, T_target_from_source=pose['transform'].tolist(),
                  radius_m=cylinder['radius'], axis_point_m=cylinder['axis_point'].tolist(),
                  axis_direction=cylinder['axis_direction'].tolist())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>3D localization and cylinder measurement</title>'
        '<h1>Known-correspondence localization and known-axis metrology</h1>'
        '<p>Synthetic verification only. Green is radial support, not a semantic object label.</p>'
        '<img src="localization_measurement.png" style="max-width:100%">', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
