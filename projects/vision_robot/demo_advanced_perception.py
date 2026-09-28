"""Known-truth shape localization, local cloud registration and calibrated PnP."""
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
from operators.pose_ops import estimate_pose_pnp, project_object_points
from operators.registration_ops import icp_point_to_point
from operators.robot_ops import Intrinsics
from operators.shape_ops import create_shape_template, match_shape


def rotation_error_deg(estimated, expected):
    relative = estimated @ expected.T
    skew = np.array([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0],
                     relative[1, 0]-relative[0, 1]])
    return float(np.degrees(np.arctan2(np.linalg.norm(skew)*.5, (np.trace(relative)-1)*.5)))


def make_inputs(seed=71):
    rng = np.random.default_rng(seed)
    template = np.zeros((76, 82), np.float32)
    polygon = np.array([[12, 10], [53, 10], [53, 27], [70, 27],
                        [65, 62], [32, 62], [32, 52], [12, 52]], np.int32)
    cv2.fillPoly(template, [polygon], .85)
    cv2.circle(template, (29, 29), 8, .08, -1)
    cv2.rectangle(template, (45, 38), (54, 48), .08, -1)
    shape_truth = [((90., 90.), -20., .9), ((258., 166.), 25., 1.1)]
    scene = np.zeros((260, 360), np.float32)
    pivot = ((template.shape[1]-1)/2, (template.shape[0]-1)/2)
    for center, angle, scale in shape_truth:
        matrix = cv2.getRotationMatrix2D(pivot, angle, scale)
        matrix[:, 2] += np.asarray(center)-pivot
        scene = np.maximum(scene, cv2.warpAffine(template, matrix, (360, 260)))
    cv2.line(scene, (15, 225), (105, 245), .6, 3)
    scene = np.clip(scene+rng.normal(0., .006, scene.shape), 0, 1).astype(np.float32)

    # An asymmetric 3D cloud with partial overlap. Unit is metres in this demo.
    target_clean = rng.uniform([-.65, -.4, -.25], [.65, .4, .25], (700, 3))
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.array([.028, -.024, .018]))[0]
    transform[:3, 3] = [.022, -.017, .014]
    source = (target_clean[:550]-transform[:3, 3]) @ transform[:3, :3]
    source += rng.normal(0, .0004, source.shape)
    source = np.vstack((source, rng.uniform([1.5, 1.5, 1.5], [2., 2., 2.], (70, 3))))
    target = np.vstack((target_clean+rng.normal(0, .0004, target_clean.shape),
                        rng.uniform([-2., -2., -2.], [-1.5, -1.5, -1.5], (80, 3))))

    camera = Intrinsics(960, 720, 760., 770., 480., 360.)
    camera_matrix = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1.]])
    objects = rng.uniform(-.3, .3, (160, 3))
    rvec, tvec = np.array([.25, -.12, .08]), np.array([.07, -.04, 2.2])
    distortion = np.array([-.12, .03, .001, -.002, .004])
    clean_pixels = cv2.projectPoints(objects, rvec, tvec, camera_matrix, distortion)[0].reshape(-1, 2)
    pixels = clean_pixels+rng.normal(0, .25, clean_pixels.shape)
    pixels[:40] = rng.uniform([0, 0], [960, 720], (40, 2))
    pose_truth = np.eye(4)
    pose_truth[:3, :3], pose_truth[:3, 3] = cv2.Rodrigues(rvec)[0], tvec
    return dict(template=template, scene=scene, shape_truth=shape_truth,
                source=source, target=target, registration_truth=transform,
                objects=objects, pixels=pixels, camera=camera, distortion=distortion,
                pose_truth=pose_truth, clean_pixels=clean_pixels)


def make_model(data):
    return create_shape_template(data['template'], angles=(-20., 0., 25.), scales=(.9, 1., 1.1))


def evaluate(data, model=None, backend='native'):
    model = make_model(data) if model is None else model
    shape = match_shape(data['scene'], model, max_matches=2, candidates_per_pose=8)
    registration = icp_point_to_point(data['source'], data['target'], max_distance=.09,
                                      trim_fraction=.9, min_overlap=.7, backend=backend)
    pose = estimate_pose_pnp(data['objects'], data['pixels'], data['camera'],
                             distortion=data['distortion'], reprojection_threshold=1.5,
                             min_inliers=110)
    assert shape['success'] and len(shape['matches']) == 2, shape['reason']
    shape_checks = []
    for found, (center, angle, scale) in zip(sorted(shape['matches'], key=lambda m: m['center_xy'][0]),
                                            data['shape_truth']):
        error = float(np.linalg.norm(found['center_xy']-center))
        assert error < 1., error
        assert found['angle_degrees'] == angle and found['scale'] == scale
        shape_checks.append(dict(center_error_px=error, angle_error_deg=found['angle_degrees']-angle,
                                 scale_error=found['scale']-scale, support=found['support_fraction']))
    assert registration['converged'], registration['status']
    reg_rot = rotation_error_deg(registration['transform'][:3, :3], data['registration_truth'][:3, :3])
    reg_trans = float(np.linalg.norm(registration['transform'][:3, 3]-data['registration_truth'][:3, 3]))
    assert reg_rot < .05 and reg_trans < .0002, (reg_rot, reg_trans)
    assert not registration['inliers'][550:].any()
    assert pose['success'], pose['reason']
    pose_rot = rotation_error_deg(pose['T_camera_from_object'][:3, :3], data['pose_truth'][:3, :3])
    pose_trans = float(np.linalg.norm(pose['T_camera_from_object'][:3, 3]-data['pose_truth'][:3, 3]))
    assert pose_rot < .15 and pose_trans < .003, (pose_rot, pose_trans)
    assert not pose['inliers'][:40].any()
    checks = dict(shape=shape_checks, candidate_budget_hit=shape['candidate_budget_hit'],
                  registration=dict(rotation_error_deg=reg_rot, translation_error_m=reg_trans,
                                    rms_m=registration['rms'], iterations=registration['iterations'],
                                    overlap=registration['overlap'], outliers_accepted=0),
                  pose=dict(rotation_error_deg=pose_rot, translation_error_m=pose_trans,
                            reprojection_rms_px=pose['rms'], inliers=pose['inlier_count'],
                            outliers_accepted=0))
    return dict(shape=shape, registration=registration, pose=pose), checks


def draw_clouds(data, result):
    canvas = np.full((480, 960, 3), 24, np.uint8)
    clouds = [data['target'][:700], data['source'][:550], result['transformed_source'][:550]]
    for panel, axes in enumerate(((0, 1), (0, 2))):
        for points, color in zip(clouds, ((170, 170, 170), (40, 80, 235), (60, 235, 65))):
            xy = np.rint(points[:, axes]*280+[240+panel*480, 240]).astype(int)
            for x, y in xy:
                cv2.circle(canvas, (int(x), int(y)), 1, color, -1)
        cv2.putText(canvas, ('XY' if panel == 0 else 'XZ')+' (metres)',
                    (20+panel*480, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 1)
    cv2.putText(canvas, 'Gray target | red initial source | green registered', (25, 455),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/advanced_perception')
    parser.add_argument('--backend', choices=('native', 'numpy', 'auto'), default='native')
    args = parser.parse_args()
    data = make_inputs()
    results, checks = evaluate(data, backend=args.backend)
    args.output.mkdir(parents=True, exist_ok=True)
    scene = cv2.cvtColor(np.rint(data['scene']*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for match in results['shape']['matches']:
        cv2.polylines(scene, [np.rint(match['corners_xy']).astype(np.int32)], True, (0, 220, 0), 1)
        cv2.drawMarker(scene, tuple(np.rint(match['center_xy']).astype(int)), (0, 200, 255), cv2.MARKER_CROSS, 10)
    write_image(args.output/'shape.png', scene)
    write_image(args.output/'registration.png', draw_clouds(data, results['registration']))
    pose = results['pose']
    projected = project_object_points(data['objects'], data['camera'], pose['T_camera_from_object'],
                                       distortion=data['distortion'])['pixels']
    canvas = np.full((720, 960, 3), 24, np.uint8)
    for i, (observed, predicted) in enumerate(zip(data['pixels'], projected)):
        observed, predicted = tuple(np.rint(observed).astype(int)), tuple(np.rint(predicted).astype(int))
        cv2.circle(canvas, observed, 3, (65, 235, 65) if pose['inliers'][i] else (40, 80, 235), 1)
        if pose['inliers'][i]:
            cv2.drawMarker(canvas, predicted, (255, 200, 40), cv2.MARKER_CROSS, 5, 1)
    cv2.putText(canvas, 'Green observations | blue reprojection | red rejected outliers', (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230), 1)
    write_image(args.output/'pose.png', canvas)
    report = dict(scope='Synthetic known-truth checks, not camera/robot field validation.',
                  registration_backend=results['registration']['backend'], checks=checks,
                  transforms=dict(T_target_from_source=results['registration']['transform'].tolist(),
                                  T_camera_from_object=pose['T_camera_from_object'].tolist()),
                  limitations=['Shape angle/scale are discrete; clipped-field proposals and finite candidate budget can miss matches.',
                               'ICP is local; convergence does not establish global alignment.',
                               'PnP requires correspondences and calibration; residuals do not prove a unique pose.'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Advanced perception</title>'
        '<style>body{background:#18222d;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}section{margin:30px 0}</style>'
        '<h1>Advanced perception: synthetic accuracy checks</h1><p>All transformations are checked against known truth. '
        'This does not establish real-camera accuracy.</p>'
        '<section><h2>Shape localization</h2><p>Green bounding boxes; yellow centers. Angle/scale on the supplied grid.</p><img src="shape.png"></section>'
        '<section><h2>Local point cloud registration</h2><p>Identical scales before/after. Outliers excluded from this display only; included in estimation.</p><img src="registration.png"></section>'
        '<section><h2>Calibrated object pose</h2><p>25% deliberately wrong correspondences; rejected observations in red.</p><img src="pose.png"></section>', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
