"""Known-truth eye-in-hand robot/camera extrinsic calibration."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.handeye_ops import calibrate_eye_in_hand, calibrate_eye_in_hand_robust
from operators.image_io import write_image


def make_scene(seed=23, count=20):
    rng = np.random.default_rng(seed)
    camera_to_gripper = np.eye(4)
    camera_to_gripper[:3, :3] = cv2.Rodrigues(np.array([.08, -.12, .05]))[0]
    camera_to_gripper[:3, 3] = [.05, -.02, .08]
    target_to_base = np.eye(4)
    target_to_base[:3, :3] = cv2.Rodrigues(np.array([.03, .05, -.02]))[0]
    target_to_base[:3, 3] = [.1, -.05, 1.4]
    robot, observations = [], []
    for _ in range(count):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(rng.uniform(-.5, .5, 3))[0]
        pose[:3, 3] = rng.uniform([-.25, -.25, .1], [.25, .25, .4])
        observed = np.linalg.inv(pose @ camera_to_gripper) @ target_to_base
        observed[:3, :3] = cv2.Rodrigues(rng.normal(0, .0005, 3))[0] @ observed[:3, :3]
        observed[:3, 3] += rng.normal(0, .0005, 3)
        robot.append(pose)
        observations.append(observed)
    return dict(robot=robot, observations=observations, truth_camera=camera_to_gripper,
                truth_target=target_to_base, pose_noise_rotation_rad=.0005,
                pose_noise_translation_m=.0005)


def evaluate(data):
    result = calibrate_eye_in_hand(data['robot'], data['observations'])
    assert result['success'], result['reason']
    true_camera = data['truth_camera']
    estimated_camera = result['T_gripper_from_camera']
    angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(
        estimated_camera[:3, :3] @ true_camera[:3, :3].T)[0])))
    displacement = float(np.linalg.norm(estimated_camera[:3, 3]-true_camera[:3, 3]))
    assert angle < .1 and displacement < .002
    checks = dict(views=len(data['robot']), method=result['method'],
                  rotation_error_deg=angle, translation_error_m=displacement,
                  fixed_target_translation_rms_m=result['translation_rms'],
                  fixed_target_rotation_rms_deg=result['rotation_rms_deg'],
                  max_robot_rotation_deg=result['max_rotation_deg'],
                  motion_axis_ratio=result['motion_axis_ratio'],
                  pose_noise_rotation_rad=data['pose_noise_rotation_rad'],
                  pose_noise_translation_m=data['pose_noise_translation_m'])
    return result, checks


def corrupted_observations(data):
    damaged = [pose.copy() for pose in data['observations']]
    indices = (1, 5, 11, 17)
    for j, index in enumerate(indices):
        damaged[index][:3, 3] += [.04+.01*j, -.03, .02]
        damaged[index][:3, :3] = cv2.Rodrigues(np.array([.04, .02, -.03]))[0] @ damaged[index][:3, :3]
    return damaged, indices


def evaluate_robust(data):
    observations, deliberate_outliers = corrupted_observations(data)
    ordinary = calibrate_eye_in_hand(data['robot'], observations)
    assert not ordinary['success']
    result = calibrate_eye_in_hand_robust(data['robot'], observations, min_inliers=12, seed=42)
    assert result['success'], result['reason']
    rejected = [int(i) for i in np.flatnonzero(~result['inliers'])]
    assert rejected == list(deliberate_outliers)
    truth = data['truth_camera']
    estimate = result['T_gripper_from_camera']
    angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(
        estimate[:3, :3] @ truth[:3, :3].T)[0])))
    distance = float(np.linalg.norm(estimate[:3, 3]-truth[:3, 3]))
    assert angle < .1 and distance < .002
    checks = dict(deliberate_outlier_indices=list(deliberate_outliers), rejected_indices=rejected,
                  inlier_count=result['inlier_count'], trials_run=result['trials_run'],
                  ordinary_reason=ordinary['reason'], robust_rotation_error_deg=angle,
                  robust_translation_error_m=distance,
                  robust_target_translation_rms_m=result['translation_rms'],
                  robust_target_rotation_rms_deg=result['rotation_rms_deg'])
    return result, checks


def visual(data, result):
    canvas = np.full((600, 900, 3), 22, np.uint8)
    target = result['T_base_from_target'][:3, 3]
    locations = np.array([(robot @ result['T_gripper_from_camera'])[:3, 3]
                          for robot in data['robot']])
    for point in locations:
        x, y = np.rint(point[:2]*600+[450, 300]).astype(int)
        cv2.circle(canvas, (int(x), int(y)), 8, (60, 180, 250), -1)
    x, y = np.rint(target[:2]*600+[450, 300]).astype(int)
    cv2.drawMarker(canvas, (int(x), int(y)), (70, 220, 80), cv2.MARKER_CROSS, 22, 3)
    cv2.putText(canvas, 'Orange: camera locations   Green: fixed target', (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (230, 230, 230), 2)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/handeye')
    args = parser.parse_args()
    data = make_scene()
    result, checks = evaluate(data)
    _, robust_checks = evaluate_robust(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'robot_views.png', visual(data, result))
    report = dict(clean=checks, four_bad_captures=robust_checks)
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Eye-in-hand calibration</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Eye-in-hand calibration</h1><p>Synthetic robot poses and fixed target observations.'
        ' Independent target-pose consistency validates the output.</p><img src="robot_views.png">',
        encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
