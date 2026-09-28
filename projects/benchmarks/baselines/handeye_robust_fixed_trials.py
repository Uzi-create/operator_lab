"""Eye-in-hand calibration with robot-motion observability and pose consistency."""
import math

import cv2
import numpy as np

from operators.robot_ops import transform_points


def _transforms(values, name):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f'{name} must be a list of rigid 4x4 poses')
    output = []
    for pose in values:
        pose = np.asarray(pose, dtype=np.float64)
        transform_points(np.empty((0, 3)), pose)
        output.append(pose.copy())
    return output


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be a positive finite scalar')
    try:
        valid = math.isfinite(value) and value > 0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f'{name} must be a positive finite scalar')
    return float(value)


def _rotation_mean(rotations):
    u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    return u @ correction @ vt


def _failure(reason, count, **extra):
    return dict(success=False, reason=reason, T_gripper_from_camera=None,
                T_base_from_target=None, translation_rms=math.inf,
                rotation_rms_deg=math.inf, per_view_translation_error=np.full(count, math.inf),
                per_view_rotation_error_deg=np.full(count, math.inf), **extra)


def calibrate_eye_in_hand(T_base_from_gripper, T_camera_from_target, *,
                          min_views=6, min_rotation_deg=5., min_axis_ratio=.05,
                          max_translation_rms=.005, max_rotation_rms_deg=.3,
                          method='park'):
    """Estimate camera-to-gripper from a fixed target seen at robot poses.

    Inputs share one physical length unit. Returned target-to-base poses are
    recomputed from every original capture to check AX=XB consistency. Robot
    poses must change orientation about at least two independent axes.
    """
    gripper = _transforms(T_base_from_gripper, 'T_base_from_gripper')
    target = _transforms(T_camera_from_target, 'T_camera_from_target')
    if len(gripper) != len(target):
        raise ValueError('Robot and camera pose lists must have matching lengths')
    if type(min_views) is not int or not 3 <= min_views <= 1000:
        raise ValueError('min_views must be an integer in 3..1000')
    minimum_angle = _positive(min_rotation_deg, 'min_rotation_deg')
    if minimum_angle >= 180:
        raise ValueError('min_rotation_deg must be below 180')
    ratio = _positive(min_axis_ratio, 'min_axis_ratio')
    if ratio >= 1:
        raise ValueError('min_axis_ratio must be below 1')
    max_translation = _positive(max_translation_rms, 'max_translation_rms')
    max_rotation = _positive(max_rotation_rms_deg, 'max_rotation_rms_deg')
    methods = {'tsai': cv2.CALIB_HAND_EYE_TSAI,
               'park': cv2.CALIB_HAND_EYE_PARK,
               'horaud': cv2.CALIB_HAND_EYE_HORAUD,
               'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
               'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS}
    if method not in methods:
        raise ValueError('Unknown hand-eye method')
    count = len(gripper)
    if count < min_views:
        return _failure('insufficient_views', count)
    rotation_vectors = []
    for i in range(count):
        for j in range(i+1, count):
            relative = gripper[i][:3, :3].T @ gripper[j][:3, :3]
            rotation_vectors.append(cv2.Rodrigues(relative)[0].ravel())
    singular = np.linalg.svd(np.array(rotation_vectors), compute_uv=False)
    max_angle = float(np.degrees(max(np.linalg.norm(row) for row in rotation_vectors)))
    axis_ratio = float(singular[1]/singular[0]) if singular[0] > 0 else 0.
    if max_angle < minimum_angle or axis_ratio < ratio:
        return _failure('insufficient_motion_diversity', count,
                        max_rotation_deg=max_angle, motion_axis_ratio=axis_ratio)
    try:
        rotation, translation = cv2.calibrateHandEye(
            [pose[:3, :3] for pose in gripper], [pose[:3, 3] for pose in gripper],
            [pose[:3, :3] for pose in target], [pose[:3, 3] for pose in target],
            method=methods[method])
    except cv2.error:
        return _failure('opencv_handeye_failed', count,
                        max_rotation_deg=max_angle, motion_axis_ratio=axis_ratio)
    if (rotation is None or translation is None or rotation.shape != (3, 3) or
            translation.size != 3 or not np.isfinite(rotation).all() or
            not np.isfinite(translation).all() or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0) or
            abs(np.linalg.det(rotation)-1.) > 1e-5):
        return _failure('invalid_handeye_solution', count,
                        max_rotation_deg=max_angle, motion_axis_ratio=axis_ratio)
    camera_to_gripper = np.eye(4)
    camera_to_gripper[:3, :3], camera_to_gripper[:3, 3] = rotation, translation.ravel()
    target_to_base_samples = np.array([robot @ camera_to_gripper @ board
                                       for robot, board in zip(gripper, target)])
    center_translation = np.mean(target_to_base_samples[:, :3, 3], axis=0)
    center_rotation = _rotation_mean(target_to_base_samples[:, :3, :3])
    center = np.eye(4)
    center[:3, :3], center[:3, 3] = center_rotation, center_translation
    translation_errors = np.linalg.norm(target_to_base_samples[:, :3, 3]-center_translation, axis=1)
    rotation_errors = np.array([np.degrees(np.linalg.norm(cv2.Rodrigues(
        pose[:3, :3] @ center_rotation.T)[0])) for pose in target_to_base_samples])
    translation_rms = float(np.sqrt(np.mean(translation_errors**2)))
    rotation_rms = float(np.sqrt(np.mean(rotation_errors**2)))
    reason = ('inconsistent_target_poses' if translation_rms > max_translation or
              rotation_rms > max_rotation else 'ok')
    return dict(success=reason == 'ok', reason=reason,
                T_gripper_from_camera=camera_to_gripper, T_base_from_target=center,
                translation_rms=translation_rms, rotation_rms_deg=rotation_rms,
                per_view_translation_error=translation_errors,
                per_view_rotation_error_deg=rotation_errors,
                max_rotation_deg=max_angle, motion_axis_ratio=axis_ratio,
                method=method)


def calibrate_eye_in_hand_robust(T_base_from_gripper, T_camera_from_target, *,
                                 max_translation_error=.005, max_rotation_error_deg=.3,
                                 min_inliers=8, max_trials=128, seed=0, method='park'):
    """Reject corrupt target-pose captures before refining eye-in-hand geometry.

    Hypotheses use six synchronized captures. Inliers must independently place
    the fixed target near the same base-frame pose in translation AND rotation.
    The final solution and every reported residual come from the selected set.
    """
    gripper = _transforms(T_base_from_gripper, 'T_base_from_gripper')
    target = _transforms(T_camera_from_target, 'T_camera_from_target')
    if len(gripper) != len(target):
        raise ValueError('Robot and camera pose lists must have matching lengths')
    translation_limit = _positive(max_translation_error, 'max_translation_error')
    rotation_limit = _positive(max_rotation_error_deg, 'max_rotation_error_deg')
    if type(min_inliers) is not int or min_inliers < 6:
        raise ValueError('min_inliers must be an integer >=6')
    if type(max_trials) is not int or not 1 <= max_trials <= 10000:
        raise ValueError('max_trials must be an integer in 1..10000')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    # This also validates the selected OpenCV method and option contracts.
    initial = calibrate_eye_in_hand(gripper, target, max_translation_rms=1e100,
                                    max_rotation_rms_deg=1e100, method=method)
    count = len(gripper)
    if count < max(6, min_inliers):
        return _failure('insufficient_views', count, inliers=np.zeros(count, bool),
                        inlier_count=0, trials_run=0)

    def evaluate(candidate):
        camera = candidate['T_gripper_from_camera']
        board = candidate['T_base_from_target']
        samples = [robot @ camera @ observation for robot, observation in zip(gripper, target)]
        t_error = np.array([np.linalg.norm(sample[:3, 3]-board[:3, 3]) for sample in samples])
        r_error = np.array([np.degrees(np.linalg.norm(cv2.Rodrigues(
            sample[:3, :3] @ board[:3, :3].T)[0])) for sample in samples])
        mask = (np.isfinite(t_error) & np.isfinite(r_error) &
                (t_error <= translation_limit) & (r_error <= rotation_limit))
        return mask, t_error, r_error

    best = None
    trials_run = 0
    rng = np.random.default_rng(seed)
    hypotheses = [initial] if initial['T_gripper_from_camera'] is not None else []
    for trial in range(max_trials):
        if hypotheses:
            candidate = hypotheses.pop()
        else:
            indices = np.sort(rng.choice(count, 6, replace=False))
            candidate = calibrate_eye_in_hand([gripper[i] for i in indices],
                                              [target[i] for i in indices], min_views=6,
                                              max_translation_rms=1e100,
                                              max_rotation_rms_deg=1e100, method=method)
        trials_run += 1
        if candidate['T_gripper_from_camera'] is None:
            continue
        mask, t_error, r_error = evaluate(candidate)
        number = int(mask.sum())
        score = (number, -float(np.mean((t_error[mask]/translation_limit)**2+
                                       (r_error[mask]/rotation_limit)**2))) if number else (0, -math.inf)
        if best is None or score > best[0]:
            best = (score, mask)
        if number == count:
            break
    if best is None or best[0][0] < min_inliers:
        return _failure('insufficient_consensus', count, inliers=np.zeros(count, bool),
                        inlier_count=0, trials_run=trials_run)
    mask = best[1]
    refined = None
    for _ in range(4):
        selected = np.flatnonzero(mask)
        if len(selected) < min_inliers:
            break
        refined = calibrate_eye_in_hand([gripper[i] for i in selected],
                                        [target[i] for i in selected], min_views=6,
                                        max_translation_rms=translation_limit,
                                        max_rotation_rms_deg=rotation_limit, method=method)
        if refined['T_gripper_from_camera'] is None:
            break
        updated, t_error, r_error = evaluate(refined)
        if np.array_equal(updated, mask):
            break
        mask = updated
    if refined is None or not refined['success'] or not np.array_equal(updated, mask):
        return _failure('refinement_failed', count, inliers=np.zeros(count, bool),
                        inlier_count=0, trials_run=trials_run)
    return {**refined, 'inliers': mask, 'inlier_count': int(mask.sum()),
            'per_view_translation_error': t_error,
            'per_view_rotation_error_deg': r_error, 'trials_run': trials_run}
