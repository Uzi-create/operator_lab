"""Calibrated object-to-camera pose from known 3D/2D correspondences.

Translation has the SAME units as object_points. Camera optical coordinates are
+X right, +Y down, +Z forward. This module never estimates correspondences or
calibration. OpenCV's standard distortion model is supported, not fisheye.
"""
import math

import cv2
import numpy as np

from operators.robot_ops import Intrinsics, transform_points


def _array(values, columns, name):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != columns or values.dtype.kind not in 'uif':
        raise ValueError(f'{name} must be a real Nx{columns} array')
    values = np.ascontiguousarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1e100):
        raise ValueError(f'{name} must be finite with magnitude at most 1e100')
    return values


def _calibration(camera, distortion):
    if not isinstance(camera, Intrinsics):
        raise TypeError('camera must be operators.robot_ops.Intrinsics')
    matrix = np.array([[camera.fx, 0., camera.cx], [0., camera.fy, camera.cy], [0., 0., 1.]])
    if distortion is None:
        return matrix, None
    distortion = np.asarray(distortion)
    if (distortion.ndim != 1 or len(distortion) not in (4, 5, 8, 12, 14)
            or distortion.dtype.kind not in 'uif' or not np.isfinite(distortion).all()):
        raise ValueError('distortion must be a finite vector with 4, 5, 8, 12 or 14 coefficients')
    return matrix, np.ascontiguousarray(distortion, dtype=np.float64)


def _positive(value, name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be a finite real scalar')
    try:
        valid = math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')


def _failure(reason, count, **extra):
    return {'success': False, 'reason': reason, 'T_camera_from_object': None,
            'inliers': np.zeros(count, dtype=bool), 'inlier_count': 0,
            'rms': math.inf, 'reprojection_errors': np.full(count, np.inf),
            'planar': False, 'ambiguous': False, 'candidates': [], **extra}


def project_object_points(object_points, camera, T_camera_from_object, *, distortion=None):
    """Project a supplied pose with distortion; pixels behind camera are NaN.

    Unlike robot_ops.project_points this accepts object coordinates, an explicit
    pose and OpenCV distortion. It does not clip to the image or test occlusion.
    Use distortion=None for already-undistorted pixels with matching intrinsics.
    """
    points = _array(object_points, 3, 'object_points')
    matrix, distortion = _calibration(camera, distortion)
    transformed = transform_points(points, T_camera_from_object)
    positive = transformed[:, 2] > 0
    pixels = np.full((len(points), 2), np.nan)
    if positive.any():
        pixels[positive] = cv2.projectPoints(
            transformed[positive], np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
    return {'pixels': pixels, 'depth': transformed[:, 2],
            'in_front': positive & np.isfinite(pixels).all(axis=1)}


def _geometry(points):
    center = points.mean(axis=0)
    centered = points - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    return center, singular, vt


def _evaluate(points, pixels, matrix, distortion, rotation, translation, threshold, min_depth):
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 1)
    translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return None
    try:
        projected = cv2.projectPoints(points, rotation, translation, matrix, distortion)[0].reshape(-1, 2)
        rotation_matrix = cv2.Rodrigues(rotation)[0]
    except cv2.error:
        return None
    depth = points @ rotation_matrix[2] + translation[2, 0]
    errors = np.linalg.norm(projected - pixels, axis=1)
    inliers = np.isfinite(errors) & (errors <= threshold) & (depth > min_depth)
    rms = float(np.sqrt(np.mean(errors[inliers] ** 2))) if inliers.any() else math.inf
    return {'rvec': rotation, 'tvec': translation, 'rotation': rotation_matrix,
            'inliers': inliers, 'inlier_count': int(inliers.sum()), 'rms': rms,
            'reprojection_errors': errors, 'depth': depth}


def _refine(points, pixels, matrix, distortion, pose, threshold, min_depth):
    # Reclassify after each fit; never return solvePnPRansac's stale inlier list.
    current = pose
    for _ in range(3):
        mask = current['inliers']
        if mask.sum() < 6:
            break
        try:
            rotation, translation = cv2.solvePnPRefineLM(
                points[mask], pixels[mask], matrix, distortion,
                current['rvec'].copy(), current['tvec'].copy(),
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-10))
        except cv2.error:
            break
        updated = _evaluate(points, pixels, matrix, distortion, rotation, translation,
                            threshold, min_depth)
        if updated is None:
            break
        # Minimize residuals without sacrificing established consensus support.
        if (updated['inlier_count'], -updated['rms']) < (current['inlier_count'], -current['rms']):
            break
        current = updated
        if np.array_equal(mask, current['inliers']):
            break
    return current


def _distinct(a, b):
    cosine = (np.trace(a['rotation'] @ b['rotation'].T) - 1.) / 2.
    angle = math.acos(float(np.clip(cosine, -1., 1.)))
    return angle > math.radians(.1) or np.linalg.norm(a['tvec'] - b['tvec']) > .001


def estimate_pose_pnp(object_points, image_points, camera, *, distortion=None,
                      reprojection_threshold=3., max_rms=2., min_inliers=6,
                      min_inlier_ratio=.5, max_iterations=1000, confidence=.999,
                      min_depth=0., ambiguity_rms_delta=.25, allow_ambiguous=False):
    """Robust PnP, final-pose residual checks and explicit planar ambiguity.

    At least six correspondences are required; this deliberately is not a
    minimal four-corner marker solver. Thresholds/RMS are in input pixels;
    min_depth and translation have object-point units. ``distortion=None`` means
    a pinhole camera (normally already-undistorted pixels). Otherwise pass raw
    distorted pixels and OpenCV coefficients (k1,k2,p1,p2,k3,...).

    Malformed inputs raise ValueError/TypeError. Insufficient support, geometric
    degeneracy and failed estimation return success=False with a reason.
    Returned inliers and RMS ALWAYS use the returned final pose, positive depth,
    and reprojection_threshold. No unique pose is promised from residual alone.

    For a planar final consensus, IPPE's two poses are evaluated as well. Distinct
    poses explaining the same consensus within ambiguity_rms_delta pixels cause
    failure by default; inspect candidates or explicitly allow_ambiguous=True.
    This is a diagnostic, not calibrated pose uncertainty or a uniqueness proof.
    """
    points = _array(object_points, 3, 'object_points')
    pixels = _array(image_points, 2, 'image_points')
    if len(points) != len(pixels):
        raise ValueError('object_points and image_points must have equal length')
    matrix, distortion = _calibration(camera, distortion)
    for name, value in [('reprojection_threshold', reprojection_threshold),
                        ('max_rms', max_rms), ('min_inlier_ratio', min_inlier_ratio),
                        ('confidence', confidence)]:
        _positive(value, name)
    _positive(min_depth, 'min_depth', True)
    _positive(ambiguity_rms_delta, 'ambiguity_rms_delta', True)
    if type(min_inliers) is not int or min_inliers < 6:
        raise ValueError('min_inliers must be an integer >= 6')
    if type(max_iterations) is not int or not 1 <= max_iterations <= 1_000_000:
        raise ValueError('max_iterations must be an integer in 1..1000000')
    if min_inlier_ratio > 1 or confidence >= 1:
        raise ValueError('min_inlier_ratio must be <= 1 and confidence < 1')
    if type(allow_ambiguous) is not bool:
        raise ValueError('allow_ambiguous must be bool')
    count = len(points)
    required = max(min_inliers, math.ceil(count * min_inlier_ratio))
    if count < required:
        return _failure('insufficient_correspondences', count)
    center, singular, _ = _geometry(points)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-8:
        return _failure('degenerate_object_geometry', count)
    _, image_singular, _ = _geometry(pixels)
    if image_singular[0] <= 0 or image_singular[1] <= image_singular[0] * 1e-8:
        return _failure('degenerate_image_geometry', count)
    scale = float(np.linalg.norm(singular) / math.sqrt(count))
    normalized = np.ascontiguousarray((points - center) / scale)
    normalized_min_depth = min_depth / scale
    try:
        success, rotation, translation, _ = cv2.solvePnPRansac(
            normalized, pixels, matrix, distortion, iterationsCount=max_iterations,
            reprojectionError=float(reprojection_threshold), confidence=float(confidence),
            flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return _failure('ransac_failed', count)
    if not success:
        return _failure('ransac_failed', count)
    best = _evaluate(normalized, pixels, matrix, distortion, rotation, translation,
                     reprojection_threshold, normalized_min_depth)
    if best is None or best['inlier_count'] < required:
        return _failure('insufficient_positive_depth_inliers', count)
    best = _refine(normalized, pixels, matrix, distortion, best, reprojection_threshold,
                   normalized_min_depth)
    mask = best['inliers']
    plane_center, plane_singular, plane_vt = _geometry(normalized[mask])
    if plane_singular[1] <= plane_singular[0] * 1e-8:
        return _failure('degenerate_inlier_geometry', count)
    _, inlier_image_singular, _ = _geometry(pixels[mask])
    if inlier_image_singular[1] <= inlier_image_singular[0] * 1e-8:
        return _failure('degenerate_inlier_image_geometry', count)
    planar = bool(plane_singular[2] <= plane_singular[0] * 1e-6)
    candidates = [best]
    if planar:
        # Make an explicit right-handed plane frame; IPPE requires z == 0.
        basis = np.column_stack((plane_vt[0], plane_vt[1], np.cross(plane_vt[0], plane_vt[1])))
        plane_points = np.ascontiguousarray((normalized[mask] - plane_center) @ basis)
        plane_points[:, 2] = 0.
        try:
            solved = cv2.solvePnPGeneric(plane_points, pixels[mask], matrix, distortion,
                                        flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return _failure('planar_ambiguity_check_failed', count, planar=True)
        if not solved[0] or len(solved[1]) < 2:
            return _failure('planar_ambiguity_check_failed', count, planar=True)
        for rv, tv in zip(solved[1], solved[2]):
            if not np.isfinite(rv).all() or not np.isfinite(tv).all():
                return _failure('planar_ambiguity_check_failed', count, planar=True)
            rm = cv2.Rodrigues(rv)[0] @ basis.T
            rv = cv2.Rodrigues(rm)[0]
            tv = tv.reshape(3) - rm @ plane_center
            candidate = _evaluate(normalized, pixels, matrix, distortion, rv, tv,
                                  reprojection_threshold, normalized_min_depth)
            if candidate is not None and candidate['inlier_count'] >= required:
                # Keep a raw second solution too: LM can collapse both IPPE
                # seeds into one solution while a competing pose still fits.
                candidates.append(candidate)
                candidates.append(_refine(normalized, pixels, matrix, distortion, candidate,
                                          reprojection_threshold, normalized_min_depth))
    candidates.sort(key=lambda c: (-c['inlier_count'], c['rms']))
    best = candidates[0]
    unique = [best]
    for candidate in candidates[1:]:
        if all(_distinct(candidate, known) for known in unique):
            unique.append(candidate)
    public_candidates = []
    candidate_depths = []
    for candidate in unique:
        transform = np.eye(4)
        transform[:3, :3] = candidate['rotation']
        transform[:3, 3] = scale * candidate['tvec'].ravel() - candidate['rotation'] @ center
        # Recompute using the actual returned pose, including normalization
        # roundoff, so externally evaluated residuals agree with this contract.
        final_rv = cv2.Rodrigues(transform[:3, :3])[0]
        final = _evaluate(points, pixels, matrix, distortion, final_rv, transform[:3, 3],
                          reprojection_threshold, min_depth)
        if final is None:
            continue
        public_candidates.append({
            'T_camera_from_object': transform, 'inliers': final['inliers'],
            'inlier_count': final['inlier_count'], 'rms': final['rms'],
            'reprojection_errors': final['reprojection_errors'],
            'positive_depth_fraction': float(np.mean(final['depth'] > min_depth))})
        candidate_depths.append(final['depth'])
    if not public_candidates:
        return _failure('invalid_final_pose', count, planar=planar)
    # Rank again after conversion to the original object coordinate system.
    # This also makes the ambiguity comparison use exactly the returned pose.
    order = sorted(range(len(public_candidates)),
                   key=lambda i: (-public_candidates[i]['inlier_count'], public_candidates[i]['rms']))
    public_candidates = [public_candidates[i] for i in order]
    candidate_depths = [candidate_depths[i] for i in order]
    best_public = public_candidates[0]
    reference = best_public['inliers']
    ambiguous = False
    for index, (candidate, depth) in enumerate(zip(public_candidates, candidate_depths)):
        reference_rms = (float(np.sqrt(np.mean(candidate['reprojection_errors'][reference] ** 2)))
                         if reference.any() else math.inf)
        candidate['reference_rms'] = reference_rms
        supported = bool((depth[reference] > min_depth).all() and np.isfinite(reference_rms))
        if index and supported and reference_rms <= best_public['rms'] + ambiguity_rms_delta:
            ambiguous = True
    if best_public['inlier_count'] < required or best_public['rms'] > max_rms:
        return _failure('final_pose_quality_failed', count, planar=planar, candidates=public_candidates)
    if ambiguous and not allow_ambiguous:
        return _failure('ambiguous_planar_pose', count, planar=planar, ambiguous=True,
                        candidates=public_candidates)
    return {**best_public, 'success': True,
            'reason': 'ambiguous_planar_pose_allowed' if ambiguous else 'ok',
            'planar': planar, 'ambiguous': ambiguous, 'candidates': public_candidates}
