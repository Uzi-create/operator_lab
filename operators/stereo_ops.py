"""Distortion-aware calibrated stereo triangulation with geometric diagnostics."""
import math

import cv2
import numpy as np

from operators.pose_ops import _calibration
from operators.ray_plane_ops import _pixels, _scalar, camera_rays
from operators.robot_ops import transform_points


def triangulate_stereo(pixels_left, pixels_right, left_camera, right_camera,
                       T_right_from_left, *, distortion_left=None, distortion_right=None,
                       min_parallax_deg=1., max_reprojection_error_px=1.,
                       max_ray_distance=math.inf, max_ray_inverse_error_px=1e-6,
                       method='midpoint'):
    """Triangulate corresponding raw pixels into the left optical frame.

    The rigid transform's translation and output points share the same metric
    unit. Both cameras use OpenCV's standard distortion model, not fisheye.
    Invalid correspondences remain in place with NaN points and a reason code.
    This routine checks geometry, but does not discover matches or occlusions.
    """
    left = _pixels(pixels_left)
    right = _pixels(pixels_right)
    if left.shape != right.shape:
        raise ValueError('left and right pixels must have matching Nx2 shapes')
    matrix_left, distortion_left = _calibration(left_camera, distortion_left)
    matrix_right, distortion_right = _calibration(right_camera, distortion_right)
    minimum = _scalar(min_parallax_deg, 'min_parallax_deg', zero=True)
    if minimum >= 90:
        raise ValueError('min_parallax_deg must be <90')
    threshold = _scalar(max_reprojection_error_px, 'max_reprojection_error_px')
    maximum = _scalar(max_ray_distance, 'max_ray_distance', infinity=True)
    inverse_tolerance = _scalar(max_ray_inverse_error_px, 'max_ray_inverse_error_px')
    if method not in ('midpoint', 'dlt'):
        raise ValueError('method must be midpoint or dlt')
    transform = np.asarray(T_right_from_left, dtype=np.float64)
    transform_points(np.empty((0, 3)), transform)  # enforce rigid transform contract
    rotation, translation = transform[:3, :3], transform[:3, 3]
    origin_right_left = -rotation.T @ translation
    if np.linalg.norm(origin_right_left) <= 1e-12:
        raise ValueError('Stereo cameras need a nonzero baseline')
    count = len(left)
    output = np.full((count, 3), np.nan)
    reprojection = np.full((count, 2), np.inf)
    parallax = np.full(count, np.nan)
    depth_left = np.full(count, np.nan)
    depth_right = np.full(count, np.nan)
    reasons = np.full(count, 'ok', dtype='<U28')
    inside = ((left[:, 0] >= 0) & (left[:, 0] <= left_camera.width-1) &
              (left[:, 1] >= 0) & (left[:, 1] <= left_camera.height-1) &
              (right[:, 0] >= 0) & (right[:, 0] <= right_camera.width-1) &
              (right[:, 1] >= 0) & (right[:, 1] <= right_camera.height-1))
    reasons[~inside] = 'outside_image'
    if not count:
        return dict(points_left=output, valid=np.empty(0, bool), reasons=reasons,
                    reprojection_errors_px=reprojection, parallax_deg=parallax,
                    depth_left=depth_left, depth_right=depth_right, valid_count=0)
    rays_left = camera_rays(left, left_camera, distortion=distortion_left,
                            max_reprojection_error_px=inverse_tolerance)
    rays_right = camera_rays(right, right_camera, distortion=distortion_right,
                             max_reprojection_error_px=inverse_tolerance)
    ray_ok = rays_left['valid'] & rays_right['valid']
    reasons[inside & ~ray_ok] = 'distortion_inverse_failed'
    index = np.flatnonzero(inside & ray_ok)
    if len(index):
        direction_left = rays_left['rays_camera'][index]
        direction_right = rays_right['rays_camera'][index] @ rotation
        cosine = np.einsum('ij,ij->i', direction_left, direction_right)
        parallax[index] = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
        enough = np.sqrt(np.maximum(0., 1.-cosine**2)) >= math.sin(math.radians(minimum))
        enough &= 1.-cosine**2 > 1e-12
        reasons[index[~enough]] = 'low_parallax'
        index = index[enough]
    if len(index):
        if method == 'midpoint':
            left_direction = rays_left['rays_camera'][index]
            right_direction = rays_right['rays_camera'][index] @ rotation
            cosines = np.einsum('ij,ij->i', left_direction, right_direction)
            left_dot = left_direction @ origin_right_left
            right_dot = right_direction @ origin_right_left
            denominator = 1.-cosines**2
            left_length = (left_dot-cosines*right_dot)/denominator
            right_length = (cosines*left_dot-right_dot)/denominator
            all_candidates = .5*(left_length[:, None]*left_direction+
                                  origin_right_left+right_length[:, None]*right_direction)
            finite = np.isfinite(all_candidates).all(axis=1)
        else:
            normalized_left = rays_left['rays_camera'][index, :2] / rays_left['rays_camera'][index, 2, None]
            normalized_right = rays_right['rays_camera'][index, :2] / rays_right['rays_camera'][index, 2, None]
            projection_left = np.column_stack((np.eye(3), np.zeros(3)))
            projection_right = transform[:3, :]
            homogeneous = cv2.triangulatePoints(projection_left, projection_right,
                                                normalized_left.T.copy(), normalized_right.T.copy()).T
            finite = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 3]) > 1e-12)
            all_candidates = np.full((len(index), 3), np.nan)
            all_candidates[finite] = homogeneous[finite, :3] / homogeneous[finite, 3, None]
        reasons[index[~finite]] = 'triangulation_failed'
        index = index[finite]
        candidate = all_candidates[finite]
        candidate_right = candidate @ rotation.T + translation
        depth_left[index] = candidate[:, 2]
        depth_right[index] = candidate_right[:, 2]
        distance_left = np.linalg.norm(candidate, axis=1)
        distance_right = np.linalg.norm(candidate_right, axis=1)
        forward = ((depth_left[index] > 0) & (depth_right[index] > 0) &
                   (distance_left <= maximum) & (distance_right <= maximum) &
                   np.isfinite(candidate).all(axis=1))
        reasons[index[~forward]] = 'behind_or_too_far'
        index = index[forward]
        candidate = candidate[forward]
        candidate_right = candidate_right[forward]
        if len(index):
            projected_left = cv2.projectPoints(candidate, np.zeros(3), np.zeros(3),
                                               matrix_left, distortion_left)[0].reshape(-1, 2)
            projected_right = cv2.projectPoints(candidate_right, np.zeros(3), np.zeros(3),
                                                matrix_right, distortion_right)[0].reshape(-1, 2)
            reprojection[index, 0] = np.linalg.norm(projected_left-left[index], axis=1)
            reprojection[index, 1] = np.linalg.norm(projected_right-right[index], axis=1)
            good = np.isfinite(reprojection[index]).all(axis=1) & (reprojection[index].max(axis=1) <= threshold)
            reasons[index[~good]] = 'reprojection_failed'
            output[index[good]] = candidate[good]
    valid = reasons == 'ok'
    return dict(points_left=output, valid=valid, reasons=reasons,
                reprojection_errors_px=reprojection, parallax_deg=parallax,
                depth_left=depth_left, depth_right=depth_right, valid_count=int(valid.sum()))
