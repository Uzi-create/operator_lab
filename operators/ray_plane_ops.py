"""Calibrated pixel rays and metric intersections with a known 3D plane."""
import math
from numbers import Real

import cv2
import numpy as np

from operators.pose_ops import _calibration
from operators.robot_ops import transform_points


def _scalar(value, name, *, zero=False, infinity=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real scalar')
    value = float(value)
    if (math.isnan(value) or (not infinity and not math.isfinite(value))
            or (value < 0 if zero else value <= 0)):
        raise ValueError(f'{name} must be {"nonnegative" if zero else "positive"}')
    return value


def _pixels(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind not in 'uif':
        raise ValueError('pixels must be a real Nx2 array')
    values = np.ascontiguousarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1e9):
        raise ValueError('pixels must be finite with magnitude <=1e9')
    return values


def _signed_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def camera_rays(pixels, camera, *, distortion=None, max_reprojection_error_px=1e-6):
    """Return unit rays in the optical frame, validating distortion inversion.

    Input pixel centers are (u,v), +X right and +Y down. Distortion uses the
    OpenCV standard model, not fisheye. Pixels may lie outside the image; no
    visibility or occlusion check is implied. Invalid inversions have NaN rays.
    """
    pixels = _pixels(pixels)
    matrix, distortion = _calibration(camera, distortion)
    tolerance = _scalar(max_reprojection_error_px, 'max_reprojection_error_px')
    if not len(pixels):
        return {'rays_camera': np.empty((0, 3)), 'valid': np.empty(0, bool),
                'reprojection_errors_px': np.empty(0)}
    if distortion is None:
        xy = np.column_stack(((pixels[:, 0]-camera.cx)/camera.fx,
                              (pixels[:, 1]-camera.cy)/camera.fy))
        reprojection_errors = np.zeros(len(pixels))
    else:
        try:
            # Most ordinary lens rays converge in a few iterations. Verify by
            # forward projection and retry only unresolved rays with 50 steps.
            criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
            xy = cv2.undistortPointsIter(pixels.reshape(-1, 1, 2), matrix, distortion,
                                          None, None, (criteria, 8, 1e-13)).reshape(-1, 2)
            reprojected = cv2.projectPoints(np.column_stack((xy, np.ones(len(xy)))),
                                            np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
            reprojection_errors = np.linalg.norm(reprojected-pixels, axis=1)
            retry = ~np.isfinite(reprojection_errors) | (reprojection_errors > tolerance)
            if retry.any():
                refined = cv2.undistortPointsIter(pixels[retry].reshape(-1, 1, 2), matrix,
                                                   distortion, None, None,
                                                   (criteria, 50, 1e-13)).reshape(-1, 2)
                xy[retry] = refined
                reprojected = cv2.projectPoints(np.column_stack((refined, np.ones(len(refined)))),
                                                np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
                reprojection_errors[retry] = np.linalg.norm(reprojected-pixels[retry], axis=1)
        except cv2.error as error:
            raise ValueError('OpenCV could not invert the supplied distortion') from error
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        rays = np.column_stack((xy, np.ones(len(xy))))
        rays /= np.linalg.norm(rays, axis=1)[:, None]
    valid = np.isfinite(rays).all(axis=1) & np.isfinite(reprojection_errors) & (reprojection_errors <= tolerance)
    rays[~valid] = np.nan
    return {'rays_camera': rays, 'valid': valid, 'reprojection_errors_px': reprojection_errors}


def intersect_image_plane(pixels, camera, plane_normal, plane_offset, *,
                          T_frame_from_camera=None, distortion=None,
                          max_ray_distance=math.inf, min_incidence_cos=1e-3,
                          max_reprojection_error_px=1e-6):
    """Intersect calibrated pixels with n·p+offset=0 in a chosen metric frame.

    Plane, returned points and max_ray_distance share units. If no transform is
    given, the plane is in the optical camera frame. Positive distances alone
    are accepted. Grazing rays with |unit_normal·unit_ray| below the given
    cosine are rejected because tiny pixel errors can then move the point far.
    This is geometry on a supplied plane, not a measured depth or occlusion test.
    """
    normal = np.asarray(plane_normal)
    if normal.shape != (3,) or normal.dtype.kind not in 'uif':
        raise ValueError('plane_normal must be a real three-vector')
    normal = np.asarray(normal, np.float64)
    length = float(np.linalg.norm(normal))
    if not np.isfinite(normal).all() or not math.isfinite(length) or length <= 1e-12:
        raise ValueError('plane_normal must be finite and nonzero')
    offset = _signed_scalar(plane_offset, 'plane_offset')
    max_ray_distance = _scalar(max_ray_distance, 'max_ray_distance', infinity=True)
    min_incidence_cos = _scalar(min_incidence_cos, 'min_incidence_cos', zero=True)
    if min_incidence_cos > 1:
        raise ValueError('min_incidence_cos must be in [0,1]')
    normal /= length
    offset /= length
    rays_result = camera_rays(pixels, camera, distortion=distortion,
                              max_reprojection_error_px=max_reprojection_error_px)
    rays = rays_result['rays_camera']
    origin = np.zeros(3)
    if T_frame_from_camera is not None:
        origin = transform_points(np.zeros((1, 3)), T_frame_from_camera)[0]
        rotated = transform_points(np.where(np.isfinite(rays), rays, 0.), T_frame_from_camera)-origin
    else:
        rotated = rays
    denominator = rotated @ normal
    numerator = -(float(origin @ normal)+offset)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        distances = numerator/denominator
    incidence = np.abs(denominator)
    valid = (rays_result['valid'] & np.isfinite(distances) &
             (distances > 0) & (distances <= max_ray_distance) &
             (incidence >= min_incidence_cos))
    points = np.full((len(rays), 3), np.nan)
    points[valid] = origin+distances[valid, None]*rotated[valid]
    valid &= np.isfinite(points).all(axis=1)
    points[~valid] = np.nan
    distances[~valid] = np.nan
    incidence[~rays_result['valid']] = np.nan
    return {'points_frame': points, 'ray_distances': distances, 'incidence_cos': incidence,
            'valid': valid, 'rays_camera': rays, 'reprojection_errors_px': rays_result['reprojection_errors_px']}
