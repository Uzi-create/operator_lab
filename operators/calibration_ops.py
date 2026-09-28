"""Planar checkerboard detection and multi-view pinhole camera calibration."""
import math
from numbers import Real

import cv2
import numpy as np

from operators.robot_ops import Intrinsics


def _positive(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real scalar')
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
    return value


def detect_chessboard_corners(image, pattern_size, *, exhaustive=False):
    """Find all inner corners, returned in row-major (column,row) order.

    pattern_size is (inner_columns, inner_rows), NOT the number of squares.
    OpenCV's SB detector already returns subpixel corners. No board identity,
    physical square size or camera calibration is inferred from one image.
    """
    if (not isinstance(pattern_size, (tuple, list)) or len(pattern_size) != 2 or
            any(type(v) is not int or v < 3 or v > 100 for v in pattern_size)):
        raise ValueError('pattern_size must be 3..100 inner columns and rows')
    if type(exhaustive) is not bool:
        raise ValueError('exhaustive must be bool')
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError('image must be a nonempty grayscale or BGR array')
    if image.ndim == 3 and (image.shape[2] != 3 or image.dtype != np.uint8):
        raise ValueError('BGR images must be uint8 with three channels')
    if image.ndim == 2 and image.dtype not in (np.uint8, np.float32, np.float64):
        raise ValueError('Grayscale image must be uint8 or float32/64 in [0,1]')
    if image.dtype != np.uint8:
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Grayscale floats must be finite in [0,1]')
        gray = np.rint(image*255).astype(np.uint8)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    if exhaustive:
        flags |= cv2.CALIB_CB_EXHAUSTIVE
    found, corners = cv2.findChessboardCornersSB(np.ascontiguousarray(gray),
                                                 tuple(pattern_size), flags=flags)
    return {'success': bool(found), 'reason': 'ok' if found else 'pattern_not_found',
            'corners_xy': (corners.reshape(-1, 2).astype(np.float64) if found else np.empty((0, 2))),
            'pattern_size': tuple(pattern_size)}


def _failure(reason, views, **extra):
    return {'success': False, 'reason': reason, 'camera': None, 'distortion': None,
            'rms_px': math.inf, 'per_view_rms_px': np.full(views, math.inf),
            'normal_spread_deg': None, 'T_camera_from_board': [], **extra}


def calibrate_planar_camera(object_points, image_points, image_size, *,
                            min_views=8, max_rms_px=1., max_view_rms_px=2.,
                            min_normal_spread_deg=8., distortion_model='standard5'):
    """Calibrate from repeated views of one known flat pattern in z=0 frame.

    All image_points arrays must correspond one-to-one with the same Nx3
    object_points. The pattern dimensions give translation physical units.
    Returns calibration plus independent per-view reprojection diagnostics.
    Diverse board tilts are required; low residual alone is insufficient.
    """
    points = np.asarray(object_points)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'uif':
        raise ValueError('object_points must be a real Nx3 array')
    points = np.ascontiguousarray(points, dtype=np.float64)
    if len(points) < 9 or not np.isfinite(points).all() or np.any(abs(points) > 1e9):
        raise ValueError('At least 9 finite metric object points required')
    scale = float(np.linalg.norm(np.ptp(points[:, :2], axis=0)))
    if scale <= 0 or np.max(abs(points[:, 2])) > max(1e-12, scale*1e-9):
        raise ValueError('Planar pattern coordinates must have z=0')
    if np.linalg.svd(points[:, :2]-points[:, :2].mean(axis=0), compute_uv=False)[-1] < scale*1e-6:
        raise ValueError('Planar pattern points must span two dimensions')
    if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2 or
            any(type(v) is not int or v < 16 or v > 32767 for v in image_size)):
        raise ValueError('image_size must be (width,height), each 16..32767')
    width, height = image_size
    if type(min_views) is not int or not 3 <= min_views <= 1000:
        raise ValueError('min_views must be an integer in 3..1000')
    max_rms_px = _positive(max_rms_px, 'max_rms_px')
    max_view_rms_px = _positive(max_view_rms_px, 'max_view_rms_px')
    min_normal_spread_deg = _positive(min_normal_spread_deg, 'min_normal_spread_deg', zero=True)
    if min_normal_spread_deg > 90:
        raise ValueError('min_normal_spread_deg must be <=90')
    if distortion_model not in ('standard5', 'rational8'):
        raise ValueError('distortion_model must be standard5 or rational8')
    if not isinstance(image_points, (tuple, list)):
        raise ValueError('image_points must be a list of Nx2 views')
    views = []
    for pixels in image_points:
        pixels = np.asarray(pixels)
        if pixels.shape != (len(points), 2) or pixels.dtype.kind not in 'uif':
            raise ValueError('Each view must have real Nx2 points matching object_points')
        pixels = np.ascontiguousarray(pixels, dtype=np.float64)
        if (not np.isfinite(pixels).all() or (pixels[:, 0] < 0).any() or
                (pixels[:, 0] > width-1).any() or (pixels[:, 1] < 0).any() or
                (pixels[:, 1] > height-1).any()):
            raise ValueError('Observed pattern corners must lie within the image')
        views.append(pixels)
    if len(views) < min_views:
        return _failure('insufficient_views', len(views))
    flags = cv2.CALIB_RATIONAL_MODEL if distortion_model == 'rational8' else 0
    try:
        output = cv2.calibrateCameraExtended(
            [points.astype(np.float32)]*len(views),
            [v.astype(np.float32) for v in views], (width, height), None, None,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12))
    except cv2.error:
        return _failure('opencv_calibration_failed', len(views))
    _, matrix, distortion, rotations, translations, intrinsic_std, _, _ = output
    distortion = distortion.ravel()[:8 if distortion_model == 'rational8' else 5].copy()
    if (not np.isfinite(matrix).all() or not np.isfinite(distortion).all() or
            matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        return _failure('invalid_calibration', len(views))
    camera = Intrinsics(width, height, float(matrix[0, 0]), float(matrix[1, 1]),
                        float(matrix[0, 2]), float(matrix[1, 2]))
    errors = []
    normals = []
    transforms = []
    valid_depth = True
    for pixels, rotation, translation in zip(views, rotations, translations):
        projected = cv2.projectPoints(points, rotation, translation, matrix, distortion)[0].reshape(-1, 2)
        distances = np.linalg.norm(projected-pixels, axis=1)
        errors.append(distances)
        R = cv2.Rodrigues(rotation)[0]
        t = translation.ravel()
        valid_depth &= bool(((points@R[2]+t[2]) > 0).all())
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R, t
        transforms.append(pose)
        normals.append(R[:, 2])
    per_view = np.array([np.sqrt(np.mean(row**2)) for row in errors])
    rms = float(np.sqrt(np.mean(np.concatenate(errors)**2)))
    normals = np.array(normals)
    spread = math.degrees(math.acos(float(np.clip(np.min(normals@normals.T), -1, 1))))
    reason = ('invalid_positive_depth' if not valid_depth else
              'insufficient_view_tilt_diversity' if spread < min_normal_spread_deg else
              'reprojection_quality_failed' if rms > max_rms_px or per_view.max() > max_view_rms_px else 'ok')
    return {'success': reason == 'ok', 'reason': reason, 'camera': camera,
            'distortion': distortion, 'rms_px': rms,
            'per_view_rms_px': per_view, 'normal_spread_deg': spread,
            'T_camera_from_board': transforms,
            'intrinsic_std': intrinsic_std.ravel()[:4].astype(np.float64),
            'distortion_std': intrinsic_std.ravel()[4:4+len(distortion)].astype(np.float64),
            'distortion_model': distortion_model}
