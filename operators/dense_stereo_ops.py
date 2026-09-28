"""Rectified stereo disparity and metric depth using OpenCV's C++ SGBM."""
import math

import cv2
import numpy as np

from operators.ray_plane_ops import _scalar
from operators.robot_ops import Intrinsics


def _gray(image, name):
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError(f'{name} must be a nonempty grayscale or BGR image')
    if image.ndim == 3:
        if image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f'{name} BGR image must be uint8 with three channels')
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if image.dtype not in (np.float32, np.float64) or not np.isfinite(image).all():
        raise ValueError(f'{name} grayscale image must be uint8 or finite float in [0,1]')
    if image.min() < 0 or image.max() > 1:
        raise ValueError(f'{name} grayscale float values must lie in [0,1]')
    return np.ascontiguousarray(np.rint(image*255).astype(np.uint8))


def dense_stereo_depth(left_image, right_image, camera, baseline, *,
                       num_disparities=64, min_disparity=0, block_size=5,
                       uniqueness_ratio=10, speckle_window_size=50,
                       speckle_range=2, left_right_check=True,
                       left_right_tolerance_px=1., min_depth=.05,
                       max_depth=math.inf):
    """Compute depth from already rectified, synchronized left/right images.

    Positive disparity is u_left-u_right; the right camera center lies on the
    positive X side of the left camera. Baseline and depth share length units.
    Invalid pixels are NaN in disparity_px/depth, and raw_disparity_px retains
    the C++ matcher output for diagnostics. No rectification is inferred here.
    """
    if not isinstance(camera, Intrinsics):
        raise TypeError('camera must be Intrinsics')
    left = _gray(left_image, 'left_image')
    right = _gray(right_image, 'right_image')
    if left.shape != right.shape or left.shape != (camera.height, camera.width):
        raise ValueError('Both images must match each other and camera dimensions')
    baseline = _scalar(baseline, 'baseline')
    if type(num_disparities) is not int or not 16 <= num_disparities <= 1024 or num_disparities % 16:
        raise ValueError('num_disparities must be a multiple of 16 in 16..1024')
    if num_disparities >= camera.width:
        raise ValueError('num_disparities must be less than image width')
    if type(min_disparity) is not int or not 0 <= min_disparity <= 1024:
        raise ValueError('min_disparity must be a nonnegative integer <=1024')
    if min_disparity+num_disparities >= camera.width:
        raise ValueError('Disparity search interval must fit image width')
    if type(block_size) is not int or not 3 <= block_size <= 21 or block_size % 2 != 1:
        raise ValueError('block_size must be odd in 3..21')
    if type(uniqueness_ratio) is not int or not 0 <= uniqueness_ratio <= 100:
        raise ValueError('uniqueness_ratio must be an integer in 0..100')
    if type(speckle_window_size) is not int or not 0 <= speckle_window_size <= 100000:
        raise ValueError('speckle_window_size must be an integer in 0..100000')
    if type(speckle_range) is not int or not 0 <= speckle_range <= 1000:
        raise ValueError('speckle_range must be an integer in 0..1000')
    if type(left_right_check) is not bool:
        raise ValueError('left_right_check must be bool')
    tolerance = _scalar(left_right_tolerance_px, 'left_right_tolerance_px', zero=True)
    minimum = _scalar(min_depth, 'min_depth')
    maximum = _scalar(max_depth, 'max_depth', infinity=True)
    if maximum <= minimum:
        raise ValueError('max_depth must exceed min_depth')
    parameters = dict(numDisparities=num_disparities, blockSize=block_size,
                      P1=8*block_size**2, P2=32*block_size**2,
                      disp12MaxDiff=1, uniquenessRatio=uniqueness_ratio,
                      speckleWindowSize=speckle_window_size,
                      speckleRange=speckle_range,
                      mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    matcher = cv2.StereoSGBM_create(minDisparity=min_disparity, **parameters)
    raw = matcher.compute(left, right).astype(np.float32)/16.
    valid = np.isfinite(raw) & (raw > 0) & (raw >= min_disparity)
    if left_right_check:
        reverse_min = -min_disparity-num_disparities+1
        reverse = cv2.StereoSGBM_create(minDisparity=reverse_min, **parameters).compute(right, left).astype(np.float32)/16.
        xx = np.broadcast_to(np.arange(camera.width), raw.shape)
        yy = np.broadcast_to(np.arange(camera.height)[:, None], raw.shape)
        right_x = np.rint(xx-raw).astype(np.int32)
        in_bounds = (right_x >= 0) & (right_x < camera.width)
        valid &= in_bounds
        reverse_matched = np.full(raw.shape, np.nan, dtype=np.float32)
        reverse_matched[valid] = reverse[yy[valid], right_x[valid]]
        valid &= ((reverse_matched > reverse_min-1) &
                  (np.abs(raw+reverse_matched) <= tolerance))
    depth = np.full(raw.shape, np.nan, dtype=np.float32)
    with np.errstate(divide='ignore', over='ignore'):
        depth[valid] = camera.fx*baseline/raw[valid]
    valid &= np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    depth[~valid] = np.nan
    disparity = raw.copy()
    disparity[~valid] = np.nan
    return dict(disparity_px=disparity, depth=depth, valid=valid,
                raw_disparity_px=raw, valid_count=int(valid.sum()))
