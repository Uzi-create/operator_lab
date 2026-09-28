"""Cached calibrated horizontal stereo rectification for metric depth."""
import math
from numbers import Real

import cv2
import numpy as np

from operators.pose_ops import _calibration
from operators.ray_plane_ops import _pixels, camera_rays
from operators.robot_ops import Intrinsics, transform_points


class StereoRectifier:
    """Precompute OpenCV C++ maps for a fixed synchronized stereo rig."""

    def __init__(self, left_camera, right_camera, T_right_from_left, *,
                 distortion_left=None, distortion_right=None, alpha=0.):
        left_matrix, left_distortion = _calibration(left_camera, distortion_left)
        right_matrix, right_distortion = _calibration(right_camera, distortion_right)
        if ((left_camera.width, left_camera.height) !=
                (right_camera.width, right_camera.height)):
            raise ValueError('Stereo rectification requires equal input image sizes')
        if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
            raise ValueError('alpha must be a finite real in [0,1]')
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        transform = np.asarray(T_right_from_left, dtype=np.float64)
        transform_points(np.empty((0, 3)), transform)
        right_origin_left = -transform[:3, :3].T @ transform[:3, 3]
        if right_origin_left[0] <= 0 or abs(right_origin_left[0]) <= abs(right_origin_left[1]):
            raise ValueError('Right camera must be primarily on the left camera +X side')
        size = (left_camera.width, left_camera.height)
        try:
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                left_matrix, left_distortion, right_matrix, right_distortion,
                size, transform[:3, :3], transform[:3, 3],
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
            left_maps = cv2.initUndistortRectifyMap(left_matrix, left_distortion,
                                                    R1, P1, size, cv2.CV_16SC2)
            right_maps = cv2.initUndistortRectifyMap(right_matrix, right_distortion,
                                                     R2, P2, size, cv2.CV_16SC2)
        except cv2.error as error:
            raise ValueError('OpenCV could not rectify the supplied stereo calibration') from error
        baseline = float(-P2[0, 3]/P2[0, 0])
        if not math.isfinite(baseline) or baseline <= 0 or abs(P2[1, 3]) > 1e-8:
            raise ValueError('Calibration did not produce positive horizontal stereo baseline')
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.left_distortion = left_distortion
        self.right_distortion = right_distortion
        self.left_matrix = left_matrix
        self.right_matrix = right_matrix
        self.R_left = R1
        self.R_right = R2
        self.P_left = P1
        self.P_right = P2
        self.Q = Q
        self.roi_left = tuple(int(v) for v in roi1)
        self.roi_right = tuple(int(v) for v in roi2)
        self.baseline = baseline
        self.rectified_camera = Intrinsics(size[0], size[1],
                                           float(P1[0, 0]), float(P1[1, 1]),
                                           float(P1[0, 2]), float(P1[1, 2]))
        self.T_right_from_left_rectified = np.eye(4)
        self.T_right_from_left_rectified[:3, :3] = R2 @ transform[:3, :3] @ R1.T
        self.T_right_from_left_rectified[:3, 3] = R2 @ transform[:3, 3]
        self._left_maps = left_maps
        self._right_maps = right_maps

    def rectify_points(self, pixels_left, pixels_right, *,
                       max_inverse_reprojection_error_px=1e-6):
        """Map corresponding raw pixels to rectified image coordinates."""
        left = _pixels(pixels_left)
        right = _pixels(pixels_right)
        if left.shape != right.shape:
            raise ValueError('Correspondence arrays must have the same Nx2 shape')
        if not len(left):
            return dict(left=np.empty((0, 2)), right=np.empty((0, 2)),
                        vertical_errors_px=np.empty(0), valid=np.empty(0, bool),
                        inverse_reprojection_errors_px=np.empty((0, 2)))
        left_rays = camera_rays(left, self.left_camera, distortion=self.left_distortion,
                                max_reprojection_error_px=max_inverse_reprojection_error_px)
        right_rays = camera_rays(right, self.right_camera, distortion=self.right_distortion,
                                 max_reprojection_error_px=max_inverse_reprojection_error_px)
        valid = left_rays['valid'] & right_rays['valid']
        rectified = []
        for rays, rotation, projection in ((left_rays, self.R_left, self.P_left),
                                           (right_rays, self.R_right, self.P_right)):
            hom = rays['rays_camera'] @ rotation.T @ projection[:, :3].T
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                points = hom[:, :2]/hom[:, 2, None]
            valid &= (np.isfinite(points).all(axis=1) & np.isfinite(hom[:, 2]) &
                      (hom[:, 2] > 0))
            rectified.append(points)
        rectified_left, rectified_right = rectified
        rectified_left[~valid] = np.nan
        rectified_right[~valid] = np.nan
        errors = np.abs(rectified_left[:, 1]-rectified_right[:, 1])
        errors[~valid] = math.inf
        return dict(left=rectified_left, right=rectified_right,
                    vertical_errors_px=errors, valid=valid,
                    inverse_reprojection_errors_px=np.column_stack((
                        left_rays['reprojection_errors_px'],
                        right_rays['reprojection_errors_px'])))

    def rectify_images(self, left_image, right_image, *, interpolation=cv2.INTER_LINEAR):
        """Remap raw images using cached maps; output is ready for dense stereo."""
        if interpolation not in (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC):
            raise ValueError('Unsupported image interpolation')
        size = (self.left_camera.height, self.left_camera.width)
        for image, name in ((left_image, 'left'), (right_image, 'right')):
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                    image.shape not in (size, size+(3,))):
                raise ValueError(f'{name} must be uint8 grayscale or BGR at calibrated size')
        left = cv2.remap(left_image, *self._left_maps, interpolation)
        right = cv2.remap(right_image, *self._right_maps, interpolation)
        return dict(left=left, right=right, camera=self.rectified_camera,
                    baseline=self.baseline, Q=self.Q)
