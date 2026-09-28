"""Gravity-aligned 3D bounding boxes with a minimum-area horizontal footprint."""
import math

import cv2
import numpy as np


def gravity_aligned_box(points, *, up=(0., 0., 1.), min_footprint=.001):
    """Enclose finite 3D points in a gravity-aligned, yaw-optimized box.

    OpenCV's minimum-area rectangle chooses horizontal yaw from all projected
    points; final extents are recomputed in float64 so each supplied point is
    enclosed. min_footprint is in point units and rejects near-collinear XY.
    The box encloses supplied visible points only; unseen object volume and
    outlier rejection are not inferred. Square footprints have ambiguous yaw.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'uif':
        raise ValueError('points must be a real Nx3 array')
    if len(points) > 2_000_000:
        raise ValueError('At most two million points supported')
    points = np.ascontiguousarray(points, dtype=np.float64)
    if not np.isfinite(points).all() or np.any(np.abs(points) > 1e9):
        raise ValueError('points must be finite with magnitude <=1e9')
    up = np.asarray(up)
    if up.shape != (3,) or up.dtype.kind not in 'uif':
        raise ValueError('up must be a real three-vector')
    up = up.astype(np.float64)
    norm_up = float(np.linalg.norm(up))
    if not np.isfinite(up).all() or not math.isfinite(norm_up) or norm_up <= 1e-12:
        raise ValueError('up must be finite and nonzero')
    if isinstance(min_footprint, (bool, np.bool_)) or not np.isscalar(min_footprint):
        raise ValueError('min_footprint must be a positive finite scalar')
    try:
        valid_min = math.isfinite(min_footprint) and min_footprint > 0
    except (TypeError, ValueError):
        valid_min = False
    if not valid_min:
        raise ValueError('min_footprint must be a positive finite scalar')
    up /= norm_up
    if len(points) < 3:
        return dict(success=False, reason='insufficient_points', center=None,
                    size=None, axes=None, corners=None, footprint_area=None,
                    volume=None, orientation_ambiguous=None)
    anchor = points.mean(axis=0)
    centered = points-anchor
    seed = np.eye(3)[np.argmin(np.abs(up))]
    u0 = seed-up*(seed@up)
    u0 /= np.linalg.norm(u0)
    v0 = np.cross(up, u0)
    xy = np.column_stack((centered@u0, centered@v0))
    if np.max(np.abs(xy)) > 1e100:
        raise ValueError('Projected coordinates too large')
    try:
        rectangle = cv2.minAreaRect(xy.astype(np.float32))
        corners_2d = cv2.boxPoints(rectangle).astype(np.float64)
    except cv2.error as error:
        raise ValueError('OpenCV cannot form a footprint rectangle') from error
    sides = np.roll(corners_2d, -1, axis=0)-corners_2d
    lengths = np.linalg.norm(sides, axis=1)
    long_side = int(np.argmax(lengths))
    direction = sides[long_side]/lengths[long_side] if lengths[long_side] > 0 else np.array([1., 0.])
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction
    horizontal = direction[0]*u0+direction[1]*v0
    horizontal /= np.linalg.norm(horizontal)
    lateral = np.cross(up, horizontal)
    axes = np.column_stack((horizontal, lateral, up))
    coordinates = centered@axes
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    size = high-low
    if min(size[:2]) < min_footprint:
        return dict(success=False, reason='degenerate_footprint', center=None,
                    size=None, axes=None, corners=None, footprint_area=None,
                    volume=None, orientation_ambiguous=None)
    center = anchor+axes@((low+high)/2)
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float64)
    corners = center+signs*(size/2)@axes.T
    return dict(success=True, reason='ok', center=center, size=size, axes=axes,
                corners=corners, footprint_area=float(size[0]*size[1]),
                volume=float(np.prod(size)),
                orientation_ambiguous=bool(abs(size[0]-size[1]) <= 1e-3*max(size[0], size[1])))
