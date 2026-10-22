"""Robust cylinder fitting with an externally supplied, fixed axis direction.

This is a constrained fit, not an unknown-axis cylinder detector. Array work is
vectorized in NumPy; RANSAC and the small refinements have bounded iteration
counts. Distances, radii and observed lengths use the input point units.
"""
import math
from numbers import Integral, Real

import numpy as np

from operators.robust_geometry import circle_model


def _scalar(value, name, *, allow_inf=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real scalar')
    value = float(value)
    if not math.isfinite(value) and not (allow_inf and value == math.inf):
        raise ValueError(f'{name} must be finite')
    return value


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < low or (high is not None and value > high)):
        raise ValueError(f'Invalid {name}')
    return int(value)


def _distances(xy, model):
    difference = xy - model[0]
    return np.abs(np.hypot(difference[:, 0], difference[:, 1]) - model[1])


def _refine(xy, initial):
    """Minimize radial, rather than algebraic, residuals in normalized units."""
    center, radius = initial[0].copy(), initial[1]
    for _ in range(12):
        difference = xy - center
        distance = np.hypot(difference[:, 0], difference[:, 1])
        if np.any(distance <= np.finfo(float).eps):
            break
        error = distance - radius
        jacobian = np.column_stack((-difference / distance[:, None], -np.ones(len(xy))))
        step, _, rank, _ = np.linalg.lstsq(jacobian, -error, rcond=None)
        if rank < 3 or not np.isfinite(step).all():
            break
        if np.linalg.norm(step) <= 1e-12 * max(1., radius):
            break
        loss = float(error @ error)
        accepted = False
        for fraction in (1., .5, .25, .125, .0625):
            trial = (center + fraction * step[:2], radius + fraction * step[2])
            if trial[1] <= 0:
                continue
            residual = _distances(xy, trial)
            if float(residual @ residual) <= loss:
                center, radius = trial
                accepted = True
                break
        if not accepted:
            break
    return center, float(radius)


def _empty(reason, count):
    return dict(success=False, reason=reason, axis_point=None, axis_direction=None,
                radius=None, segment_xyz=None, length_observed=None,
                coverage_deg=None, inliers=np.zeros(count, dtype=bool),
                residuals=None, rms=None, inlier_count=0, inlier_ratio=0.,
                candidate=None)


def fit_cylinder_known_axis(points, axis_direction, *, threshold=.005,
                            min_radius=0., max_radius=math.inf, min_inliers=12,
                            min_inlier_ratio=.6, min_coverage_deg=120.,
                            min_axial_span=.01, max_iterations=512, seed=0):
    """Fit a cylinder to finite Nx3 surface points using a known axis.

    The direction is fixed, normalized without changing its sign, and must be
    known independently (for example from gravity or a calibrated fixture).
    RANSAC fits circles in its perpendicular plane; geometric least squares
    then refines the radial residuals. Check ``success`` before using geometry.

    ``axis_point`` lies on the axis near the supported segment center.
    ``segment_xyz`` contains the minimum and maximum axial coordinates of
    radial inliers, ordered along the supplied direction. They describe only
    visible support, not physical cylinder ends. Points arbitrarily far along
    the same infinite cylinder are radial inliers and can extend this segment.

    ``coverage_deg`` is 360 minus the largest gap between inlier angles. It
    measures angular span, not uniform sampling density or fit probability.
    Rejected fits retain diagnostics and a ``candidate`` geometry dictionary;
    their top-level geometry is None. ``residuals`` are unsigned radial errors
    for every input point; ``inliers == (residuals <= threshold)`` and ``rms``
    uses that final mask even for rejected candidates.

    Finite real coordinates with magnitude <=1e150 and up to two million
    points are accepted; unrepresentable numeric ranges raise ValueError.
    Invalid parameters raise ValueError. Insufficient or degenerate geometric
    support returns success=False. No input is modified.
    """
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'uif'
            or len(points) > 2_000_000):
        raise ValueError('points must be a real Nx3 array with at most two million points')
    points = np.ascontiguousarray(points, dtype=np.float64)
    if not np.isfinite(points).all() or np.any(np.abs(points) > 1e150):
        raise ValueError('points must be finite with magnitude <=1e150')
    direction = np.asarray(axis_direction)
    if direction.shape != (3,) or direction.dtype.kind not in 'uif':
        raise ValueError('axis_direction must be a real three-vector')
    direction = direction.astype(np.float64)
    if not np.isfinite(direction).all() or not np.any(direction):
        raise ValueError('axis_direction must be finite and nonzero')
    # Scaling first also supports very small/large nonzero input directions.
    direction /= np.max(np.abs(direction))
    direction /= np.linalg.norm(direction)
    threshold = _scalar(threshold, 'threshold')
    min_radius = _scalar(min_radius, 'min_radius')
    max_radius = _scalar(max_radius, 'max_radius', allow_inf=True)
    min_inlier_ratio = _scalar(min_inlier_ratio, 'min_inlier_ratio')
    min_coverage_deg = _scalar(min_coverage_deg, 'min_coverage_deg')
    min_axial_span = _scalar(min_axial_span, 'min_axial_span')
    if not (threshold > 0 and 0 <= min_radius < max_radius
            and 0 < min_inlier_ratio <= 1 and 0 <= min_coverage_deg <= 360
            and min_axial_span >= 0):
        raise ValueError('Invalid threshold, radius, support, coverage or span bounds')
    min_inliers = _integer(min_inliers, 'min_inliers', 3)
    max_iterations = _integer(max_iterations, 'max_iterations', 1, 10000)
    seed = _integer(seed, 'seed', 0)
    count = len(points)
    required = max(min_inliers, math.ceil(min_inlier_ratio * count))
    if count < required:
        return _empty('insufficient_points', count)

    anchor = points.min(axis=0) / 2 + points.max(axis=0) / 2
    centered = points - anchor
    basis_seed = np.eye(3)[np.argmin(np.abs(direction))]
    u = basis_seed - direction * (basis_seed @ direction)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    xy = np.column_stack((centered @ u, centered @ v))
    scale = float(np.max(np.abs(xy)))
    if scale == 0:
        return _empty('degenerate_cross_section', count)
    xy /= scale
    normalized_threshold = threshold / scale
    if not math.isfinite(normalized_threshold) or normalized_threshold == 0:
        raise ValueError('threshold and point spread exceed the supported numeric range')

    rng = np.random.default_rng(seed)
    best_model, best_mask, best_count, best_error = None, None, 0, math.inf
    fallback_model, fallback_mask, fallback_count, fallback_error = None, None, 0, math.inf
    limit = max_iterations
    for iteration in range(max_iterations):
        if iteration >= limit:
            break
        model = circle_model(xy[rng.choice(count, 3, replace=False)])
        if model is None:
            continue
        distance = _distances(xy, model)
        mask = distance <= normalized_threshold
        support = int(mask.sum())
        error = float(distance[mask].sum()) if support else math.inf
        if support > fallback_count or (support == fallback_count and error < fallback_error):
            fallback_model, fallback_mask = model, mask
            fallback_count, fallback_error = support, error
        # Radius limits constrain the search, not only the winning model's
        # final validation. A larger excluded cylinder must not hide a valid
        # smaller surface when the configured inlier ratio permits both.
        if not min_radius <= model[1] * scale <= max_radius:
            continue
        if support > best_count or (support == best_count and error < best_error):
            best_model, best_mask, best_count, best_error = model, mask, support, error
            probability = (support / count) ** 3
            if probability >= 1:
                limit = iteration + 1
            elif probability > 0:
                limit = min(limit, max(iteration + 1,
                            math.ceil(math.log(.001) / math.log1p(-probability))))
    if best_model is None:
        best_model, best_mask, best_count = fallback_model, fallback_mask, fallback_count
    if best_model is None or best_count < 3:
        return _empty('degenerate_cross_section', count)

    model, mask = best_model, best_mask
    for _ in range(6):
        if int(mask.sum()) < 3:
            break
        fitted = _refine(xy[mask], model)
        updated = _distances(xy, fitted) <= normalized_threshold
        model = fitted
        if np.array_equal(mask, updated):
            break
        mask = updated

    radius = float(model[1] * scale)
    axis_origin = anchor + scale * (model[0][0] * u + model[0][1] * v)
    if not math.isfinite(radius) or not np.isfinite(axis_origin).all():
        return _empty('numerical_range', count)
    # Choose a point near the visible middle to avoid an unnecessarily remote
    # line origin. Residuals below are recomputed from the actual returned axis.
    if np.any(mask):
        axial = (points[mask] - axis_origin) @ direction
        axis_origin = axis_origin + (.5 * float(axial.min()) + .5 * float(axial.max())) * direction
    delta = points - axis_origin
    axial = delta @ direction
    radial = delta - axial[:, None] * direction
    residuals = np.abs(np.linalg.norm(radial, axis=1) - radius)
    mask = residuals <= threshold
    support = int(mask.sum())
    ratio = support / count
    rms = float(np.sqrt(np.mean(residuals[mask] ** 2))) if support else None
    coverage = 0.
    length = 0.
    segment = np.tile(axis_origin, (2, 1))
    if support:
        angles = np.sort(np.arctan2(radial[mask] @ v, radial[mask] @ u))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
        coverage = float(np.degrees(2 * math.pi - gaps.max()))
        low, high = float(axial[mask].min()), float(axial[mask].max())
        length = high - low
        segment = axis_origin + np.array([low, high])[:, None] * direction
    candidate = dict(axis_point=axis_origin, axis_direction=direction,
                     radius=radius, segment_xyz=segment, length_observed=length,
                     coverage_deg=coverage)
    reason = ('insufficient_support' if support < required else
              'radius_out_of_bounds' if not min_radius <= radius <= max_radius else
              'insufficient_angular_coverage' if coverage < min_coverage_deg else
              'insufficient_axial_span' if length < min_axial_span else 'ok')
    result = _empty(reason, count)
    result.update(success=reason == 'ok', inliers=mask, residuals=residuals,
                  inlier_count=support, inlier_ratio=ratio, rms=rms)
    if reason == 'ok':
        result.update(candidate)
    else:
        result['candidate'] = candidate
    return result
