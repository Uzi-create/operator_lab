"""Robust metric 3D sphere measurement with final-model residual validation."""
import math
from numbers import Real

import numpy as np


def _scalar(value, name, *, infinite=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real scalar')
    value = float(value)
    if math.isnan(value) or (not infinite and not math.isfinite(value)):
        raise ValueError(f'{name} must be finite')
    return value


def _failure(reason, count, iterations=0):
    return dict(success=False, reason=reason, center=None, radius=None,
                inliers=np.zeros(count, bool), inlier_count=0, inlier_ratio=0.,
                rms=math.inf, residuals=np.full(count, math.inf), iterations=iterations)


def _four_point(q, ids):
    sample = q[ids]
    matrix = 2*(sample[1:]-sample[0])
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[0] == 0 or singular[-1] <= singular[0]*1e-7:
        return None
    right = np.sum(sample[1:]**2, axis=1)-np.sum(sample[0]**2)
    center = np.linalg.solve(matrix, right)
    radius = float(np.linalg.norm(sample[0]-center))
    if not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 0:
        return None
    return center, radius


def _least_squares(q):
    matrix = np.column_stack((2*q, np.ones(len(q))))
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(q*q, axis=1), rcond=None)
    if rank < 4:
        return None
    center = solution[:3]
    radius_squared = float(solution[3]+center@center)
    if radius_squared <= 0 or not math.isfinite(radius_squared):
        return None
    radius = math.sqrt(radius_squared)
    for _ in range(12):
        vectors = q-center
        distances = np.linalg.norm(vectors, axis=1)
        if (distances <= 1e-12).any():
            break
        residuals = distances-radius
        jacobian = np.column_stack((-vectors/distances[:, None], -np.ones(len(q))))
        delta = np.linalg.lstsq(jacobian, -residuals, rcond=None)[0]
        if not np.isfinite(delta).all():
            break
        old_loss = float(residuals@residuals)
        accepted = False
        for step in (1., .5, .25, .125, .0625):
            trial_center = center+step*delta[:3]
            trial_radius = radius+step*delta[3]
            if trial_radius <= 0:
                continue
            trial = np.linalg.norm(q-trial_center, axis=1)-trial_radius
            if float(trial@trial) <= old_loss:
                center, radius, accepted = trial_center, trial_radius, True
                break
        if not accepted or np.linalg.norm(delta) < 1e-12:
            break
    return center, radius


def fit_sphere(points, *, threshold=.01, min_radius=0., max_radius=math.inf,
               min_inliers=12, min_inlier_ratio=.5, max_iterations=1024,
               confidence=.999, min_spread_ratio=.005, seed=0):
    """RANSAC four-point sphere plus geometric least squares on final inliers.

    Radius, center and thresholds share point units. RANSAC iterations adapt
    to support only after required support is reached. Near-coplanar support
    is rejected: a small visible patch does not determine a reliable sphere.
    Failure returns success=False; it never silently treats a plane as a sphere.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'uif':
        raise ValueError('points must be a real Nx3 array')
    if len(points) > 2_000_000:
        raise ValueError('At most two million points supported')
    points = np.ascontiguousarray(points, dtype=np.float64)
    if not np.isfinite(points).all() or np.any(np.abs(points) > 1e9):
        raise ValueError('Points must be finite with magnitude <=1e9')
    threshold = _scalar(threshold, 'threshold')
    min_radius = _scalar(min_radius, 'min_radius')
    max_radius = _scalar(max_radius, 'max_radius', infinite=True)
    min_inlier_ratio = _scalar(min_inlier_ratio, 'min_inlier_ratio')
    confidence = _scalar(confidence, 'confidence')
    min_spread_ratio = _scalar(min_spread_ratio, 'min_spread_ratio')
    if (threshold <= 0 or min_radius < 0 or max_radius <= min_radius or
            not 0 < min_inlier_ratio <= 1 or not 0 < confidence < 1 or
            not 0 < min_spread_ratio <= 1):
        raise ValueError('Invalid sphere distance, radius, ratio or confidence')
    if (type(min_inliers) is not int or not 4 <= min_inliers <= 2_000_000 or
            type(max_iterations) is not int or not 1 <= max_iterations <= 100_000 or
            type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('Invalid minimum support, iterations or seed')
    count = len(points)
    required = max(min_inliers, math.ceil(count*min_inlier_ratio))
    if count < required:
        return _failure('insufficient_points', count)
    anchor = points.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum((points-anchor)**2, axis=1))))
    if not math.isfinite(scale) or scale <= 0:
        return _failure('degenerate_point_cloud', count)
    q = (points-anchor)/scale
    limit = threshold/scale
    rng = np.random.default_rng(seed)
    best_mask, best_score = None, (-1, -math.inf)
    budget = max_iterations
    iterations = 0
    for iterations in range(1, max_iterations+1):
        candidate = _four_point(q, rng.choice(count, 4, replace=False))
        if candidate is not None:
            center, radius = candidate
            radius_units = radius*scale
            if min_radius <= radius_units <= max_radius:
                residuals = np.linalg.norm(q-center, axis=1)-radius
                mask = abs(residuals) <= limit
                support = int(mask.sum())
                if support:
                    score = (support, -float(residuals[mask]@residuals[mask]))
                    if score > best_score:
                        best_mask, best_score = mask, score
                        if support >= required:
                            probability = (support/count)**4
                            needed = (1 if probability >= 1 else
                                      math.ceil(math.log1p(-confidence)/math.log1p(-probability)))
                            budget = min(budget, max(1, needed))
        if iterations >= budget:
            break
    if best_mask is None or best_score[0] < required:
        return _failure('insufficient_sphere_support', count, iterations)

    mask = best_mask
    fitted = None
    for _ in range(5):
        singular = np.linalg.svd(q[mask]-q[mask].mean(axis=0), compute_uv=False)
        if singular[-1] <= singular[0]*min_spread_ratio:
            return _failure('degenerate_sphere_support', count, iterations)
        fitted = _least_squares(q[mask])
        if fitted is None:
            return _failure('sphere_refinement_failed', count, iterations)
        center, radius = fitted
        if not min_radius <= radius*scale <= max_radius:
            return _failure('sphere_radius_out_of_range', count, iterations)
        updated = abs(np.linalg.norm(q-center, axis=1)-radius) <= limit
        if np.array_equal(updated, mask):
            break
        mask = updated
        if mask.sum() < required:
            return _failure('refined_sphere_support_failed', count, iterations)
    center = anchor+scale*center
    radius *= scale
    # Public residuals and consensus are from the exact returned parameters.
    residuals = np.linalg.norm(points-center, axis=1)-radius
    inliers = np.isfinite(residuals) & (abs(residuals) <= threshold)
    support = int(inliers.sum())
    if support < required:
        return _failure('final_sphere_support_failed', count, iterations)
    final_geometry = q[inliers]-q[inliers].mean(axis=0)
    singular = np.linalg.svd(final_geometry, compute_uv=False)
    if singular[-1] <= singular[0]*min_spread_ratio:
        return _failure('degenerate_final_sphere_support', count, iterations)
    rms = float(np.sqrt(np.mean(residuals[inliers]**2)))
    return dict(success=True, reason='ok', center=center, radius=float(radius),
                inliers=inliers, inlier_count=support, inlier_ratio=support/count,
                rms=rms, residuals=residuals, iterations=iterations)
