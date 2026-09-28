"""Robust orthogonal 3D line and observed segment measurement."""
import math
from numbers import Real

import numpy as np


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _failure(reason, count, iterations=0):
    return dict(success=False, reason=reason, point=None, direction=None,
                segment=None, span=None, inliers=np.zeros(count, bool),
                inlier_count=0, inlier_ratio=0., rms=math.inf,
                distances=np.full(count, math.inf), iterations=iterations)


def _distances(points, point, direction):
    delta = points-point
    transverse = delta-np.outer(delta@direction, direction)
    return np.linalg.norm(transverse, axis=1)


def fit_line_3d(points, *, threshold=.01, min_inliers=12,
                min_inlier_ratio=.5, min_span=0., max_iterations=1024,
                confidence=.999, seed=0):
    """Two-point RANSAC + orthogonal TLS, exact final-pose residual checks.

    Threshold, span and output coordinates have the input point units. The
    returned segment spans observed inlier points, not unseen line extent.
    A direction sign is fixed by making its largest-magnitude component
    positive; without an external orientation the sign has no physical meaning.
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
    min_span = _scalar(min_span, 'min_span')
    min_inlier_ratio = _scalar(min_inlier_ratio, 'min_inlier_ratio')
    confidence = _scalar(confidence, 'confidence')
    if threshold <= 0 or min_span < 0 or not 0 < min_inlier_ratio <= 1 or not 0 < confidence < 1:
        raise ValueError('Invalid distance, span, ratio or confidence')
    if (type(min_inliers) is not int or not 2 <= min_inliers <= 2_000_000 or
            type(max_iterations) is not int or not 1 <= max_iterations <= 100_000 or
            type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('Invalid minimum support, iterations or seed')
    count = len(points)
    required = max(min_inliers, math.ceil(count*min_inlier_ratio))
    if count < required:
        return _failure('insufficient_points', count)
    anchor = points.mean(axis=0)
    spread = np.max(np.abs(points-anchor))
    if spread <= 0:
        return _failure('degenerate_point_cloud', count)
    q = (points-anchor)/spread
    tolerance = threshold/spread
    rng = np.random.default_rng(seed)
    best_mask, best_score = None, (-1, -math.inf)
    budget = max_iterations
    iterations = 0
    for iterations in range(1, max_iterations+1):
        a, b = q[rng.choice(count, 2, replace=False)]
        direction = b-a
        length = float(np.linalg.norm(direction))
        if length > 1e-12:
            direction /= length
            errors = _distances(q, a, direction)
            mask = errors <= tolerance
            support = int(mask.sum())
            if support:
                score = (support, -float(errors[mask]@errors[mask]))
                if score > best_score:
                    best_mask, best_score = mask, score
                    if support >= required:
                        probability = (support/count)**2
                        needed = (1 if probability >= 1 else
                                  math.ceil(math.log1p(-confidence)/math.log1p(-probability)))
                        budget = min(budget, max(1, needed))
        if iterations >= budget:
            break
    if best_mask is None or best_score[0] < required:
        return _failure('insufficient_line_support', count, iterations)

    mask = best_mask
    for _ in range(5):
        subset = q[mask]
        point = subset.mean(axis=0)
        _, singular, vt = np.linalg.svd(subset-point, full_matrices=False)
        if singular[0] <= 0:
            return _failure('degenerate_line_support', count, iterations)
        direction = vt[0]
        if direction[np.argmax(abs(direction))] < 0:
            direction = -direction
        updated = _distances(q, point, direction) <= tolerance
        if np.array_equal(updated, mask):
            break
        mask = updated
        if mask.sum() < required:
            return _failure('refined_line_support_failed', count, iterations)
    point = anchor+spread*point
    errors = _distances(points, point, direction)
    inliers = np.isfinite(errors) & (errors <= threshold)
    support = int(inliers.sum())
    if support < required:
        return _failure('final_line_support_failed', count, iterations)
    projection = (points[inliers]-point)@direction
    span = float(projection.max()-projection.min())
    if span <= 0 or span < min_span:
        return _failure('line_span_failed', count, iterations)
    segment = point+np.array([projection.min(), projection.max()])[:, None]*direction
    rms = float(np.sqrt(np.mean(errors[inliers]**2)))
    return dict(success=True, reason='ok', point=point, direction=direction,
                segment=segment, span=span, inliers=inliers, inlier_count=support,
                inlier_ratio=support/count, rms=rms, distances=errors,
                iterations=iterations)
