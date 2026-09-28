"""Gravity-constrained ground candidate and signed obstacle-height segmentation."""
import math
from numbers import Real

import numpy as np


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _failure(reason, count, iterations=0):
    return dict(success=False, reason=reason, normal=None, offset=None,
                ground_height_at_origin=None, tilt_deg=None, rms=None,
                ground=np.zeros(count, bool), obstacles=np.zeros(count, bool),
                below=np.zeros(count, bool), unknown=np.ones(count, bool),
                vertical_heights=np.full(count, np.nan), ground_count=0,
                ground_ratio=0., iterations=iterations)


def _candidate(points, samples, up, cos_tilt, expected_height, height_tolerance):
    a, b, c = points[samples]
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-10 * np.linalg.norm(ab) * np.linalg.norm(ac):
        return None
    normal /= norm
    upward = float(normal @ up)
    if upward < 0:
        normal = -normal
        upward = -upward
    if upward < cos_tilt:
        return None
    offset = -float(normal @ a)
    height = -offset/upward
    if expected_height is not None and abs(height-expected_height) > height_tolerance:
        return None
    return normal, offset


def segment_ground_obstacles(points, *, up=(0., 0., 1.), max_tilt_deg=20.,
                             distance_threshold=.02, obstacle_min_height=.05,
                             obstacle_max_height=2., expected_ground_height=None,
                             height_tolerance=.15, max_iterations=1024,
                             confidence=.999, min_ground_points=100,
                             min_ground_ratio=.1, seed=0):
    """Fit a gravity-consistent plane and classify points by vertical height.

    Inputs/thresholds share arbitrary metric units, normally metres. up points
    away from gravity; frame origin defines expected_ground_height. RANSAC
    rejects planes exceeding max_tilt_deg and optionally planes whose vertical
    intercept at the origin differs from expected_ground_height. Without an
    intercept prior, a large horizontal table can win over a smaller floor.

    The returned plane normal faces up, ground uses perpendicular plane
    distance, and obstacle heights are measured along up. This classifies
    geometry, not semantic traversability or collision safety. Final masks,
    heights, support and RMS are recomputed from the returned refined plane.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'uif':
        raise ValueError('points must be a real Nx3 array')
    if len(points) > 2_000_000:
        raise ValueError('At most two million points supported')
    points = np.ascontiguousarray(points, dtype=np.float64)
    if not np.isfinite(points).all() or np.any(np.abs(points) > 1e9):
        raise ValueError('Points must be finite with magnitude <=1e9')
    up = np.asarray(up)
    if up.shape != (3,) or up.dtype.kind not in 'uif':
        raise ValueError('up must be a real three-vector')
    up = up.astype(np.float64)
    norm_up = float(np.linalg.norm(up))
    if not np.isfinite(up).all() or not math.isfinite(norm_up) or norm_up <= 1e-12:
        raise ValueError('up must be finite and nonzero')
    up /= norm_up
    max_tilt_deg = _real(max_tilt_deg, 'max_tilt_deg')
    distance_threshold = _real(distance_threshold, 'distance_threshold')
    obstacle_min_height = _real(obstacle_min_height, 'obstacle_min_height')
    obstacle_max_height = _real(obstacle_max_height, 'obstacle_max_height')
    height_tolerance = _real(height_tolerance, 'height_tolerance')
    confidence = _real(confidence, 'confidence')
    min_ground_ratio = _real(min_ground_ratio, 'min_ground_ratio')
    if expected_ground_height is not None:
        expected_ground_height = _real(expected_ground_height, 'expected_ground_height')
    if (not 0 <= max_tilt_deg <= 45 or distance_threshold <= 0 or
            not 0 <= obstacle_min_height < obstacle_max_height or height_tolerance <= 0 or
            not 0 < confidence < 1 or not 0 < min_ground_ratio <= 1):
        raise ValueError('Invalid tilt, distance, height, confidence or ground ratio')
    if (type(max_iterations) is not int or not 1 <= max_iterations <= 100_000 or
            type(min_ground_points) is not int or not 3 <= min_ground_points <= 2_000_000 or
            type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('Invalid iteration count, support count or seed')
    count = len(points)
    required = max(min_ground_points, math.ceil(count*min_ground_ratio))
    if count < required:
        return _failure('insufficient_points', count)
    cos_tilt = math.cos(math.radians(max_tilt_deg))
    rng = np.random.default_rng(seed)
    best_mask, best_plane, best_score = None, None, (-1, -math.inf)
    budget = max_iterations
    iterations = 0
    for iterations in range(1, max_iterations+1):
        sample = rng.choice(count, 3, replace=False)
        plane = _candidate(points, sample, up, cos_tilt,
                           expected_ground_height, height_tolerance)
        if plane is not None:
            normal, offset = plane
            distances = points @ normal+offset
            mask = np.abs(distances) <= distance_threshold
            support = int(mask.sum())
            if support:
                score = (support, -float(np.dot(distances[mask], distances[mask])))
                if score > best_score:
                    best_score, best_mask, best_plane = score, mask, plane
                    # Adaptive RANSAC only reduces iterations once the current
                    # support itself meets the requested final minimum.
                    if support >= required:
                        probability = (support/count)**3
                        needed = (1 if probability >= 1 else
                                  math.ceil(math.log1p(-confidence)/math.log1p(-probability)))
                        budget = min(budget, max(1, needed))
        if iterations >= budget:
            break
    if best_mask is None or best_score[0] < required:
        return _failure('insufficient_ground_support', count, iterations)

    # Least-squares orthogonal refinement; reclassify after every fit. If a
    # refinement violates the gravity/intercept prior, keep the previous plane.
    normal, offset = best_plane
    mask = best_mask
    for _ in range(4):
        subset = points[mask]
        center = subset.mean(axis=0)
        singular = np.linalg.svd(subset-center, full_matrices=False)[2]
        candidate_normal = singular[-1]
        upward = float(candidate_normal @ up)
        if upward < 0:
            candidate_normal = -candidate_normal
            upward = -upward
        candidate_offset = -float(candidate_normal @ center)
        candidate_height = -candidate_offset/upward if upward > 0 else math.inf
        if (upward < cos_tilt or
                (expected_ground_height is not None and
                 abs(candidate_height-expected_ground_height) > height_tolerance)):
            break
        normal, offset = candidate_normal, candidate_offset
        updated = np.abs(points @ normal+offset) <= distance_threshold
        if np.array_equal(mask, updated):
            mask = updated
            break
        mask = updated
        if mask.sum() < 3:
            break
    distances = points @ normal+offset
    ground = np.abs(distances) <= distance_threshold
    support = int(ground.sum())
    if support < required:
        return _failure('refined_ground_support_failed', count, iterations)
    vertical = distances/float(normal @ up)
    obstacles = ~ground & (vertical >= obstacle_min_height) & (vertical <= obstacle_max_height)
    below = ~ground & (vertical < -distance_threshold/float(normal @ up))
    unknown = ~(ground | obstacles | below)
    rms = float(np.sqrt(np.mean(distances[ground]**2)))
    return dict(success=True, reason='ok', normal=normal, offset=offset,
                ground_height_at_origin=-offset/float(normal @ up),
                tilt_deg=math.degrees(math.acos(float(np.clip(normal @ up, -1, 1)))),
                rms=rms, ground=ground, obstacles=obstacles, below=below,
                unknown=unknown, vertical_heights=vertical, ground_count=support,
                ground_ratio=support/count, iterations=iterations)
