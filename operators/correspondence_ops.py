"""Robust SE(3) alignment of known 3-D correspondences (NumPy CPU backend)."""
import math
from numbers import Integral, Real

import numpy as np


# Reject line-like support; planar but noncollinear points remain observable.
_RANK_RATIO = 1e-6
_REFINEMENT_STEPS = 8
_MAX_POINTS = 2_000_000
_MAX_ITERATIONS = 10_000


def _points(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind not in "uif":
        raise ValueError(f"{name} must be a real numeric Nx3 array")
    if len(array) > _MAX_POINTS:
        raise ValueError(f"{name} exceeds the {_MAX_POINTS} point limit")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return array


def _real(value, name, lower, upper=None, inclusive_upper=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= lower:
        raise ValueError(f"{name} must be finite and greater than {lower}")
    if upper is not None and (value > upper if inclusive_upper else value >= upper):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _fit(a, b):
    """Kabsch in local coordinates, independently scaled to avoid covariance overflow."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - ca, b - cb
    sa, sb = np.max(np.abs(ac)), np.max(np.abs(bc))
    if sa == 0 or sb == 0 or not np.isfinite([sa, sb]).all():
        return None
    ac, bc = ac / sa, bc / sb
    # Both clouds need two observable axes. SVD accepts planar support.
    asv = np.linalg.svd(ac, compute_uv=False)
    bsv = np.linalg.svd(bc, compute_uv=False)
    if asv[1] <= _RANK_RATIO * asv[0] or bsv[1] <= _RANK_RATIO * bsv[0]:
        return None
    u, _, vt = np.linalg.svd(ac.T @ bc)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    offset = cb - rotation @ ca
    return rotation, offset


def _distances(source, target, rotation, offset):
    difference = source @ rotation.T + offset - target
    # hypot avoids overflow/underflow from squaring physical coordinates.
    return np.hypot(np.hypot(difference[:, 0], difference[:, 1]), difference[:, 2])


def _score(distances, threshold):
    mask = distances <= threshold
    count = int(mask.sum())
    # Normalizing before squaring keeps the score finite at any threshold scale.
    loss = float(np.mean((distances[mask] / threshold) ** 2)) if count else math.inf
    return mask, count, loss


def _better(count, loss, best_count, best_loss):
    return count > best_count or (count == best_count and loss < best_loss)


def _iteration_limit(count, total, confidence, maximum):
    if count < 3:
        return maximum
    if count == total:
        return 0
    # Exact probability of three inliers sampled without replacement.
    probability = (count / total) * ((count - 1) / (total - 1)) * ((count - 2) / (total - 2))
    needed = math.ceil(math.log1p(-confidence) / math.log1p(-probability))
    return min(maximum, needed)


def _refine(source, target, model, threshold):
    distances = _distances(source, target, *model)
    mask, count, loss = _score(distances, threshold)
    best = model, mask, count, loss
    for _ in range(_REFINEMENT_STEPS):
        if count < 3:
            break
        refined = _fit(source[mask], target[mask])
        if refined is None:
            break
        new_mask, new_count, new_loss = _score(_distances(source, target, *refined), threshold)
        if _better(new_count, new_loss, best[2], best[3]):
            best = refined, new_mask, new_count, new_loss
        if np.array_equal(mask, new_mask):
            break
        model, mask, count = refined, new_mask, new_count
    return best


def estimate_rigid_transform_robust(source, target, *, threshold=.01,
                                    max_iterations=512, confidence=.999,
                                    min_inliers=6, min_inlier_ratio=.5, seed=0):
    """Estimate T_target_from_source from known, possibly mismatched Nx3 pairs.

    Uses seeded three-point RANSAC, adaptive stopping and bounded Kabsch
    refinement. Threshold and residuals have the same units as the input.
    No correspondences are discovered; a coherent wrong correspondence set
    cannot be distinguished from the intended object by geometry alone.

    Invalid arrays/parameters raise ValueError. Valid but insufficient or
    degenerate data return success=False and transform=None. See
    docs/CORRESPONDENCE_POSE.md for failure diagnostics and numerical limits.
    """
    source, target = _points(source, "source"), _points(target, "target")
    if source.shape != target.shape:
        raise ValueError("Correspondences must have equal Nx3 shape")
    threshold = _real(threshold, "threshold", 0)
    confidence = _real(confidence, "confidence", 0, 1)
    min_inlier_ratio = _real(min_inlier_ratio, "min_inlier_ratio", 0, 1, True)
    max_iterations = _integer(max_iterations, "max_iterations", 1)
    if max_iterations > _MAX_ITERATIONS:
        raise ValueError(f"max_iterations must be <= {_MAX_ITERATIONS}")
    min_inliers = _integer(min_inliers, "min_inliers", 3)
    seed = _integer(seed, "seed", 0)
    n = len(source)
    required = max(min_inliers, math.ceil(min_inlier_ratio * n))
    result = {"success": False, "reason": "insufficient_points", "transform": None,
              "candidate_transform": None, "inliers": np.zeros(n, dtype=bool),
              "residuals": np.full(n, np.inf), "inlier_count": 0,
              "inlier_ratio": 0., "rms": None, "iterations": 0}
    if n < required:
        return result

    # An origin near the data preserves local geometry with large world offsets.
    # Scale first only after subtraction: dividing global coordinates first can
    # discard small but representable local differences.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        a, b = source - source[0], target - target[0]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            result["reason"] = "numeric_range"
            return result
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        if scale == 0:
            result["reason"] = "degenerate_geometry"
            return result
        a, b = a / scale, b / scale
        scaled_threshold = threshold / scale
        if not math.isfinite(scaled_threshold) or scaled_threshold == 0:
            result["reason"] = "numeric_range"
            return result

    rng = np.random.default_rng(seed)
    best = None
    best_count, best_loss = 0, math.inf
    limit = max_iterations
    # Clean correspondences take one fit and no random trials.
    initial = _fit(a, b)
    if initial is not None:
        refined = _refine(a, b, initial, scaled_threshold)
        if refined[2] >= 3:
            best, _, best_count, best_loss = refined
    # A good initial fit has the same support evidence as a sampled candidate;
    # leaving its budget at maximum wastes trials when it cannot be improved.
    limit = _iteration_limit(best_count, n, confidence, max_iterations)

    iterations = 0
    while iterations < limit:
        indices = rng.choice(n, 3, replace=False)
        iterations += 1
        model = _fit(a[indices], b[indices])
        if model is None:
            continue
        mask, count, loss = _score(_distances(a, b, *model), scaled_threshold)
        if not _better(count, loss, best_count, best_loss):
            continue
        refined = _refine(a, b, model, scaled_threshold)
        if _better(refined[2], refined[3], best_count, best_loss):
            best, _, best_count, best_loss = refined
            needed = _iteration_limit(best_count, n, confidence, max_iterations)
            limit = min(limit, max(iterations, needed))
    result["iterations"] = iterations
    if best is None:
        result["reason"] = "degenerate_geometry"
        return result

    rotation, local_offset = best
    with np.errstate(over="ignore", invalid="ignore"):
        offset = target[0] + local_offset * scale - rotation @ source[0]
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, offset
        distances = _distances(source, target, rotation, offset)
    if not np.isfinite(transform).all() or not np.isfinite(distances).all():
        result["reason"] = "numeric_range"
        return result
    # Public diagnostics must describe the actual returned matrix, including
    # roundoff incurred when converting back to the user's world coordinates.
    mask, count, loss = _score(distances, threshold)
    result.update(candidate_transform=transform, inliers=mask, residuals=distances,
                  inlier_count=count, inlier_ratio=count / n,
                  rms=threshold * math.sqrt(loss) if count else None)
    if count < required:
        result["reason"] = "insufficient_inliers"
        return result
    if _fit(a[mask], b[mask]) is None:
        result["reason"] = "degenerate_inliers"
        return result
    result.update(success=True, reason="ok", transform=transform)
    return result
