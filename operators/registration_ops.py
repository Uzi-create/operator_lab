"""Exact 3D nearest neighbors and local point-to-point rigid ICP.

Transforms are T_target_from_source, acting on column vectors R*p+t.
ICP requires a useful initial pose and sufficient overlap; convergence is not
proof of a globally correct pose, especially for symmetric/repeated geometry.
"""
import ctypes
import math
from numbers import Real
from pathlib import Path
import sys
import weakref

import numpy as np


_NATIVE = None
_MAX_POINTS = 10_000_000
_MAX_COORDINATE = 1e150


def _real_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real scalar')
    return float(value)


def _points(values, minimum=0):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[1] != 3 or values.dtype.kind not in 'uif'
            or not minimum <= len(values) <= _MAX_POINTS):
        raise ValueError(f'Expected {minimum}..{_MAX_POINTS} real Nx3 points')
    values = np.require(values, dtype=np.float64, requirements=['C', 'A'])
    if not np.isfinite(values).all() or np.any(np.abs(values) > _MAX_COORDINATE):
        raise ValueError('Points must be finite and have magnitude at most 1e150')
    return values


def _native(backend):
    global _NATIVE
    if backend not in ('auto', 'native', 'numpy'):
        raise ValueError('backend must be auto, native or numpy')
    if backend == 'numpy':
        return None
    if _NATIVE is None:
        filename = {'win32': 'operators.dll', 'darwin': 'liboperators.dylib'}.get(sys.platform, 'liboperators.so')
        try:
            library = ctypes.CDLL(str(Path(__file__).resolve().parent / filename))
            d, i = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64)
            library.nn3_create.argtypes = [d, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
            library.nn3_query.argtypes = [ctypes.c_uint64, d, ctypes.c_size_t, ctypes.c_double, i, d]
            library.nn3_release.argtypes = [ctypes.c_uint64]
            for function in (library.nn3_create, library.nn3_query, library.nn3_release):
                function.restype = ctypes.c_int
            _NATIVE = library
        except (OSError, AttributeError) as error:
            if backend == 'native':
                raise RuntimeError('Native nearest-neighbor library unavailable; run python build.py') from error
            return None
    return _NATIVE


def _pointer(array, kind=ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(kind))


def _numpy_query(target, query, max_distance):
    """Bounded-memory brute force, no dot-product cancellation approximation."""
    indices = np.full(len(query), -1, np.int64)
    squared = np.full(len(query), np.inf)
    limit = max_distance * max_distance if max_distance <= 1e154 else np.inf
    for start in range(0, len(query), 256):
        q = query[start:start + 256]
        best = np.full(len(q), limit)
        index = np.full(len(q), -1, np.int64)
        for offset in range(0, len(target), 2048):
            t = target[offset:offset + 2048]
            # Separate differences preserve accuracy when coordinates share a
            # large offset; norm(q)^2+norm(t)^2-2*q*t would lose these distances.
            distance = (q[:, None, 0] - t[None, :, 0]) ** 2
            distance += (q[:, None, 1] - t[None, :, 1]) ** 2
            distance += (q[:, None, 2] - t[None, :, 2]) ** 2
            nearest = np.argmin(distance, axis=1)
            candidate = distance[np.arange(len(q)), nearest]
            take = (candidate < best) | ((candidate == best) & (index < 0))
            best[take] = candidate[take]
            index[take] = nearest[take] + offset
        indices[start:start + len(q)] = index
        squared[start:start + len(q)] = np.where(index >= 0, best, np.inf)
    return indices, squared


class NearestNeighborIndex:
    """Reusable exact 3D target index owning a snapshot of its input.

    ``native`` is a balanced C++ KD tree; ``numpy`` is chunked exact brute force.
    ``auto`` selects the native backend if available. Equal squared distances
    choose the smallest original target index. No approximate search is used.
    ``query`` supports concurrent readers; do not close while queries are active.
    Use a context manager or close() to promptly release native tree memory.
    """
    def __init__(self, target, *, backend='auto'):
        self._points = _points(target, 1).copy()
        self._points.setflags(write=False)
        self._library = _native(backend)
        self.backend = 'native' if self._library is not None else 'numpy'
        self._handle = 0
        self._closed = False
        self._finalizer = None
        if self._library is not None:
            handle = ctypes.c_uint64()
            code = self._library.nn3_create(_pointer(self._points), len(self._points), ctypes.byref(handle))
            if code == 2:
                raise MemoryError('Could not allocate native target index')
            if code:
                raise ValueError('Native target index rejected points')
            self._handle = handle.value
            self._finalizer = weakref.finalize(self, self._library.nn3_release, self._handle)

    @property
    def target_points(self):
        """Return a copy so callers cannot mutate the cached target geometry."""
        return self._points.copy()

    def __len__(self):
        return len(self._points)

    def query(self, query, *, max_distance=math.inf):
        """Return indices, squared_distances, distances, valid for Nx3 queries.

        max_distance is inclusive and nonnegative (infinity allowed). Missing
        neighbors have index -1 and infinite distance. Empty (0,3) queries work.
        Target and query coordinates use the same units and frame.
        """
        if self._closed:
            raise RuntimeError('Nearest-neighbor index is closed')
        max_distance = _real_scalar(max_distance, 'max_distance')
        if math.isnan(max_distance) or max_distance < 0:
            raise ValueError('max_distance must be nonnegative, or positive infinity')
        query = _points(query)
        if self._library is None:
            indices, squared = _numpy_query(self._points, query, max_distance)
        else:
            indices = np.empty(len(query), np.int64)
            squared = np.empty(len(query), np.float64)
            code = self._library.nn3_query(self._handle, _pointer(query), len(query), max_distance,
                                          _pointer(indices, ctypes.c_int64), _pointer(squared))
            if code == 3:
                raise RuntimeError('Nearest-neighbor index was released')
            if code:
                raise ValueError('Native nearest-neighbor query rejected input')
        return {'indices': indices, 'squared_distances': squared,
                'distances': np.sqrt(squared), 'valid': indices >= 0}

    def close(self):
        if not self._closed:
            if self._finalizer is not None:
                self._finalizer()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Nearest-neighbor index is closed')
        return self

    def __exit__(self, *args):
        self.close()


def _pose(value):
    if value is None:
        return np.eye(4)
    value = np.asarray(value)
    if value.dtype.kind not in 'uif' or value.shape != (4, 4):
        raise ValueError('initial_transform must be a real rigid 4x4 matrix')
    value = value.astype(np.float64, copy=True)
    if (not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), rtol=0, atol=1e-8)
            or not math.isclose(np.linalg.det(value[:3, :3]), 1., abs_tol=1e-8)):
        raise ValueError('initial_transform must have orthonormal rotation with determinant +1')
    value[3] = [0, 0, 0, 1]
    return value


def _centered_nondegenerate(points):
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) < 2 or singular[0] == 0 or singular[1] <= singular[0] * 1e-8:
        raise ValueError('Coincident, collinear or nearly collinear geometry is ambiguous')
    return centered


def _rigid_update(source, target):
    a, b = _centered_nondegenerate(source), _centered_nondegenerate(target)
    # Normalize to avoid overflow/underflow of covariance without changing R.
    a /= np.max(np.abs(a))
    b /= np.max(np.abs(b))
    u, singular, vt = np.linalg.svd(a.T @ b)
    if singular[1] <= singular[0] * 1e-10:
        raise ValueError('Correspondence covariance is geometrically degenerate')
    correction = np.eye(3)
    correction[-1, -1] = 1. if np.linalg.det(vt.T @ u.T) >= 0 else -1.
    rotation = vt.T @ correction @ u.T
    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    update = np.eye(4)
    update[:3, :3], update[:3, 3] = rotation, translation
    return update


def _correspondences(index, transformed, distance, trim_fraction, minimum, min_overlap):
    nearest = index.query(transformed, max_distance=distance)
    selected = np.flatnonzero(nearest['valid'])
    if len(selected) < minimum or len(selected) / len(transformed) < min_overlap:
        raise ValueError('Insufficient overlap / correspondences within max_distance')
    keep = math.ceil(len(selected) * trim_fraction)
    if keep < minimum:
        raise ValueError('Too few correspondences remain after trimming')
    inliers = np.zeros(len(transformed), bool)
    if keep == len(selected):
        inliers[selected] = True
    else:
        order = np.argsort(nearest['squared_distances'][selected], kind='stable')[:keep]
        inliers[selected[order]] = True
    nearest['inliers'] = inliers
    nearest['rms'] = float(np.sqrt(np.mean(nearest['squared_distances'][inliers])))
    nearest['overlap'] = len(selected) / len(transformed)
    return nearest


def icp_point_to_point(source, target, *, max_distance, initial_transform=None,
                       trim_fraction=1., min_correspondences=6, min_overlap=.1,
                       max_iterations=50, translation_tolerance=1e-6,
                       rotation_tolerance=1e-6, backend='auto'):
    """Local rigid ICP with exact matching and optional trimmed residuals.

    target is Nx3 or a cached NearestNeighborIndex (its backend wins). Coordinates
    and translation thresholds share units; rotation_tolerance is radians.
    max_distance must be finite and positive. min_overlap is the fraction of all
    source points passing distance gating, BEFORE trimming. trim_fraction keeps
    ceil(fraction*candidate_count) smallest residuals, tie by source index.

    Requires at least min_correspondences non-collinear pairs after trimming.
    Rejects failed overlap and geometric degeneracy with ValueError. Planar
    non-collinear pairs are allowed. Many-to-one matching is allowed. This does
    not infer semantic overlap, resolve symmetry, or prove global registration.

    Stops only when BOTH incremental rotation and translation are small; hitting
    max_iterations returns converged=False. Final matches, inliers, residuals,
    overlap and rms are recomputed at the returned transform, never one iteration
    stale. rms measures retained nearest-neighbor pairs, not true pose error.
    """
    source = _points(source, 3)
    if backend not in ('auto', 'native', 'numpy'):
        raise ValueError('backend must be auto, native or numpy')
    max_distance = _real_scalar(max_distance, 'max_distance')
    trim_fraction = _real_scalar(trim_fraction, 'trim_fraction')
    min_overlap = _real_scalar(min_overlap, 'min_overlap')
    translation_tolerance = _real_scalar(translation_tolerance, 'translation_tolerance')
    rotation_tolerance = _real_scalar(rotation_tolerance, 'rotation_tolerance')
    if (not math.isfinite(max_distance) or max_distance <= 0
            or not math.isfinite(trim_fraction) or not 0 < trim_fraction <= 1
            or not math.isfinite(min_overlap) or not 0 < min_overlap <= 1
            or type(min_correspondences) is not int or min_correspondences < 3
            or type(max_iterations) is not int or not 1 <= max_iterations <= 10000
            or not math.isfinite(translation_tolerance) or translation_tolerance <= 0
            or not math.isfinite(rotation_tolerance) or rotation_tolerance <= 0):
        raise ValueError('Invalid ICP distance, trim, overlap, iteration or tolerance parameter')
    if len(source) < min_correspondences:
        raise ValueError('Too few source points for min_correspondences')
    _centered_nondegenerate(source)
    transform = _pose(initial_transform)
    owned = not isinstance(target, NearestNeighborIndex)
    index = NearestNeighborIndex(target, backend=backend) if owned else target
    try:
        if index._closed:
            raise RuntimeError('Nearest-neighbor index is closed')
        _centered_nondegenerate(index._points)
        transformed = source @ transform[:3, :3].T + transform[:3, 3]
        current = _correspondences(index, transformed, max_distance, trim_fraction,
                                   min_correspondences, min_overlap)
        history = []
        converged = False
        for iteration in range(1, max_iterations + 1):
            inliers = current['inliers']
            update = _rigid_update(transformed[inliers], index._points[current['indices'][inliers]])
            transform = update @ transform
            transformed = source @ transform[:3, :3].T + transform[:3, 3]
            current = _correspondences(index, transformed, max_distance, trim_fraction,
                                       min_correspondences, min_overlap)
            # acos(trace) loses tiny rotations. atan2(skew norm, trace) remains
            # accurate near zero while still returning [0, pi].
            rotation = update[:3, :3]
            skew = np.array([rotation[2, 1] - rotation[1, 2],
                             rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
            rotation_step = math.atan2(float(np.linalg.norm(skew)) * .5,
                                      float(np.clip((np.trace(rotation) - 1.) * .5, -1., 1.)))
            translation_step = float(np.linalg.norm(update[:3, 3]))
            history.append({'iteration': iteration, 'rms': current['rms'],
                            'inlier_count': int(current['inliers'].sum()), 'overlap': current['overlap'],
                            'translation_step': translation_step, 'rotation_step': rotation_step})
            if translation_step <= translation_tolerance and rotation_step <= rotation_tolerance:
                converged = True
                break
        # Validate final selected geometry too: the last update can change the
        # selected pairs even when no next fitting iteration will be executed.
        final = current['inliers']
        _rigid_update(transformed[final], index._points[current['indices'][final]])
        return {'transform': transform, 'transformed_source': transformed,
                'target_indices': current['indices'], 'distances': current['distances'],
                'inliers': final, 'rms': current['rms'], 'overlap': current['overlap'],
                'inlier_fraction': float(final.mean()), 'iterations': len(history),
                'converged': converged, 'status': 'converged' if converged else 'max_iterations',
                'history': history, 'backend': index.backend}
    finally:
        if owned:
            index.close()
