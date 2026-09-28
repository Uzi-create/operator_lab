"""Exact voxel-grid graph clustering for unorganized 3D point clouds."""
from collections import deque
import ctypes
import math
from numbers import Real
from pathlib import Path
import sys

import numpy as np

_LIBRARY = None


def _native(backend):
    global _LIBRARY
    if backend not in ('auto', 'native', 'numpy'):
        raise ValueError('backend must be auto, native or numpy')
    if backend == 'numpy':
        return None
    if _LIBRARY is None:
        name = {'win32': 'operators.dll', 'darwin': 'liboperators.dylib'}.get(sys.platform, 'liboperators.so')
        try:
            library = ctypes.CDLL(str(Path(__file__).resolve().parent/name))
            library.voxel_connected_components.argtypes = [
                ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t, ctypes.c_int,
                ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32)]
            library.voxel_connected_components.restype = ctypes.c_int
            _LIBRARY = library
        except (OSError, AttributeError) as error:
            if backend == 'native':
                raise RuntimeError('Native voxel-component library unavailable; run python build.py') from error
            return None
    return _LIBRARY


def _offsets(connectivity):
    return [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if 0 < abs(dx)+abs(dy)+abs(dz) <= (1 if connectivity == 6 else 2 if connectivity == 18 else 3)]


def _reference(keys, connectivity, min_voxels):
    lookup = {tuple(row): i for i, row in enumerate(keys)}
    visited = np.zeros(len(keys), bool)
    labels = np.zeros(len(keys), np.int32)
    offsets = _offsets(connectivity)
    count = 0
    for start in range(len(keys)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            key = keys[current]
            for dx, dy, dz in offsets:
                neighbor = lookup.get((key[0]+dx, key[1]+dy, key[2]+dz))
                if neighbor is not None and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if len(component) >= min_voxels:
            count += 1
            labels[component] = count
    return labels, count


def voxel_clusters(points, voxel_size, *, connectivity=26,
                   min_voxels=1, min_points=3, backend='auto'):
    """Group occupied voxel cells by 6/18/26 graph adjacency.

    This is an exact grid-graph partition, not an exact Euclidean radius
    clustering: diagonal voxels may touch while their points are far apart.
    Components are transitive and labels follow the lexicographically first
    occupied voxel; 0 means a filtered component. No downsampling occurs.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'uif':
        raise ValueError('points must be a real Nx3 array')
    if len(points) > 2_000_000:
        raise ValueError('At most two million points supported')
    points = np.ascontiguousarray(points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError('Points must be finite')
    if isinstance(voxel_size, (bool, np.bool_)) or not isinstance(voxel_size, Real):
        raise ValueError('voxel_size must be a finite positive scalar')
    voxel_size = float(voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('voxel_size must be a finite positive scalar')
    if type(connectivity) is not int or connectivity not in (6, 18, 26):
        raise ValueError('connectivity must be 6, 18 or 26')
    if (type(min_voxels) is not int or min_voxels < 1 or
            type(min_points) is not int or min_points < 1):
        raise ValueError('min_voxels and min_points must be positive integers')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        scaled = points/voxel_size
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 9e18):
        raise ValueError('Voxel index exceeds supported int64 range')
    library = _native(backend)
    if len(points):
        keys, inverse = np.unique(np.floor(scaled).astype(np.int64),
                                  axis=0, return_inverse=True)
        keys = np.ascontiguousarray(keys)
    else:
        keys, inverse = np.empty((0, 3), np.int64), np.empty(0, np.int64)
    if len(keys) == 0 or min_voxels > len(keys):
        voxel_labels = np.zeros(len(keys), np.int32)
        raw_count = 0
    elif library is None:
        voxel_labels, raw_count = _reference(keys, connectivity, min_voxels)
    else:
        voxel_labels = np.empty(len(keys), np.int32)
        count = ctypes.c_int32()
        code = library.voxel_connected_components(
            keys.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(keys),
            connectivity, min_voxels,
            voxel_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.byref(count))
        if code == 2:
            raise MemoryError('Native voxel clustering allocation failed')
        if code:
            raise RuntimeError(f'Native voxel clustering returned code {code}')
        raw_count = count.value
    point_labels = voxel_labels[inverse]
    counts = np.bincount(point_labels, minlength=raw_count+1)
    keep = np.flatnonzero(counts[1:] >= min_points)+1
    if len(keep) > 100_000:
        raise ValueError('More than 100000 output clusters; raise min_points/min_voxels or voxel_size')
    remap = np.zeros(raw_count+1, np.int32)
    remap[keep] = np.arange(1, len(keep)+1, dtype=np.int32)
    voxel_labels = remap[voxel_labels]
    point_labels = remap[point_labels]
    total = len(keep)
    clusters = []
    if total:
        accepted = point_labels > 0
        labeled = point_labels[accepted]
        xyz = points[accepted]
        area = np.bincount(labeled, minlength=total+1)
        voxels = np.bincount(voxel_labels, minlength=total+1)
        centroid = np.column_stack([np.bincount(labeled, weights=xyz[:, axis], minlength=total+1)/
                                    np.maximum(area, 1) for axis in range(3)])
        low = np.full((total+1, 3), np.inf)
        high = np.full((total+1, 3), -np.inf)
        for axis in range(3):
            np.minimum.at(low[:, axis], labeled, xyz[:, axis])
            np.maximum.at(high[:, axis], labeled, xyz[:, axis])
        clusters = [{'label': i, 'point_count': int(area[i]), 'voxel_count': int(voxels[i]),
                     'centroid': centroid[i].copy(), 'min_xyz': low[i].copy(),
                     'max_xyz': high[i].copy()} for i in range(1, total+1)]
    return {'labels': point_labels, 'voxel_keys': keys, 'voxel_labels': voxel_labels,
            'clusters': clusters, 'cluster_count': total,
            'backend': 'native' if library is not None else 'numpy'}
