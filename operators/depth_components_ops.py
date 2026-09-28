"""Exact metric-jump connected components on an organized depth image."""
from collections import deque
import ctypes
import math
from numbers import Real
from pathlib import Path
import sys

import numpy as np

_LIBRARY = None


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


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
            library.depth_connected_components.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_int, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32)]
            library.depth_connected_components.restype = ctypes.c_int
            _LIBRARY = library
        except (OSError, AttributeError) as error:
            if backend == 'native':
                raise RuntimeError('Native depth-component library unavailable; run python build.py') from error
            return None
    return _LIBRARY


def _pointer(array, dtype):
    return array.ctypes.data_as(ctypes.POINTER(dtype))


def _reference(z, low, high, absolute, relative, connectivity, min_area):
    height, width = z.shape
    valid = np.isfinite(z) & (z > low) & (z <= high)
    seen = np.zeros(z.shape, bool)
    labels = np.zeros(z.shape, np.int32)
    rows = []
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        neighbors += ((-1, -1), (1, -1), (-1, 1), (1, 1))
    for y, x in zip(*np.nonzero(valid)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        pending = deque([(int(x), int(y))])
        points = []
        min_x = max_x = int(x)
        min_y = max_y = int(y)
        min_z = max_z = float(z[y, x])
        total = 0.
        while pending:
            px, py = pending.popleft()
            a = float(z[py, px])
            points.append((px, py))
            min_x, max_x = min(min_x, px), max(max_x, px)
            min_y, max_y = min(min_y, py), max(max_y, py)
            min_z, max_z = min(min_z, a), max(max_z, a)
            total += a
            for dx, dy in neighbors:
                nx, ny = px+dx, py+dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height or seen[ny, nx] or not valid[ny, nx]:
                    continue
                b = float(z[ny, nx])
                if abs(a-b) <= absolute+relative*min(a, b):
                    seen[ny, nx] = True
                    pending.append((nx, ny))
        if len(points) < min_area:
            continue
        label = len(rows)+1
        for px, py in points:
            labels[py, px] = label
        rows.append([len(points), min_x, min_y, max_x, max_y, min_z, max_z, total/len(points)])
    return labels, np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def depth_components(depth, *, depth_scale=1., min_depth=0., max_depth=10.,
                     absolute_jump=.02, relative_jump=0., connectivity=4,
                     min_area=1, backend='auto'):
    """Segment adjacent valid depths using exact local metric-jump connectivity.

    An edge connects two neighboring pixels when
    abs(z1-z2) <= absolute_jump + relative_jump*min(z1,z2).
    Both terms and output depths have the units of depth*depth_scale. A chain
    of individually valid edges can bridge a long ramp; this is transitive
    connectivity, not Euclidean clustering or semantic object detection.

    Labels are 0 for invalid/rejected pixels, positive components in raster
    first-pixel order. Bounding boxes are (x,y,width,height), end exclusive.
    NaN/Inf and nonpositive/out-of-range depths are holes. Input is not edited.
    """
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.dtype.kind not in 'uif':
        raise ValueError('depth must be a real 2D array')
    height, width = depth.shape
    if not (1 <= width <= 16384 and 1 <= height <= 16384 and depth.size <= 16_000_000):
        raise ValueError('depth dimensions must be 1..16384 and at most 16 million pixels')
    depth_scale = _scalar(depth_scale, 'depth_scale')
    min_depth = _scalar(min_depth, 'min_depth')
    max_depth = _scalar(max_depth, 'max_depth')
    absolute_jump = _scalar(absolute_jump, 'absolute_jump')
    relative_jump = _scalar(relative_jump, 'relative_jump')
    if (depth_scale <= 0 or min_depth < 0 or not min_depth < max_depth <= 1e100
            or not 0 <= absolute_jump <= 1e100 or not 0 <= relative_jump <= 1000):
        raise ValueError('Invalid depth scale, range or jump threshold')
    if connectivity not in (4, 8) or type(connectivity) is not int:
        raise ValueError('connectivity must be 4 or 8')
    if type(min_area) is not int or not 1 <= min_area <= depth.size:
        raise ValueError('min_area must be 1..number of pixels')
    z = np.ascontiguousarray(depth, dtype=np.float64)
    library = _native(backend)
    if library is None:
        with np.errstate(over='ignore', invalid='ignore'):
            labels, rows = _reference(z*depth_scale, min_depth, max_depth,
                                      absolute_jump, relative_jump, connectivity, min_area)
        selected_backend = 'numpy'
    else:
        labels = np.empty((height, width), np.int32)
        capacity = min(32, z.size)
        while True:
            rows = np.empty((capacity, 8), np.float64)
            count = ctypes.c_int32()
            code = library.depth_connected_components(
                _pointer(z, ctypes.c_double), width, height, depth_scale,
                min_depth, max_depth, absolute_jump, relative_jump, connectivity,
                min_area, _pointer(labels, ctypes.c_int32), _pointer(rows, ctypes.c_double),
                capacity, ctypes.byref(count))
            if code == 3 and capacity < z.size:
                capacity = min(capacity*2, z.size)
                continue
            if code == 2:
                raise MemoryError('Native depth-component allocation failed')
            if code:
                raise RuntimeError(f'Native depth-component call failed with code {code}')
            rows = rows[:count.value].copy()
            break
        selected_backend = 'native'
    regions = []
    for label, row in enumerate(rows, 1):
        area, x0, y0, x1, y1, zmin, zmax, zmean = row
        regions.append({'label': label, 'area': int(area),
                        'bbox_xywh': (int(x0), int(y0), int(x1-x0+1), int(y1-y0+1)),
                        'min_depth': float(zmin), 'max_depth': float(zmax),
                        'mean_depth': float(zmean)})
    return {'labels': labels, 'regions': regions, 'region_count': len(regions),
            'labeled_pixel_count': int(np.count_nonzero(labels)), 'backend': selected_backend}
