"""Vectorized raster operators for already transformed, finite 3D point clouds.

Depth uses the camera optical frame (+X right, +Y down, +Z forward).
Elevation uses a caller-supplied frame with +Z up. These frames differ: this
module never guesses a transform, gravity direction, or point-cloud units.
"""
import ctypes
import math
from pathlib import Path
import sys

import numpy as np

from operators.robot_ops import Intrinsics
from operators.robust_geometry import points_array


_MAX_IMAGE_PIXELS = 16_777_216
_MAX_GRID_CELLS = 4_000_000
_NATIVE = None


def _native(backend):
    global _NATIVE
    if backend not in ('auto', 'native', 'numpy'):
        raise ValueError('backend must be auto, native or numpy')
    if backend == 'numpy':
        return None
    if _NATIVE is None:
        path = Path(__file__).resolve().parent / {
            'win32': 'operators.dll', 'darwin': 'liboperators.dylib'}.get(sys.platform, 'liboperators.so')
        try:
            library = ctypes.CDLL(str(path))
            d, i = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64)
            library.raster_depth.argtypes = [d, ctypes.c_size_t, ctypes.c_int, ctypes.c_int] + [ctypes.c_double]*4 + [d, i]
            library.raster_depth.restype = ctypes.c_int
            library.raster_elevation.argtypes = [d, ctypes.c_size_t] + [ctypes.c_double]*5 + [ctypes.c_int]*2 + [i, d, d, d]
            library.raster_elevation.restype = ctypes.c_int
            _NATIVE = library
        except (OSError, AttributeError) as error:
            if backend == 'native':
                raise RuntimeError('Native raster library unavailable; run python build.py') from error
            return None
    return _NATIVE


def _native_points(points):
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError('Need finite Nx3 points')
    # The kernel validates all finite values before touching any output.
    return np.require(values, dtype=np.float64, requirements=['C', 'A'])


def _ptr(array, kind=ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(kind))


def pointcloud_to_depth(points, camera, *, backend='auto'):
    """Rasterize finite Nx3 camera-frame points with a deterministic z-buffer.

    Coordinates share the desired depth unit; depth is optical Z, not range.
    Pixels are nearest centers, using floor(coordinate + 0.5); exact half ties
    go toward the larger coordinate. The supported image footprint is
    [-0.5, width-0.5) x [-0.5, height-0.5). This footprint differs from the
    center-only bounds of robot_ops.project_points. Nonpositive Z, projections
    that overflow, and points outside this footprint are discarded.

    A pixel keeps the smallest positive Z; equal Z keeps the smallest original
    point index. Returns float64 ``depth`` (missing=0), bool ``valid``, and int64
    ``source_index`` (missing=-1), all (height,width). Empty input is (0,3).
    Nonfinite source coordinates are rejected, matching robot_ops. Intrinsics
    must describe undistorted pixels; no registration or external pose is used.
    Images above 16,777,216 pixels are rejected before allocating the rasters.
    auto uses the C++ fused kernel when available, otherwise the NumPy reference.
    """
    if not isinstance(camera, Intrinsics):
        raise TypeError('camera must be Intrinsics')
    size = camera.width * camera.height
    if size > _MAX_IMAGE_PIXELS:
        raise ValueError('Camera exceeds the 16,777,216-pixel raster limit')
    library = _native(backend)
    if library is not None:
        points = _native_points(points)
        shape = (camera.height, camera.width)
        depth = np.empty(shape, np.float64)
        source_index = np.empty(shape, np.int64)
        code = library.raster_depth(_ptr(points), len(points), camera.width, camera.height,
                                    camera.fx, camera.fy, camera.cx, camera.cy,
                                    _ptr(depth), _ptr(source_index, ctypes.c_int64))
        if code:
            raise ValueError('Native depth raster rejected input')
        return {'depth': depth, 'valid': source_index >= 0, 'source_index': source_index}
    points = points_array(points, 3, 0)
    depth = np.full(size, np.inf, dtype=np.float64)
    source_index = np.full(size, len(points), dtype=np.int64)

    selected = np.flatnonzero(points[:, 2] > 0)
    front = points[selected]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        u = front[:, 0] / front[:, 2] * camera.fx + camera.cx
        v = front[:, 1] / front[:, 2] * camera.fy + camera.cy
    inside = (np.isfinite(u) & np.isfinite(v)
              & (u >= -0.5) & (u < camera.width - 0.5)
              & (v >= -0.5) & (v < camera.height - 0.5))
    selected = selected[inside]
    # Bounds are checked before integer conversion. Adding 0.5 can round a
    # coordinate immediately below the upper edge onto width/height itself.
    x = np.minimum(np.floor(u[inside] + 0.5).astype(np.int64), camera.width - 1)
    y = np.minimum(np.floor(v[inside] + 0.5).astype(np.int64), camera.height - 1)
    flat = y * camera.width + x
    z = points[selected, 2]
    np.minimum.at(depth, flat, z)
    nearest = z == depth[flat]
    np.minimum.at(source_index, flat[nearest], selected[nearest])
    valid = source_index != len(points)
    depth[~valid] = 0
    source_index[~valid] = -1
    shape = (camera.height, camera.width)
    return {'depth': depth.reshape(shape), 'valid': valid.reshape(shape),
            'source_index': source_index.reshape(shape)}


def elevation_grid(points, bounds_xy, resolution, *, min_points=1, backend='auto'):
    """Aggregate +Z-up points into a bounded 2.5D XY grid without interpolation.

    bounds_xy=(xmin,ymin,xmax,ymax) is half-open on both upper edges. Rows grow
    in +Y and columns in +X. A cell index is floor((coordinate-min)/resolution);
    the last row/column can be narrower than resolution. Units are those of
    points. Points outside bounds are discarded; nonfinite points are rejected.

    Returns ``count`` (int64), ``min_z``, ``max_z``, ``mean_z`` (float64), and
    ``valid`` (count >= min_points). Empty cell heights are NaN. Cells below
    min_points retain their statistics but are marked invalid. ``origin_xy``,
    ``resolution`` and ``bounds_xy`` specify coordinates unambiguously.
    Empty input must have shape (0,3). At most 4,000,000 cells are allowed.
    auto uses the C++ fused kernel when available, otherwise the NumPy reference.

    Heights are geometric statistics; max_z is not a classified obstacle, and
    min_z is not automatically traversable floor. Supply a gravity-aligned
    cloud if these values should represent terrain elevation.
    """
    bounds = np.asarray(bounds_xy, dtype=np.float64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError('bounds_xy must be finite (xmin,ymin,xmax,ymax)')
    if (isinstance(resolution, (bool, np.bool_)) or not np.isscalar(resolution)
            or not isinstance(resolution, (int, float, np.integer, np.floating))
            or not math.isfinite(resolution) or resolution <= 0):
        raise ValueError('resolution must be a positive finite scalar')
    resolution = float(resolution)
    if type(min_points) is not int or min_points < 1:
        raise ValueError('min_points must be a positive integer')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        extents = bounds[2:] - bounds[:2]
        cells = extents / resolution
    if (not np.isfinite(cells).all() or (extents <= 0).any()
            or (cells <= 0).any() or (cells > _MAX_GRID_CELLS).any()):
        raise ValueError('Invalid bounds/resolution or excessive grid dimensions')
    nx, ny = (math.ceil(float(value)) for value in cells)
    size = nx * ny
    if size > _MAX_GRID_CELLS:
        raise ValueError('Grid exceeds the 4,000,000-cell limit')
    library = _native(backend)
    if library is not None:
        points = _native_points(points)
        shape = (ny, nx)
        count = np.empty(shape, np.int64)
        low, high, mean = (np.empty(shape, np.float64) for _ in range(3))
        code = library.raster_elevation(_ptr(points), len(points), *bounds, resolution, nx, ny,
                                        _ptr(count, ctypes.c_int64), _ptr(low), _ptr(high), _ptr(mean))
        if code:
            raise ValueError('Native elevation raster rejected input')
        return {'count': count, 'min_z': low, 'max_z': high, 'mean_z': mean,
                'valid': count >= min_points, 'origin_xy': bounds[:2].copy(),
                'resolution': resolution, 'bounds_xy': bounds.copy()}
    points = points_array(points, 3, 0)
    inside = ((points[:, 0] >= bounds[0]) & (points[:, 0] < bounds[2])
              & (points[:, 1] >= bounds[1]) & (points[:, 1] < bounds[3]))
    kept = points[inside]
    xy = np.floor((kept[:, :2] - bounds[:2]) / resolution).astype(np.int64)
    # A point already proven inside can round up at the final partial cell.
    xy[:, 0] = np.minimum(xy[:, 0], nx - 1)
    xy[:, 1] = np.minimum(xy[:, 1], ny - 1)
    flat = xy[:, 1] * nx + xy[:, 0]
    count = np.bincount(flat, minlength=size).astype(np.int64, copy=False)
    min_z = np.full(size, np.inf)
    max_z = np.full(size, -np.inf)
    np.minimum.at(min_z, flat, kept[:, 2])
    np.maximum.at(max_z, flat, kept[:, 2])
    # Divide before accumulating so a mean of large finite heights does not
    # first overflow an intermediate sum (e.g. two heights of 1e308).
    mean_z = np.bincount(flat, weights=kept[:, 2] / count[flat], minlength=size).astype(np.float64, copy=False)
    empty = count == 0
    if np.isinf(mean_z).any():
        # At DBL_MAX, division rounding can still make repeated partial means
        # overflow by an ULP. Rescale within each cell for this rare case.
        scale = np.maximum(np.abs(min_z), np.abs(max_z))
        scale[empty | (scale == 0)] = 1
        normalized = np.bincount(flat, weights=kept[:, 2] / scale[flat], minlength=size)
        mean_z = normalized / np.maximum(count, 1) * scale
    min_z[empty] = np.nan
    max_z[empty] = np.nan
    mean_z[empty] = np.nan
    shape = (ny, nx)
    return {'count': count.reshape(shape), 'min_z': min_z.reshape(shape),
            'max_z': max_z.reshape(shape), 'mean_z': mean_z.reshape(shape),
            'valid': (count >= min_points).reshape(shape),
            'origin_xy': bounds[:2].copy(), 'resolution': resolution,
            'bounds_xy': bounds.copy()}
