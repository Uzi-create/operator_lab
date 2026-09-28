import math
import unittest

import numpy as np

from operators.perception_ops import elevation_grid, pointcloud_to_depth
from operators.robot_ops import Intrinsics, depth_to_points


def reference_depth(points, camera):
    """Independent scalar rasterizer: ties preserve first source point."""
    depth = np.zeros((camera.height, camera.width))
    source_index = np.full(depth.shape, -1, dtype=np.int64)
    for index, (x, y, z) in enumerate(points):
        if z <= 0:
            continue
        u = float(x) / float(z) * camera.fx + camera.cx
        v = float(y) / float(z) * camera.fy + camera.cy
        if not (-.5 <= u < camera.width - .5 and -.5 <= v < camera.height - .5):
            continue
        col = min(math.floor(u + .5), camera.width - 1)
        row = min(math.floor(v + .5), camera.height - 1)
        if source_index[row, col] < 0 or z < depth[row, col]:
            depth[row, col] = z
            source_index[row, col] = index
    return depth, source_index


class DepthRasterTests(unittest.TestCase):
    def setUp(self):
        self.camera = Intrinsics(4, 3, 1, 1, 0, 0)

    def test_collision_nearest_depth_and_first_equal_source(self):
        points = np.array([[2, 2, 2], [1, 1, 1], [1.1, 1.1, 1],
                           [0, 0, -1], [0, 0, 0], [100, 0, 1]])
        result = pointcloud_to_depth(points, self.camera)
        self.assertEqual(result['depth'][1, 1], 1)
        self.assertEqual(result['source_index'][1, 1], 1)
        self.assertEqual(result['valid'].sum(), 1)
        np.testing.assert_array_equal(result['depth'][~result['valid']], 0)
        np.testing.assert_array_equal(result['source_index'][~result['valid']], -1)

    def test_pixel_footprint_half_ties_and_adjacent_float(self):
        points = np.array([[-.5, -.5, 1], [.5, .5, 2],
                           [np.nextafter(-.5, -np.inf), 0, 1],
                           [3.5, 0, 1], [0, 2.5, 1],
                           [np.nextafter(3.5, -np.inf), 2, 1],
                           [1.5, .5, 1]])
        result = pointcloud_to_depth(points, self.camera)
        self.assertEqual(result['valid'].sum(), 3)
        self.assertEqual(result['source_index'][0, 0], 0)
        self.assertEqual(result['source_index'][2, 3], 5)
        self.assertEqual(result['source_index'][1, 2], 6)

    def test_random_matches_independent_scalar_reference(self):
        rng = np.random.default_rng(34)
        points = rng.uniform([-2, -2, -.1], [5, 4, 3], (2000, 3))
        camera = Intrinsics(9, 7, 2.3, 3.1, 2.2, 1.5)
        expected_depth, expected_index = reference_depth(points, camera)
        result = pointcloud_to_depth(points[::-1][::-1], camera)
        np.testing.assert_array_equal(result['depth'], expected_depth)
        np.testing.assert_array_equal(result['source_index'], expected_index)
        np.testing.assert_array_equal(result['valid'], expected_index >= 0)

    def test_round_trip_depth_backprojection(self):
        rng = np.random.default_rng(4)
        camera = Intrinsics(40, 30, 83.2, 84.1, 20.3, 13.7)
        depth = rng.uniform(.3, 5, (30, 40))
        depth[::3, ::4] = 0
        points = depth_to_points(depth, camera)['points']
        result = pointcloud_to_depth(points, camera)
        np.testing.assert_array_equal(result['depth'], depth)
        np.testing.assert_array_equal(result['valid'], depth > 0)

    def test_empty_and_overflowing_projection(self):
        result = pointcloud_to_depth(np.empty((0, 3)), self.camera)
        self.assertFalse(result['valid'].any())
        np.testing.assert_array_equal(result['source_index'], -1)
        with np.errstate(all='raise'):
            result = pointcloud_to_depth([[1e308, 1e308, 1e-300]], self.camera)
        self.assertFalse(result['valid'].any())

    def test_invalid_points_camera_and_raster_size(self):
        for points in ([[1, 2]], [1, 2, 3], [[0, 0, np.nan]], [[np.inf, 0, 1]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                pointcloud_to_depth(points, self.camera)
        with self.assertRaises(TypeError):
            pointcloud_to_depth([[0, 0, 1]], None)
        with self.assertRaises(ValueError):
            pointcloud_to_depth(np.empty((0, 3)), Intrinsics(10**9, 10**9, 1, 1, 0, 0))


class ElevationGridTests(unittest.TestCase):
    def test_height_statistics_support_and_negative_coordinates(self):
        points = np.array([[-1, -1, -2], [-.9, -.8, 2], [.2, -.5, 4],
                           [.7, .4, 10], [.8, .7, 12]])
        result = elevation_grid(points, (-1, -1, 1, 1), 1, min_points=2)
        np.testing.assert_array_equal(result['count'], [[2, 1], [0, 2]])
        np.testing.assert_array_equal(result['valid'], [[True, False], [False, True]])
        np.testing.assert_allclose(result['min_z'], [[-2, 4], [np.nan, 10]], equal_nan=True)
        np.testing.assert_allclose(result['max_z'], [[2, 4], [np.nan, 12]], equal_nan=True)
        np.testing.assert_allclose(result['mean_z'], [[0, 4], [np.nan, 11]], equal_nan=True)
        np.testing.assert_array_equal(result['origin_xy'], [-1, -1])

    def test_half_open_bounds_and_partial_final_cell(self):
        points = np.array([[0, 0, 1], [1, 1, 2], [2.1, 2.2, 3],
                           [2.5, 1, 4], [1, 2.5, 5], [-1e-6, 0, 6],
                           [np.nextafter(2.5, -np.inf), 0, 7]])
        result = elevation_grid(points, (0, 0, 2.5, 2.5), 1)
        self.assertEqual(result['count'].shape, (3, 3))
        self.assertEqual(result['count'].sum(), 4)
        self.assertEqual(result['min_z'][0, 2], 7)
        self.assertEqual(result['max_z'][2, 2], 3)
        self.assertEqual(result['mean_z'][1, 1], 2)

    def test_random_matches_cell_membership_reference(self):
        rng = np.random.default_rng(22)
        points = rng.uniform([-3, -2, -1], [4, 3, 8], (3000, 3))
        bounds = (-2., -1., 3.25, 2.5)
        resolution = .75
        result = elevation_grid(points, bounds, resolution, min_points=8)
        for row in range(result['count'].shape[0]):
            for col in range(result['count'].shape[1]):
                x0 = bounds[0] + col * resolution
                y0 = bounds[1] + row * resolution
                members = points[(points[:, 0] >= x0)
                                 & (points[:, 0] < min(x0 + resolution, bounds[2]))
                                 & (points[:, 1] >= y0)
                                 & (points[:, 1] < min(y0 + resolution, bounds[3])), 2]
                self.assertEqual(result['count'][row, col], len(members))
                self.assertEqual(result['valid'][row, col], len(members) >= 8)
                if len(members):
                    self.assertEqual(result['min_z'][row, col], min(members))
                    self.assertEqual(result['max_z'][row, col], max(members))
                    self.assertAlmostEqual(result['mean_z'][row, col], float(np.mean(members)), places=12)

    def test_empty_and_all_points_outside(self):
        for points in (np.empty((0, 3)), np.array([[100., 100, 4]])):
            result = elevation_grid(points, (-1, -1, 1, 1), .5)
            self.assertFalse(result['valid'].any())
            self.assertEqual(result['count'].sum(), 0)
            for key in ('min_z', 'max_z', 'mean_z'):
                self.assertTrue(np.isnan(result[key]).all())

    def test_large_finite_heights_do_not_overflow_mean(self):
        with np.errstate(all='raise'):
            result = elevation_grid([[0, 0, 1e308], [0, 0, 1e308]], (-1, -1, 1, 1), 2)
            self.assertEqual(result['mean_z'][0, 0], 1e308)
            result = elevation_grid([[0, 0, -1e308], [0, 0, 1e308]], (-1, -1, 1, 1), 2)
            self.assertEqual(result['mean_z'][0, 0], 0)
            for sign in (-1, 1):
                limit = sign * np.finfo(np.float64).max
                result = elevation_grid([[0, 0, limit]] * 100, (-1, -1, 1, 1), 2)
                self.assertEqual(result['mean_z'][0, 0], limit)

    def test_bad_bounds_resolution_counts_and_points(self):
        points = np.array([[0., 0, 0]])
        for bounds in ((0, 0, 0, 1), (1, 0, 0, 1), (0, 0, np.inf, 1),
                       (0, 1), (-1e308, -1e308, 1e308, 1e308), (0, 0, 10000, 10000)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                elevation_grid(points, bounds, 1)
        for resolution in (0, -1, np.nan, np.inf, 1e-300, True, '1', [1]):
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                elevation_grid(points, (0, 0, 1, 1), resolution)
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                elevation_grid(points, (0, 0, 1, 1), 1, min_points=count)
        for bad_points in ([[0, 0]], [[0, 0, np.nan]], [[0, 0, np.inf]]):
            with self.assertRaises(ValueError):
                elevation_grid(bad_points, (0, 0, 1, 1), 1)


class NativeRasterTests(unittest.TestCase):
    def test_backends_views_and_pixel_boundaries(self):
        import operators.perception_ops as p
        rng = np.random.default_rng(441)
        camera = Intrinsics(31, 23, 17.7, 20.3, 14.2, 10.4)
        points = rng.uniform([-2, -2, -.3], [2, 2, 3], (3000, 3))
        for pixel in [-.5, .5, 10.5, 30.5]:
            for u in [np.nextafter(pixel, -np.inf), pixel, np.nextafter(pixel, np.inf)]:
                points = np.vstack((points, [(u-camera.cx)/camera.fx, 0., 1.]))
        for source in [points, points[::-1], points.astype(np.float32)]:
            native = p.pointcloud_to_depth(source, camera, backend='native')
            reference = p.pointcloud_to_depth(source, camera, backend='numpy')
            for key in reference:
                np.testing.assert_array_equal(native[key], reference[key])
            a = p.elevation_grid(source, (-1.2, -1.3, 1.7, 1.8), .1, backend='native')
            b = p.elevation_grid(source, (-1.2, -1.3, 1.7, 1.8), .1, backend='numpy')
            for key in ['count', 'valid', 'min_z', 'max_z', 'mean_z']:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-14, atol=1e-14, equal_nan=True)
        unaligned = np.ndarray(points.shape, np.float64, buffer=bytearray(points.nbytes+1), offset=1)
        unaligned[:] = points
        np.testing.assert_array_equal(p.pointcloud_to_depth(unaligned, camera, backend='native')['depth'],
                                      p.pointcloud_to_depth(points, camera, backend='numpy')['depth'])

    def test_rare_overflow_fallback_and_empty(self):
        import operators.perception_ops as p
        limit = np.finfo(np.float64).max
        for points in [np.empty((0, 3)), np.array([[0., 0., limit]]*100),
                       np.array([[0., 0., -limit], [0., 0., limit]]*50)]:
            a = p.elevation_grid(points, (-1, -1, 1, 1), 1, backend='native')
            b = p.elevation_grid(points, (-1, -1, 1, 1), 1, backend='numpy')
            for key in ['count', 'min_z', 'max_z', 'mean_z']:
                np.testing.assert_array_equal(a[key], b[key])

    def test_auto_fallback_and_bad_backend(self):
        import operators.perception_ops as p
        from unittest.mock import patch
        points = np.array([[0., 0., 1.]])
        camera = Intrinsics(3, 3, 1, 1, 1, 1)
        with patch.object(p, '_NATIVE', None), patch.object(p.ctypes, 'CDLL', side_effect=OSError('missing')):
            np.testing.assert_array_equal(p.pointcloud_to_depth(points, camera)['depth'],
                                          p.pointcloud_to_depth(points, camera, backend='numpy')['depth'])
            np.testing.assert_array_equal(p.elevation_grid(points, (-1, -1, 1, 1), 1)['count'],
                                          p.elevation_grid(points, (-1, -1, 1, 1), 1, backend='numpy')['count'])
            with self.assertRaises(RuntimeError):
                p.pointcloud_to_depth(points, camera, backend='native')
        for function, args in [(p.pointcloud_to_depth, (points, camera)),
                               (p.elevation_grid, (points, (-1, -1, 1, 1), 1))]:
            with self.assertRaises(ValueError):
                function(*args, backend='bogus')

    def test_direct_abi_invalid_never_writes(self):
        import ctypes
        import operators.perception_ops as p
        library = p._native('native')
        points = np.array([[0., 0., 1.], [0., 0., np.nan]])
        depth = np.full(4, 11.)
        index = np.full(4, 12, np.int64)
        def call(src, dst=depth, ids=index, n=2):
            return library.raster_depth(p._ptr(src), n, 2, 2, 1., 1., 0., 0.,
                                        p._ptr(dst), p._ptr(ids, ctypes.c_int64))
        self.assertEqual(call(points), 1)
        np.testing.assert_array_equal(depth, 11.)
        np.testing.assert_array_equal(index, 12)
        points[-1, -1] = 1
        original = points.copy()
        self.assertEqual(call(points, points.ravel()[:4]), 1)
        np.testing.assert_array_equal(points, original)
        self.assertEqual(call(points, depth, depth.view(np.int64)), 1)
        np.testing.assert_array_equal(depth, 11.)
        self.assertEqual(call(points, n=2**63), 1)
        low, high, mean = (np.full(4, v) for v in (13., 14., 15.))
        points[-1, -1] = np.inf
        self.assertEqual(library.raster_elevation(p._ptr(points), 2, -1., -1., 1., 1., 1., 2, 2,
                         p._ptr(index, ctypes.c_int64), p._ptr(low), p._ptr(high), p._ptr(mean)), 1)
        np.testing.assert_array_equal(index, 12)
        np.testing.assert_array_equal(low, 13.)
        np.testing.assert_array_equal(high, 14.)
        np.testing.assert_array_equal(mean, 15.)


if __name__ == '__main__':
    unittest.main()
