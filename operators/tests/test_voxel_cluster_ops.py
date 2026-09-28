import ctypes
import unittest

import numpy as np

from operators import voxel_cluster_ops as ops
from operators.voxel_cluster_ops import voxel_clusters


class VoxelClusterTests(unittest.TestCase):
    def compare(self, points, voxel_size, **kwargs):
        native = voxel_clusters(points, voxel_size, backend='native', **kwargs)
        reference = voxel_clusters(points, voxel_size, backend='numpy', **kwargs)
        for name in ('labels', 'voxel_keys', 'voxel_labels'):
            np.testing.assert_array_equal(native[name], reference[name])
        self.assertEqual(native['cluster_count'], reference['cluster_count'])
        for actual, expected in zip(native['clusters'], reference['clusters']):
            self.assertEqual(actual.keys(), expected.keys())
            for name in actual:
                if isinstance(actual[name], np.ndarray):
                    np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-14)
                else:
                    self.assertEqual(actual[name], expected[name])
        return native

    def test_known_clusters_negative_coordinates_and_stats(self):
        points = np.array([[.1, .1, .1], [.6, .1, .1], [.2, .2, .1],
                           [4.1, 4.1, 4.1], [4.2, 4.3, 4.1],
                           [-1.1, -1.1, -1.1], [10., 0., 0.]])
        found = self.compare(points, voxel_size=.5, connectivity=6, min_points=2)
        self.assertEqual(found['cluster_count'], 2)
        np.testing.assert_array_equal(found['labels'], [1, 1, 1, 2, 2, 0, 0])
        self.assertEqual([item['point_count'] for item in found['clusters']], [3, 2])
        self.assertEqual([item['voxel_count'] for item in found['clusters']], [2, 1])
        np.testing.assert_allclose(found['clusters'][0]['centroid'], points[:3].mean(axis=0))
        np.testing.assert_array_equal(found['clusters'][1]['min_xyz'], [4.1, 4.1, 4.1])
        np.testing.assert_array_equal(points[5]//.5, [-3, -3, -3])

    def test_six_eighteen_twenty_six_edges_and_voxel_filter(self):
        points = np.array([[.1, .1, .1], [1.1, 1.1, .1], [2.1, 2.1, 1.1]])
        self.assertEqual(self.compare(points, 1., connectivity=6, min_points=1)['cluster_count'], 3)
        eighteen = self.compare(points, 1., connectivity=18, min_points=1)
        np.testing.assert_array_equal(eighteen['labels'], [1, 1, 2])
        twenty_six = self.compare(points, 1., connectivity=26, min_points=1)
        np.testing.assert_array_equal(twenty_six['labels'], [1, 1, 1])
        filtered = self.compare(points, 1., connectivity=18, min_voxels=2, min_points=1)
        np.testing.assert_array_equal(filtered['labels'], [1, 1, 0])

    def test_transitive_chain_is_not_radius_clustering(self):
        points = np.array([[i+.01, 0, 0] for i in range(20)])
        result = self.compare(points, 1., connectivity=6, min_points=1)
        self.assertEqual(result['cluster_count'], 1)
        self.assertEqual(result['clusters'][0]['point_count'], 20)
        self.assertGreater(np.linalg.norm(points[-1]-points[0]), 18.)

    def test_random_views_and_empty(self):
        rng = np.random.default_rng(173)
        points = rng.normal(0, 1, (1300, 3))[:, ::-1]
        for connectivity in (6, 18, 26):
            for min_points in (1, 3, 9):
                self.compare(points, .35, connectivity=connectivity,
                             min_voxels=2, min_points=min_points)
        empty = self.compare(np.empty((0, 3)), 1.)
        self.assertEqual(empty['cluster_count'], 0)
        self.assertEqual(empty['labels'].shape, (0,))

    def test_native_rejects_duplicate_keys_without_mutating_outputs(self):
        library = ops._native('native')
        keys = np.array([[0, 0, 0], [0, 0, 0]], np.int64)
        labels = np.full(2, 73, np.int32)
        count = ctypes.c_int32(87)
        code = library.voxel_connected_components(
            keys.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), 2, 6, 1,
            labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.byref(count))
        self.assertEqual(code, 1)
        np.testing.assert_array_equal(labels, 73)
        self.assertEqual(count.value, 87)

    def test_invalid_inputs(self):
        for bad in ([[0, 0]], [[np.nan, 0, 0]], [[1j, 0, 0]], np.zeros((4, 4))):
            with self.assertRaises(ValueError):
                voxel_clusters(bad, 1.)
        for options in ({'voxel_size': 0}, {'voxel_size': np.nan},
                        {'connectivity': 4}, {'min_voxels': 0}, {'min_points': 0},
                        {'backend': 'invalid'}):
            parameters = dict(options)
            size = parameters.pop('voxel_size', 1.)
            with self.subTest(options=options), self.assertRaises(ValueError):
                voxel_clusters(np.zeros((3, 3)), size, **parameters)


if __name__ == '__main__':
    unittest.main()
