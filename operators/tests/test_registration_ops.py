import ctypes
import gc
from concurrent.futures import ThreadPoolExecutor
import math
import unittest
from unittest.mock import patch

import numpy as np

from operators import registration_ops as ops
from operators.registration_ops import NearestNeighborIndex, icp_point_to_point


def brute_force(target, query, limit=math.inf):
    """Independent scalar correspondence reference, stable original-index ties."""
    indices, squared = [], []
    for q in query:
        candidates = [sum(float(q[k] - t[k]) ** 2 for k in range(3)) for t in target]
        chosen = min(range(len(target)), key=candidates.__getitem__)
        if candidates[chosen] <= limit * limit:
            indices.append(chosen)
            squared.append(candidates[chosen])
        else:
            indices.append(-1)
            squared.append(math.inf)
    return np.array(indices), np.array(squared)


def rigid_pose(angle=.025, translation=(.018, -.012, .009)):
    c, s = math.cos(angle), math.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = translation
    return result


def apply_pose(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


class NearestNeighborTests(unittest.TestCase):
    def test_random_matches_scalar_reference_and_both_backends(self):
        rng = np.random.default_rng(703)
        target, query = rng.normal(size=(179, 3)), rng.normal(size=(73, 3))
        for limit in (math.inf, .45):
            expected_index, expected_squared = brute_force(target, query, limit)
            for backend in ('native', 'numpy'):
                with self.subTest(backend=backend, limit=limit), NearestNeighborIndex(target, backend=backend) as index:
                    found = index.query(query, max_distance=limit)
                    np.testing.assert_array_equal(found['indices'], expected_index)
                    np.testing.assert_allclose(found['squared_distances'], expected_squared, rtol=2e-16)
                    np.testing.assert_array_equal(found['valid'], expected_index >= 0)
                    np.testing.assert_array_equal(found['distances'], np.sqrt(found['squared_distances']))

    def test_chunk_boundaries_duplicate_and_equidistant_ties(self):
        rng = np.random.default_rng(704)
        target = rng.normal(10, 2, (4101, 3))
        target[[13, 2048, 4100]] = [1, 0, 0]
        target[3079] = [-1, 0, 0]
        query = np.zeros((517, 3))
        for backend in ('native', 'numpy'):
            with NearestNeighborIndex(target, backend=backend) as index:
                found = index.query(query, max_distance=1)
                np.testing.assert_array_equal(found['indices'], 13)
                np.testing.assert_array_equal(found['squared_distances'], 1)

    def test_native_numpy_exact_random_and_large_common_offset(self):
        rng = np.random.default_rng(705)
        for offset, scale in ((0., 1.), (1e12, .01), (0., 1e149), (0., 1e-140)):
            target = offset + rng.normal(size=(2101, 3)) * scale
            query = offset + rng.normal(size=(287, 3)) * scale
            with NearestNeighborIndex(target, backend='native') as native, NearestNeighborIndex(target, backend='numpy') as numpy:
                actual, expected = native.query(query), numpy.query(query)
                np.testing.assert_array_equal(actual['indices'], expected['indices'])
                np.testing.assert_array_equal(actual['squared_distances'], expected['squared_distances'])

    def test_inclusive_gate_zero_and_adjacent_float(self):
        for backend in ('native', 'numpy'):
            with NearestNeighborIndex([[0., 0, 0]], backend=backend) as index:
                query = np.array([[0., 0, 0], [1., 0, 0], [np.nextafter(1., 2.), 0, 0]])
                np.testing.assert_array_equal(index.query(query, max_distance=1)['indices'], [0, 0, -1])
                np.testing.assert_array_equal(index.query(query, max_distance=0)['indices'], [0, -1, -1])
                found = index.query(query, max_distance=np.float32(1))
                self.assertTrue(np.isinf(found['distances'][2]))

    def test_snapshot_readonly_unaligned_noncontiguous_and_empty(self):
        raw = np.empty(1 + 8 * 18, np.uint8)
        target = np.ndarray((6, 3), dtype=np.float64, buffer=raw, offset=1)
        target[:] = np.arange(18).reshape(6, 3)
        expected = target.copy()
        for backend in ('native', 'numpy'):
            target[:] = expected
            with NearestNeighborIndex(target[::-1], backend=backend) as index:
                target[:] = 999
                copy = index.target_points
                copy[:] = -999
                query = expected[::-2]
                query.setflags(write=False)
                np.testing.assert_array_equal(index.query(query)['indices'], [0, 2, 4])
                empty = index.query(np.empty((0, 3)))
                for values in empty.values():
                    self.assertEqual(values.shape, (0,))
                self.assertEqual(len(index), 6)

    def test_rejects_invalid_points_and_scalar_options(self):
        for backend in ('native', 'numpy'):
            for bad in ([], np.empty((0, 3)), [[0, 1]], [[math.nan, 0, 0]],
                        [[math.inf, 0, 0]], [[1e151, 0, 0]], [['0', '0', '0']], [[1j, 0, 0]]):
                with self.subTest(backend=backend, bad=bad), self.assertRaises(ValueError):
                    NearestNeighborIndex(bad, backend=backend)
            with NearestNeighborIndex([[0., 0, 0]], backend=backend) as index:
                for bad in (-1, -math.inf, math.nan, True, np.bool_(True), '1', [1], 1j):
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        index.query([[0., 0, 0]], max_distance=bad)
                with self.assertRaises(ValueError):
                    index.query([[math.nan, 0, 0]])
        with self.assertRaises(ValueError):
            NearestNeighborIndex([[0., 0, 0]], backend='approximate')

    def test_native_fallback_and_explicit_native_failure(self):
        with patch.object(ops, '_NATIVE', None), patch.object(ops.ctypes, 'CDLL', side_effect=OSError('missing')):
            with NearestNeighborIndex([[0., 0, 0]]) as index:
                self.assertEqual(index.backend, 'numpy')
                np.testing.assert_array_equal(index.query([[0., 0, 0]])['indices'], [0])
            with self.assertRaises(RuntimeError):
                NearestNeighborIndex([[0., 0, 0]], backend='native')

    def test_context_close_and_finalizer_release(self):
        library = ops._native('native')
        first = NearestNeighborIndex([[0., 0, 0]], backend='native')
        handle = first._handle
        first.close()
        first.close()
        for action in (lambda: first.query([[0., 0, 0]]), lambda: first.__enter__()):
            with self.assertRaises(RuntimeError):
                action()
        self.assertEqual(library.nn3_release(handle), 3)
        second = NearestNeighborIndex([[0., 0, 0]], backend='native')
        self.assertNotEqual(second._handle, handle)
        second_handle = second._handle
        del second
        gc.collect()
        self.assertEqual(library.nn3_release(second_handle), 3)

    def test_concurrent_queries_share_immutable_cache(self):
        rng = np.random.default_rng(708)
        target, query = rng.normal(size=(1800, 3)), rng.normal(size=(531, 3))
        with NearestNeighborIndex(target, backend='native') as index:
            expected = index.query(query)
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(lambda _: index.query(query), range(18)))
            for found in results:
                np.testing.assert_array_equal(found['indices'], expected['indices'])
                np.testing.assert_array_equal(found['squared_distances'], expected['squared_distances'])

    def test_native_abi_invalid_query_preserves_outputs(self):
        library = ops._native('native')
        ptr = ops._pointer
        query = np.zeros((4, 3))
        indices, squared = np.full(4, 319, np.int64), np.full(4, 811.)
        with NearestNeighborIndex(query, backend='native') as index:
            for bad_distance in (-1., math.nan):
                self.assertEqual(library.nn3_query(index._handle, ptr(query), 4, bad_distance,
                                                  ptr(indices, ctypes.c_int64), ptr(squared)), 1)
            query[-1, -1] = math.nan
            self.assertEqual(library.nn3_query(index._handle, ptr(query), 4, math.inf,
                                              ptr(indices, ctypes.c_int64), ptr(squared)), 1)
            query[-1, -1] = 0
            self.assertEqual(library.nn3_query(0, ptr(query), 4, math.inf,
                                              ptr(indices, ctypes.c_int64), ptr(squared)), 3)
            self.assertEqual(library.nn3_query(index._handle, ptr(query), ops._MAX_POINTS + 1, math.inf,
                                              ptr(indices, ctypes.c_int64), ptr(squared)), 1)
            np.testing.assert_array_equal(indices, 319)
            np.testing.assert_array_equal(squared, 811)

    def test_native_abi_overlaps_and_create_validation(self):
        library, ptr = ops._native('native'), ops._pointer
        target = np.zeros((5, 3))
        handle = ctypes.c_uint64(729)
        target[-1, -1] = math.nan
        self.assertEqual(library.nn3_create(ptr(target), 5, ctypes.byref(handle)), 1)
        self.assertEqual(handle.value, 729)
        target[-1, -1] = 0
        handle_alias = target.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        self.assertEqual(library.nn3_create(ptr(target), 5, handle_alias), 1)
        with NearestNeighborIndex(target, backend='native') as index:
            out = np.full(12, 83.)
            index_alias = ptr(out[1:], ctypes.c_int64)
            self.assertEqual(library.nn3_query(index._handle, ptr(target), 5, math.inf,
                                              index_alias, ptr(out)), 1)
            self.assertEqual(library.nn3_query(index._handle, ptr(target), 5, math.inf,
                                              ptr(out, ctypes.c_int64), ptr(target.ravel()[2:])), 1)
            self.assertEqual(library.nn3_query(index._handle, ptr(target), 5, math.inf,
                                              ptr(target.ravel()[1:], ctypes.c_int64), ptr(out)), 1)
            np.testing.assert_array_equal(out, 83)
            np.testing.assert_array_equal(target, 0)


class ICPTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(902)
        self.source = self.rng.uniform(-1, 1, (280, 3))
        self.pose = rigid_pose()
        self.target = apply_pose(self.source, self.pose)

    def assert_final_consistent(self, source, target, found, limit, trim=1.):
        np.testing.assert_allclose(found['transformed_source'], apply_pose(source, found['transform']), atol=0, rtol=0)
        with NearestNeighborIndex(target, backend='numpy') as index:
            nearest = index.query(found['transformed_source'], max_distance=limit)
        np.testing.assert_array_equal(found['target_indices'], nearest['indices'])
        np.testing.assert_array_equal(found['distances'], nearest['distances'])
        eligible = np.flatnonzero(nearest['valid'])
        keep = math.ceil(len(eligible) * trim)
        selected = eligible[np.argsort(nearest['squared_distances'][eligible], kind='stable')[:keep]]
        expected_mask = np.zeros(len(source), bool)
        expected_mask[selected] = True
        np.testing.assert_array_equal(found['inliers'], expected_mask)
        self.assertEqual(found['rms'], float(np.sqrt(np.mean(nearest['squared_distances'][expected_mask]))))
        self.assertEqual(found['overlap'], len(eligible) / len(source))
        self.assertEqual(found['history'][-1]['rms'], found['rms'])
        self.assertEqual(found['iterations'], len(found['history']))

    def test_known_rigid_transform_and_backend_agreement(self):
        native = icp_point_to_point(self.source, self.target, max_distance=.25, backend='native')
        numpy = icp_point_to_point(self.source, self.target, max_distance=.25, backend='numpy')
        self.assertTrue(native['converged'])
        self.assertEqual(native['status'], 'converged')
        self.assertLess(native['rms'], 1e-14)
        np.testing.assert_allclose(native['transform'], self.pose, atol=2e-14, rtol=0)
        for key in ('transform', 'target_indices', 'distances', 'inliers'):
            np.testing.assert_array_equal(native[key], numpy[key])
        self.assert_final_consistent(self.source, self.target, native, .25)

    def test_large_motion_with_initial_pose_and_cached_target(self):
        pose = rigid_pose(.8, (3., -2., 1.))
        target = apply_pose(self.source, pose)
        initial = pose.copy()
        initial[:3, 3] += [.015, -.01, .005]
        saved = initial.copy()
        with NearestNeighborIndex(target, backend='native') as index:
            found = icp_point_to_point(self.source, index, initial_transform=initial, max_distance=.2)
            self.assertFalse(index._closed)
            self.assertEqual(found['backend'], 'native')
            np.testing.assert_allclose(found['transform'], pose, atol=1e-14, rtol=0)
            np.testing.assert_array_equal(initial, saved)

    def test_noise_outliers_partial_overlap_with_trimming(self):
        source = np.vstack([self.source, self.rng.uniform(5, 8, (70, 3))])
        target = np.vstack([self.target + self.rng.normal(0, .001, self.target.shape),
                            self.rng.uniform(-8, -5, (100, 3))])
        result = icp_point_to_point(source, target, max_distance=.18, trim_fraction=.85,
                                    min_overlap=.7, backend='native')
        self.assertTrue(result['converged'])
        self.assertLess(result['rms'], .0025)
        self.assertLess(np.linalg.norm(result['transform'][:3, 3] - self.pose[:3, 3]), .0005)
        self.assertLess(np.linalg.norm(result['transform'][:3, :3] - self.pose[:3, :3]), .001)
        self.assertFalse(result['inliers'][-70:].any())
        self.assert_final_consistent(source, target, result, .18, .85)

    def test_last_iteration_outputs_recomputed_at_returned_pose(self):
        found = icp_point_to_point(self.source, self.target, max_distance=.25, max_iterations=1,
                                   translation_tolerance=1e-14, rotation_tolerance=1e-14)
        self.assertFalse(found['converged'])
        self.assertEqual(found['status'], 'max_iterations')
        self.assertGreater(found['history'][0]['translation_step'], .001)
        self.assert_final_consistent(self.source, self.target, found, .25)

    def test_planar_points_are_valid_and_rotation_is_proper(self):
        source = self.source.copy()
        source[:, 2] = 0
        target = apply_pose(source, self.pose)
        found = icp_point_to_point(source, target, max_distance=.2)
        np.testing.assert_allclose(found['transform'], self.pose, atol=2e-14, rtol=0)
        self.assertAlmostEqual(np.linalg.det(found['transform'][:3, :3]), 1., places=13)

    def test_no_overlap_and_insufficient_trim_fail(self):
        for kwargs in ({'max_distance': .001}, {'max_distance': .3, 'trim_fraction': .001},
                       {'max_distance': .3, 'min_correspondences': 1000}):
            with self.assertRaises(ValueError):
                icp_point_to_point(self.source, self.target, **kwargs)
        with self.assertRaises(ValueError):
            icp_point_to_point(self.source + 100, self.target, max_distance=.1)
        partial = np.vstack([self.source[:20], self.source[20:] + 100])
        with self.assertRaises(ValueError):
            icp_point_to_point(partial, self.target, max_distance=.2, min_overlap=.5)

    def test_collinear_coincident_and_collapsed_correspondences_fail(self):
        line = np.column_stack([np.arange(20), np.zeros((20, 2))])
        for source, target in ((line, line), (np.zeros((20, 3)), self.target),
                               (self.source, line)):
            with self.assertRaises(ValueError):
                icp_point_to_point(source, target, max_distance=10)
        # Whole target is nondegenerate, but all gated points choose origin.
        target = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])
        with self.assertRaises(ValueError):
            icp_point_to_point(self.source * .001, target, max_distance=1)

    def test_invalid_rigid_initial_pose_and_parameters(self):
        bad_poses = [np.eye(3), np.diag([2., 1, 1, 1]), np.diag([-1., 1, 1, 1]),
                     np.full((4, 4), math.nan), np.eye(4, dtype=complex)]
        for initial in bad_poses:
            with self.assertRaises(ValueError):
                icp_point_to_point(self.source, self.target, max_distance=.2, initial_transform=initial)
        for option, value in (('max_distance', math.inf), ('max_distance', 0), ('max_distance', True),
                              ('trim_fraction', 1.1), ('trim_fraction', math.nan), ('min_overlap', 0),
                              ('max_iterations', 0), ('max_iterations', True), ('min_correspondences', 2),
                              ('translation_tolerance', 0), ('rotation_tolerance', '1')):
            kwargs = {'max_distance': .2, option: value}
            with self.subTest(option=option, value=value), self.assertRaises(ValueError):
                icp_point_to_point(self.source, self.target, **kwargs)

    def test_closed_target_and_caller_inputs_untouched(self):
        source, target = self.source.copy(), self.target.copy()
        source.setflags(write=False)
        target.setflags(write=False)
        icp_point_to_point(source, target, max_distance=.25)
        np.testing.assert_array_equal(source, self.source)
        np.testing.assert_array_equal(target, self.target)
        with NearestNeighborIndex(target) as index:
            pass
        with self.assertRaises(RuntimeError):
            icp_point_to_point(source, index, max_distance=.25)


if __name__ == '__main__':
    unittest.main()
