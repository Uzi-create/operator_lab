import unittest

import cv2
import numpy as np

from operators.icp_plane_ops import icp_point_to_plane
from operators.registration_ops import NearestNeighborIndex


def make_scene(seed=521, face_count=600, outliers=120):
    rng = np.random.default_rng(seed)
    faces, normals = [], []
    for axis in range(3):
        points = rng.uniform(.05, 1., (face_count, 3))
        points[:, axis] = 0.
        normal = np.zeros(3)
        normal[axis] = -1.
        faces.append(points)
        normals.append(np.tile(normal, (face_count, 1)))
    clean, normals = np.vstack(faces), np.vstack(normals)
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.array([.025, -.02, .015]))[0]
    transform[:3, 3] = [.02, -.014, .01]
    chosen = np.concatenate([np.arange(axis*face_count, axis*face_count+int(.8*face_count))
                             for axis in range(3)])
    source = (clean[chosen]-transform[:3, 3])@transform[:3, :3]
    source += rng.normal(0, .0003, source.shape)
    source = np.vstack((source, rng.uniform(2., 3., (outliers, 3))))
    target = clean+rng.normal(0, .0003, clean.shape)
    return source, target, normals, transform, outliers


class PlaneICPTests(unittest.TestCase):
    def test_three_faces_known_pose_and_final_plane_residual(self):
        source, target, normals, truth, outliers = make_scene()
        result = icp_point_to_plane(source, target, normals, max_distance=.09,
                                    trim_fraction=.9, min_overlap=.7)
        self.assertTrue(result['converged'], result['status'])
        relative = result['transform'][:3, :3]@truth[:3, :3].T
        error_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(relative)[0])))
        error_m = float(np.linalg.norm(result['transform'][:3, 3]-truth[:3, 3]))
        self.assertLess(error_deg, .02)
        self.assertLess(error_m, .0002)
        self.assertFalse(result['inliers'][-outliers:].any())
        self.assertLess(result['rms'], .001)
        np.testing.assert_allclose(result['transformed_source'],
                                   source@result['transform'][:3, :3].T+result['transform'][:3, 3], atol=1e-14)
        with NearestNeighborIndex(target, backend='numpy') as independent:
            final = independent.query(result['transformed_source'], max_distance=.09)
        np.testing.assert_array_equal(result['target_indices'], final['indices'])
        np.testing.assert_allclose(result['point_distances'], final['distances'], atol=1e-14)
        valid = final['valid']
        raw = np.einsum('ij,ij->i', result['transformed_source'][valid]-target[final['indices'][valid]],
                        normals[final['indices'][valid]])
        np.testing.assert_allclose(result['plane_residuals'][valid], raw, atol=1e-15)
        self.assertTrue(np.isinf(result['plane_residuals'][~valid]).all())
        self.assertAlmostEqual(result['rms'], np.sqrt(np.mean(result['plane_residuals'][result['inliers']]**2)),
                               delta=1e-15)

    def test_cached_target_and_numpy_backend_agree(self):
        source, target, normals, _, _ = make_scene(face_count=120, outliers=15)
        with NearestNeighborIndex(target, backend='native') as index:
            cached = icp_point_to_plane(source, index, normals, max_distance=.09,
                                        trim_fraction=.9, min_overlap=.7)
        reference = icp_point_to_plane(source, target, normals, max_distance=.09,
                                       trim_fraction=.9, min_overlap=.7, backend='numpy')
        np.testing.assert_allclose(cached['transform'], reference['transform'], atol=1e-12)
        np.testing.assert_array_equal(cached['inliers'], reference['inliers'])
        self.assertEqual(cached['status'], reference['status'])

    def test_single_plane_is_six_dof_degenerate(self):
        rng = np.random.default_rng(37)
        target = np.column_stack((rng.uniform(-1, 1, (200, 2)), np.zeros(200)))
        normals = np.tile([0., 0., 1.], (len(target), 1))
        with self.assertRaisesRegex(ValueError, 'six pose freedoms'):
            icp_point_to_plane(target.copy(), target, normals, max_distance=.1,
                               min_correspondences=20)

    def test_invalid_normals_gate_and_pose(self):
        source, target, normals, _, _ = make_scene(face_count=50, outliers=0)
        for bad in (np.ones((10, 3)), np.zeros_like(normals),
                    np.full_like(normals, np.nan)):
            with self.assertRaises(ValueError):
                icp_point_to_plane(source, target, bad, max_distance=.1)
        for options in ({'max_distance': 0}, {'max_distance': .1, 'max_plane_residual': .2},
                        {'max_distance': .1, 'trim_fraction': 0},
                        {'max_distance': .1, 'min_overlap': 0},
                        {'max_distance': .1, 'max_iterations': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                icp_point_to_plane(source, target, normals, **options)
        pose = np.eye(4)
        pose[0, 0] = 2
        with self.assertRaises(ValueError):
            icp_point_to_plane(source, target, normals, max_distance=.1,
                               initial_transform=pose)


if __name__ == '__main__':
    unittest.main()
