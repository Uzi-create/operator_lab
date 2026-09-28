import unittest

import cv2
import numpy as np

from operators.ray_plane_ops import camera_rays, intersect_image_plane
from operators.robot_ops import Intrinsics


class RayPlaneTests(unittest.TestCase):
    def setUp(self):
        self.camera = Intrinsics(1280, 960, 900., 920., 640., 480.)
        self.matrix = np.array([[900., 0., 640.], [0., 920., 480.], [0., 0., 1.]])

    def test_pinhole_exact_plane_and_scale(self):
        pixels = np.array([[640., 480.], [740., 580.], [140., 170.]])
        output = intersect_image_plane(pixels, self.camera, [0, 0, 4], -8.)
        self.assertTrue(output['valid'].all())
        expected = np.column_stack(((pixels[:, 0]-640)*2/900,
                                    (pixels[:, 1]-480)*2/920, np.full(3, 2.)))
        np.testing.assert_allclose(output['points_frame'], expected, rtol=0, atol=1e-15)
        metric = intersect_image_plane(pixels, self.camera, [0, 0, 1], -2000.)
        np.testing.assert_allclose(metric['points_frame'], expected*1000, rtol=0, atol=1e-12)
        self.assertEqual(output['ray_distances'].shape, (3,))

    def test_rotated_camera_and_distortion_inversion(self):
        rng = np.random.default_rng(512)
        frame_points = np.column_stack((rng.uniform(-.3, .3, (100, 2)), np.full(100, 2.)))
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(np.array([.11, -.17, .03]))[0]
        transform[:3, 3] = [.08, -.03, .1]
        camera_points = (frame_points-transform[:3, 3]) @ transform[:3, :3]
        for length in (4, 5, 8, 12, 14):
            with self.subTest(length=length):
                distortion = np.zeros(length)
                distortion[:4] = [-.18, .05, .002, -.001]
                pixels = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3),
                                            self.matrix, distortion)[0].reshape(-1, 2)
                result = intersect_image_plane(pixels, self.camera, [0, 0, 1], -2.,
                                               T_frame_from_camera=transform, distortion=distortion)
                self.assertTrue(result['valid'].all())
                np.testing.assert_allclose(result['points_frame'], frame_points, rtol=0, atol=1e-10)
                self.assertLess(result['reprojection_errors_px'].max(), 1e-8)

    def test_parallel_grazing_behind_and_range_rejected(self):
        pixels = np.array([[640., 480.], [730., 480.], [1090., 480.]])
        parallel = intersect_image_plane(pixels, self.camera, [1, 0, 0], -1., min_incidence_cos=.2)
        np.testing.assert_array_equal(parallel['valid'], [False, False, True])
        self.assertTrue(np.isnan(parallel['points_frame'][:2]).all())
        behind = intersect_image_plane(pixels, self.camera, [0, 0, 1], 2.)
        self.assertFalse(behind['valid'].any())
        limited = intersect_image_plane(pixels, self.camera, [0, 0, 1], -2., max_ray_distance=2.)
        np.testing.assert_array_equal(limited['valid'], [True, False, False])

    def test_empty_outside_pixels_and_invalid_inputs(self):
        empty = camera_rays(np.empty((0, 2)), self.camera)
        self.assertEqual(empty['rays_camera'].shape, (0, 3))
        outside = camera_rays([[-10., 1000.]], self.camera)
        self.assertTrue(outside['valid'][0])
        for bad in ([[1]], [[np.nan, 0]], np.ones((3, 3)), [[1j, 2]], [[1e10, 2]]):
            with self.assertRaises(ValueError):
                camera_rays(bad, self.camera)
        for options in ({'plane_normal': [0, 0, 0], 'plane_offset': -1},
                        {'plane_normal': [0, 0, 1], 'plane_offset': np.inf},
                        {'plane_normal': [0, 0, 1], 'plane_offset': -1, 'min_incidence_cos': 1.1},
                        {'plane_normal': [0, 0, 1], 'plane_offset': -1, 'max_ray_distance': -1}):
            with self.assertRaises(ValueError):
                intersect_image_plane([[640., 480.]], self.camera, **options)
        bad_pose = np.eye(4)
        bad_pose[0, 0] = 2
        with self.assertRaises(ValueError):
            intersect_image_plane([[640., 480.]], self.camera, [0, 0, 1], -1.,
                                  T_frame_from_camera=bad_pose)

    def test_strict_inverse_accuracy_retries_unresolved_pixels(self):
        distortion = np.array([-.22, .07, .002, -.003, .01])
        pixels = np.array([[1000., 800.], [1100., 840.], [900., 100.], [200., 700.]])
        rays = camera_rays(pixels, self.camera, distortion=distortion,
                           max_reprojection_error_px=1e-12)
        self.assertTrue(rays['valid'].all())
        self.assertLessEqual(rays['reprojection_errors_px'].max(), 1e-12)
        unit = np.linalg.norm(rays['rays_camera'], axis=1)
        np.testing.assert_allclose(unit, 1., rtol=0, atol=2e-16)


if __name__ == '__main__':
    unittest.main()
