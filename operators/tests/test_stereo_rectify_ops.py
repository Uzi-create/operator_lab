import unittest

import cv2
import numpy as np

from operators.stereo_ops import triangulate_stereo
from operators.stereo_rectify_ops import StereoRectifier
from operators.tests.test_stereo_ops import scene


class StereoRectifyTests(unittest.TestCase):
    def setUp(self):
        self.distortion = np.array([-.13, .025, .001, -.002, .004])
        self.points, self.left, self.right, self.camera, self.transform = scene(
            distortion=self.distortion)
        self.rectifier = StereoRectifier(self.camera, self.camera, self.transform,
                                        distortion_left=self.distortion,
                                        distortion_right=self.distortion)

    def test_exact_epipolar_geometry_and_stereo_link(self):
        rectified = self.rectifier.rectify_points(self.left, self.right)
        self.assertLess(rectified['vertical_errors_px'].max(), 1e-5)
        self.assertGreater((rectified['left'][:, 0]-rectified['right'][:, 0]).min(), 0.)
        self.assertAlmostEqual(self.rectifier.baseline, np.linalg.norm(self.transform[:3, 3]), delta=1e-10)
        result = triangulate_stereo(rectified['left'], rectified['right'],
                                    self.rectifier.rectified_camera,
                                    self.rectifier.rectified_camera,
                                    self.rectifier.T_right_from_left_rectified)
        self.assertEqual(result['valid_count'], len(self.points))
        np.testing.assert_allclose(result['points_left'],
                                   self.points @ self.rectifier.R_left.T, atol=1e-7)

    def test_cached_image_maps_align_projected_dots(self):
        size = (self.camera.height, self.camera.width)
        left = np.zeros(size, np.uint8)
        right = np.zeros(size, np.uint8)
        for x, y in self.left[:30]:
            cv2.circle(left, (round(x), round(y)), 5, 255, -1)
        for x, y in self.right[:30]:
            cv2.circle(right, (round(x), round(y)), 5, 255, -1)
        images = self.rectifier.rectify_images(left, right)
        self.assertEqual(images['left'].shape, size)
        self.assertEqual(images['right'].shape, size)
        self.assertEqual(images['camera'], self.rectifier.rectified_camera)
        self.assertAlmostEqual(images['baseline'], self.rectifier.baseline)
        points = self.rectifier.rectify_points(self.left[:30], self.right[:30])
        for image, positions in ((images['left'], points['left']),
                                 (images['right'], points['right'])):
            for x, y in positions:
                ix, iy = round(x), round(y)
                self.assertGreater(image[iy-2:iy+3, ix-2:ix+3].max(), 200)

    def test_strong_distortion_near_image_edges(self):
        rng = np.random.default_rng(18)
        distortion = np.array([-.3, .12, .002, -.002, -.03])
        camera = self.camera
        matrix = np.array([[camera.fx, 0., camera.cx], [0., camera.fy, camera.cy], [0., 0., 1.]])
        points = rng.uniform([-.9, -.65, 1.1], [.9, .65, 3.], (1500, 3))
        left = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix,
                                 distortion)[0].reshape(-1, 2)
        right_points = points @ self.transform[:3, :3].T + self.transform[:3, 3]
        right = cv2.projectPoints(right_points, np.zeros(3), np.zeros(3), matrix,
                                  distortion)[0].reshape(-1, 2)
        inside = ((left[:, 0] > 5) & (left[:, 0] < camera.width-5) &
                  (left[:, 1] > 5) & (left[:, 1] < camera.height-5) &
                  (right[:, 0] > 5) & (right[:, 0] < camera.width-5) &
                  (right[:, 1] > 5) & (right[:, 1] < camera.height-5))
        rectifier = StereoRectifier(camera, camera, self.transform,
                                    distortion_left=distortion, distortion_right=distortion)
        result = rectifier.rectify_points(left[inside], right[inside])
        self.assertTrue(result['valid'].all())
        self.assertLess(result['vertical_errors_px'].max(), 1e-4)
        self.assertLess(result['inverse_reprojection_errors_px'].max(), 1e-6)

    def test_invalid_pose_sizes_and_points(self):
        with self.assertRaises(ValueError):
            self.rectifier.rectify_points(self.left, self.right[:-1])
        with self.assertRaises(ValueError):
            self.rectifier.rectify_images(np.zeros((20, 20), np.uint8),
                                          np.zeros((20, 20), np.uint8))
        with self.assertRaises(ValueError):
            StereoRectifier(self.camera, self.camera, np.eye(4))
        reversed_baseline = self.transform.copy()
        reversed_baseline[:3, 3] = [.18, .004, .002]
        with self.assertRaises(ValueError):
            StereoRectifier(self.camera, self.camera, reversed_baseline)
        rotation = cv2.Rodrigues(np.array([0., 0., np.deg2rad(50.)]))[0]
        turned = np.eye(4)
        turned[:3, :3] = rotation
        turned[:3, 3] = -rotation @ np.array([.18, 0., 0.])
        rotated_rig = StereoRectifier(self.camera, self.camera, turned)
        self.assertGreater(rotated_rig.baseline, 0.)
        with self.assertRaises(ValueError):
            StereoRectifier(self.camera, self.camera, self.transform, alpha=2.)
        empty = self.rectifier.rectify_points(np.empty((0, 2)), np.empty((0, 2)))
        self.assertEqual(empty['left'].shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
