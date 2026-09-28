import unittest

import cv2
import numpy as np

from operators.robot_ops import Intrinsics
from operators.stereo_ops import triangulate_stereo


def scene(seed=58, count=400, distortion=None):
    rng = np.random.default_rng(seed)
    points = rng.uniform([-.45, -.3, 1.5], [.45, .3, 4.], (count, 3))
    camera = Intrinsics(1280, 960, 900., 910., 640., 480.)
    matrix = np.array([[900., 0., 640.], [0., 910., 480.], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.array([.008, -.013, .005]))[0]
    transform[:3, 3] = [-.18, .004, .002]
    right_points = points @ transform[:3, :3].T + transform[:3, 3]
    left = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
    right = cv2.projectPoints(right_points, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
    return points, left, right, camera, transform


class StereoTests(unittest.TestCase):
    def test_exact_geometry_with_and_without_distortion(self):
        for distortion in (None, np.array([-.13, .025, .001, -.002, .004])):
            with self.subTest(distortion=distortion is not None):
                points, left, right, camera, transform = scene(distortion=distortion)
                result = triangulate_stereo(left, right, camera, camera, transform,
                                            distortion_left=distortion, distortion_right=distortion)
                self.assertEqual(result['valid_count'], len(points))
                self.assertTrue(result['valid'].all())
                np.testing.assert_allclose(result['points_left'], points, atol=1e-7)
                self.assertLess(result['reprojection_errors_px'].max(), 1e-7)
                np.testing.assert_allclose(result['depth_left'], points[:, 2], atol=1e-7)

    def test_noise_wrong_matches_low_parallax_and_outside(self):
        points, left, right, camera, transform = scene(count=800)
        rng = np.random.default_rng(164)
        left += rng.normal(0., .1, left.shape)
        right += rng.normal(0., .1, right.shape)
        right[0, 1] += 20.
        right[1, 0] = -1.
        infinite_direction_right = transform[:3, :3] @ np.array([0., 0., 1.])
        left[2] = [640., 480.]
        right[2] = [640.+900.*infinite_direction_right[0]/infinite_direction_right[2],
                    480.+910.*infinite_direction_right[1]/infinite_direction_right[2]]
        result = triangulate_stereo(left, right, camera, camera, transform,
                                    max_reprojection_error_px=1., min_parallax_deg=.5)
        self.assertEqual(result['reasons'][0], 'reprojection_failed')
        self.assertEqual(result['reasons'][1], 'outside_image')
        self.assertEqual(result['reasons'][2], 'low_parallax')
        self.assertTrue(np.isnan(result['points_left'][:3]).all())
        self.assertGreater(result['valid_count'], 790)
        errors = np.linalg.norm(result['points_left'][3:]-points[3:], axis=1)
        self.assertLess(np.nanmedian(errors), .006)

    def test_negative_depth_zero_baseline_and_invalid_inputs(self):
        points, left, right, camera, transform = scene(count=10)
        with self.assertRaises(ValueError):
            triangulate_stereo(left, right[:-1], camera, camera, transform)
        with self.assertRaises(ValueError):
            triangulate_stereo(left, right, camera, camera, np.eye(4))
        with self.assertRaises(ValueError):
            triangulate_stereo(left, right, camera, camera, transform, min_parallax_deg=90.)
        with self.assertRaises(ValueError):
            triangulate_stereo(left, right, camera, camera, transform, distortion_left=[1., 2.])
        empty = triangulate_stereo(np.empty((0, 2)), np.empty((0, 2)), camera, camera, transform)
        self.assertEqual(empty['points_left'].shape, (0, 3))
        self.assertEqual(empty['valid_count'], 0)
        simple = np.eye(4)
        simple[0, 3] = -.18
        behind = triangulate_stereo(np.array([[640., 480.]]), np.array([[670., 480.]]),
                                    camera, camera, simple)
        self.assertEqual(behind['reasons'][0], 'behind_or_too_far')
        self.assertFalse(behind['valid'][0])

    def test_midpoint_vs_dlt_same_noisy_correspondences(self):
        points, left, right, camera, transform = scene(count=800)
        rng = np.random.default_rng(165)
        left += rng.normal(0., .1, left.shape)
        right += rng.normal(0., .1, right.shape)
        midpoint = triangulate_stereo(left, right, camera, camera, transform)
        dlt = triangulate_stereo(left, right, camera, camera, transform, method='dlt')
        np.testing.assert_array_equal(midpoint['valid'], dlt['valid'])
        mask = midpoint['valid']
        delta = np.linalg.norm(midpoint['points_left'][mask]-dlt['points_left'][mask], axis=1)
        self.assertLess(np.percentile(delta, 95), .0002)
        self.assertLess(np.abs(midpoint['reprojection_errors_px'][mask]-
                                   dlt['reprojection_errors_px'][mask]).max(), .01)


if __name__ == '__main__':
    unittest.main()
