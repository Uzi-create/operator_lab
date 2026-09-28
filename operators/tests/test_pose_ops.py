import unittest
from unittest.mock import patch

import cv2
import numpy as np

from operators.pose_ops import estimate_pose_pnp, project_object_points
from operators.robot_ops import Intrinsics


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.camera = Intrinsics(1280, 960, 900., 920., 640., 480.)
        self.matrix = np.array([[900., 0., 640.], [0., 920., 480.], [0., 0., 1.]])
        self.rvec = np.array([.2, -.15, .1])
        self.tvec = np.array([.1, -.05, 2.])
        self.rng = np.random.default_rng(123)
        self.points = self.rng.uniform(-.3, .3, (120, 3))

    def pixels(self, points=None, rvec=None, tvec=None, distortion=None):
        return cv2.projectPoints(self.points if points is None else points,
                                 self.rvec if rvec is None else np.asarray(rvec, dtype=float),
                                 self.tvec if tvec is None else np.asarray(tvec, dtype=float),
                                 self.matrix, distortion)[0].reshape(-1, 2)

    def check_final(self, result, points, pixels, threshold, distortion=None, min_depth=0.):
        self.assertTrue(result['success'], result['reason'])
        transform = result['T_camera_from_object']
        rotation, translation = transform[:3, :3], transform[:3, 3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1., delta=1e-12)
        projected = self.pixels(points, cv2.Rodrigues(rotation)[0], translation, distortion)
        errors = np.linalg.norm(projected - pixels, axis=1)
        depth = (points @ rotation.T + translation)[:, 2]
        expected = (errors <= threshold) & (depth > min_depth) & np.isfinite(errors)
        np.testing.assert_array_equal(result['inliers'], expected)
        np.testing.assert_allclose(result['reprojection_errors'], errors, rtol=1e-12, atol=1e-12)
        self.assertEqual(result['inlier_count'], int(expected.sum()))
        self.assertAlmostEqual(result['rms'], np.sqrt(np.mean(errors[expected] ** 2)), delta=1e-12)

    def rotation_error(self, result, truth=None):
        truth = self.rvec if truth is None else truth
        relative = result['T_camera_from_object'][:3, :3] @ cv2.Rodrigues(np.asarray(truth, dtype=float))[0].T
        return np.linalg.norm(cv2.Rodrigues(relative)[0])

    def test_nonplanar_exact_pose_and_final_residual_contract(self):
        pixels = self.pixels()
        result = estimate_pose_pnp(self.points, pixels, self.camera)
        self.check_final(result, self.points, pixels, 3.)
        self.assertEqual(result['inlier_count'], len(self.points))
        self.assertFalse(result['planar'])
        self.assertFalse(result['ambiguous'])
        self.assertLess(result['rms'], 1e-8)
        self.assertLess(self.rotation_error(result), 1e-8)
        np.testing.assert_allclose(result['T_camera_from_object'][:3, 3], self.tvec, atol=1e-9)

    def test_noise_and_30_percent_outliers(self):
        pixels = self.pixels() + self.rng.normal(0., .3, (120, 2))
        pixels[:36] = self.rng.uniform([0., 0.], [1280., 960.], (36, 2))
        result = estimate_pose_pnp(self.points, pixels, self.camera,
                                   reprojection_threshold=1.5, min_inliers=75)
        self.check_final(result, self.points, pixels, 1.5)
        self.assertFalse(result['inliers'][:36].any())
        self.assertGreaterEqual(result['inlier_count'], 83)
        self.assertLess(result['rms'], .5)
        self.assertLess(self.rotation_error(result), np.deg2rad(.15))
        self.assertLess(np.linalg.norm(result['T_camera_from_object'][:3, 3] - self.tvec), .002)

    def test_standard_distortion_models(self):
        for length in (4, 5, 8, 12, 14):
            with self.subTest(length=length):
                distortion = np.zeros(length)
                distortion[:4] = [-.18, .04, .001, -.002]
                if length >= 5:
                    distortion[4] = .005
                if length >= 8:
                    distortion[5:8] = [.01, -.005, .001]
                if length >= 12:
                    distortion[8:12] = [.001, -.001, .002, -.002]
                if length == 14:
                    distortion[12:] = [.01, -.015]
                pixels = self.pixels(distortion=distortion)
                result = estimate_pose_pnp(self.points, pixels, self.camera, distortion=distortion)
                self.check_final(result, self.points, pixels, 3., distortion)
                self.assertLess(result['rms'], 1e-7)
                self.assertLess(self.rotation_error(result), 1e-7)
                np.testing.assert_allclose(result['T_camera_from_object'][:3, 3], self.tvec, atol=1e-8)

    def test_tilted_planar_target_arbitrary_object_frame(self):
        points = self.points.copy()
        points[:, 2] = 0.
        basis = cv2.Rodrigues(np.array([.3, .2, -.1]))[0]
        points = points @ basis.T + np.array([.1, -.15, .2])
        pixels = self.pixels(points, [.5, -.2, .1])
        result = estimate_pose_pnp(points, pixels, self.camera)
        self.check_final(result, points, pixels, 3.)
        self.assertTrue(result['planar'])
        self.assertFalse(result['ambiguous'])
        self.assertLess(result['rms'], 1e-7)
        self.assertLess(self.rotation_error(result, [.5, -.2, .1]), 1e-7)
        np.testing.assert_allclose(result['T_camera_from_object'][:3, 3], self.tvec, atol=1e-8)

    def test_planar_ambiguity_is_not_silently_accepted(self):
        points = self.points.copy()
        points[:, 2] = 0.
        pixels = self.pixels(points, [.02, -.02, .01], [0., 0., 8.])
        pixels += self.rng.normal(0., .05, pixels.shape)
        rejected = estimate_pose_pnp(points, pixels, self.camera)
        self.assertFalse(rejected['success'])
        self.assertEqual(rejected['reason'], 'ambiguous_planar_pose')
        self.assertTrue(rejected['planar'])
        self.assertTrue(rejected['ambiguous'])
        self.assertIsNone(rejected['T_camera_from_object'])
        self.assertGreaterEqual(len(rejected['candidates']), 2)
        self.assertLess(rejected['candidates'][1]['reference_rms'], .3)
        allowed = estimate_pose_pnp(points, pixels, self.camera, allow_ambiguous=True)
        self.check_final(allowed, points, pixels, 3.)
        self.assertTrue(allowed['ambiguous'])
        self.assertEqual(allowed['reason'], 'ambiguous_planar_pose_allowed')
        self.assertLess(allowed['rms'], .1)

    def test_planar_ambiguity_check_failure_is_explicit(self):
        points = self.points.copy()
        points[:, 2] = 0.
        pixels = self.pixels(points, [.5, -.2, .1])
        for result in ((0, (), ()),
                       (1, (np.zeros(3),), (np.ones(3),)),
                       (2, (np.full(3, np.nan), np.zeros(3)), (np.ones(3), np.ones(3)))):
            with self.subTest(result=result), patch('operators.pose_ops.cv2.solvePnPGeneric', return_value=result):
                pose = estimate_pose_pnp(points, pixels, self.camera)
            self.assertFalse(pose['success'])
            self.assertEqual(pose['reason'], 'planar_ambiguity_check_failed')
            self.assertTrue(pose['planar'])
            self.assertIsNone(pose['T_camera_from_object'])
        with patch('operators.pose_ops.cv2.solvePnPGeneric', side_effect=cv2.error('failed')):
            pose = estimate_pose_pnp(points, pixels, self.camera)
        self.assertEqual(pose['reason'], 'planar_ambiguity_check_failed')

    def test_planar_noise_outliers_and_distortion(self):
        points = self.points.copy()
        points[:, 2] = 0.
        rotation = np.array([.6, -.3, .1])
        distortion = np.array([-.18, .04, .001, -.002, .005])
        pixels = self.pixels(points, rotation, distortion=distortion)
        pixels += self.rng.normal(0., .2, pixels.shape)
        pixels[:24] = self.rng.uniform([0., 0.], [1280., 960.], (24, 2))
        result = estimate_pose_pnp(points, pixels, self.camera, distortion=distortion,
                                   reprojection_threshold=1., min_inliers=90)
        self.check_final(result, points, pixels, 1., distortion)
        self.assertTrue(result['planar'])
        self.assertFalse(result['ambiguous'])
        self.assertEqual(result['inlier_count'], 96)
        self.assertFalse(result['inliers'][:24].any())
        self.assertLess(result['rms'], .35)
        self.assertLess(self.rotation_error(result, rotation), np.deg2rad(.2))
        self.assertLess(np.linalg.norm(result['T_camera_from_object'][:3, 3] - self.tvec), .002)

    def test_metric_units_and_offset_normalization(self):
        offset = np.array([12., -45., 30.])
        points = 1000. * (self.points + offset)
        pixels = self.pixels()
        result = estimate_pose_pnp(points, pixels, self.camera)
        self.check_final(result, points, pixels, 3.)
        expected = 1000. * (self.tvec - cv2.Rodrigues(self.rvec)[0] @ offset)
        np.testing.assert_allclose(result['T_camera_from_object'][:3, 3], expected, atol=1e-6)
        self.assertLess(result['rms'], 1e-8)

    def test_projection_explicit_pose_distortion_and_negative_depth(self):
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(self.rvec)[0]
        transform[:3, 3] = self.tvec
        distortion = np.array([-.2, .04, .001, -.001, .005])
        projected = project_object_points(self.points, self.camera, transform, distortion=distortion)
        np.testing.assert_allclose(projected['pixels'], self.pixels(distortion=distortion), atol=1e-12)
        self.assertTrue(projected['in_front'].all())
        projected = project_object_points([[0., 0., -1.], [0., 0., 0.], [0., 0., 1.]],
                                          self.camera, np.eye(4))
        np.testing.assert_array_equal(projected['in_front'], [False, False, True])
        self.assertTrue(np.isnan(projected['pixels'][:2]).all())
        np.testing.assert_array_equal(projected['pixels'][2], [640., 480.])
        empty = project_object_points(np.empty((0, 3)), self.camera, np.eye(4))
        self.assertEqual(empty['pixels'].shape, (0, 2))

    def test_negative_depth_estimate_rejected(self):
        negative = np.array([.1, -.05, -2.])
        pixels = self.pixels(tvec=negative)
        # Force the exact mathematical behind-camera solution to ensure the
        # wrapper independently rejects it regardless of OpenCV's policy.
        center = self.points.mean(axis=0)
        scale = np.sqrt(np.mean(np.sum((self.points - center) ** 2, axis=1)))
        normalized_translation = (negative + cv2.Rodrigues(self.rvec)[0] @ center) / scale
        with patch('operators.pose_ops.cv2.solvePnPRansac', return_value=(
                True, self.rvec, normalized_translation, np.arange(len(self.points)))):
            result = estimate_pose_pnp(self.points, pixels, self.camera)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'insufficient_positive_depth_inliers')
        result = estimate_pose_pnp(self.points, self.pixels(), self.camera, min_depth=3.)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'insufficient_positive_depth_inliers')

    def test_quality_and_support_failures(self):
        pixels = self.pixels() + self.rng.normal(0., .3, (120, 2))
        result = estimate_pose_pnp(self.points, pixels, self.camera, max_rms=.01)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'final_pose_quality_failed')
        pixels[:50] = self.rng.uniform([0., 0.], [1280., 960.], (50, 2))
        result = estimate_pose_pnp(self.points, pixels, self.camera, min_inlier_ratio=.95)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'insufficient_positive_depth_inliers')
        for count in (0, 1, 4, 5):
            result = estimate_pose_pnp(self.points[:count], pixels[:count], self.camera)
            self.assertFalse(result['success'])
            self.assertEqual(result['reason'], 'insufficient_correspondences')

    def test_coincident_and_collinear_degeneracy(self):
        for points in (np.ones((12, 3)), np.column_stack((np.arange(12), np.zeros((12, 2))))):
            result = estimate_pose_pnp(points, self.pixels()[:12], self.camera)
            self.assertFalse(result['success'])
            self.assertEqual(result['reason'], 'degenerate_object_geometry')
        result = estimate_pose_pnp(self.points, np.column_stack((np.arange(120), np.ones(120))), self.camera)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'degenerate_image_geometry')

    def test_invalid_inputs(self):
        pixels = self.pixels()
        for points in (np.ones((10, 2)), self.points.astype(complex), np.full((120, 3), np.nan),
                       np.full((120, 3), np.inf), np.full((120, 3), 1e101)):
            with self.assertRaises(ValueError):
                estimate_pose_pnp(points, pixels, self.camera)
        with self.assertRaises(ValueError):
            estimate_pose_pnp(self.points, pixels[:-1], self.camera)
        with self.assertRaises(TypeError):
            estimate_pose_pnp(self.points, pixels, None)
        for arguments in ({'distortion': [0., 0., 0.]}, {'distortion': np.full(5, np.nan)},
                          {'distortion': np.zeros((5, 1))}, {'reprojection_threshold': 0.},
                          {'reprojection_threshold': True}, {'min_depth': -1.},
                          {'min_inlier_ratio': 1.1}, {'min_inliers': 4}, {'max_iterations': 0},
                          {'confidence': 1.}, {'confidence': np.nan}, {'max_rms': -1.},
                          {'allow_ambiguous': 1}, {'ambiguity_rms_delta': -1.}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                estimate_pose_pnp(self.points, pixels, self.camera, **arguments)


if __name__ == '__main__':
    unittest.main()
