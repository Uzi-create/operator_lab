import unittest

import numpy as np

from operators.correspondence_ops import estimate_rigid_transform_robust


def fixture(n=240, noise=0., outlier_fraction=0., seed=14):
    rng = np.random.default_rng(seed)
    source = rng.uniform(-1., 1., (n, 3))
    axis = np.array([.4, -.7, .3])
    axis /= np.linalg.norm(axis)
    angle = 2.3
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, [1.2, -.8, .35]
    target = source @ rotation.T + transform[:3, 3]
    target += rng.normal(0, noise, target.shape)
    outliers = rng.choice(n, int(n * outlier_fraction), replace=False)
    target[outliers] = rng.uniform(-4, 4, (len(outliers), 3))
    return source, target, transform, outliers


class CorrespondenceTests(unittest.TestCase):
    def assert_consistent(self, result, source, target, threshold):
        transform = result['transform'] if result['success'] else result['candidate_transform']
        if transform is None:
            return
        errors = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)
        np.testing.assert_allclose(result['residuals'], errors, rtol=1e-13, atol=1e-14)
        np.testing.assert_array_equal(result['inliers'], errors <= threshold)
        self.assertEqual(result['inlier_count'], int(result['inliers'].sum()))
        self.assertEqual(result['inlier_ratio'], result['inlier_count'] / len(source))
        if result['inlier_count']:
            self.assertAlmostEqual(result['rms'], np.sqrt(np.mean(errors[result['inliers']] ** 2)), delta=1e-12)
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1., places=12)

    def test_clean_large_rotation_translation(self):
        source, target, transform, _ = fixture()
        result = estimate_rigid_transform_robust(source, target)
        self.assertTrue(result['success'], result['reason'])
        np.testing.assert_allclose(result['transform'], transform, atol=1e-12)
        self.assertEqual(result['iterations'], 0)
        self.assert_consistent(result, source, target, .01)

    def test_noise_and_thirty_to_sixty_percent_mismatches(self):
        for fraction in (.3, .45, .6):
            with self.subTest(fraction=fraction):
                source, target, transform, outliers = fixture(noise=.0005, outlier_fraction=fraction)
                result = estimate_rigid_transform_robust(source, target, threshold=.003,
                                                         min_inlier_ratio=.35, seed=3)
                self.assertTrue(result['success'], result['reason'])
                self.assertEqual(result['inlier_count'], len(source) - len(outliers))
                self.assertFalse(result['inliers'][outliers].any())
                np.testing.assert_allclose(result['transform'], transform, atol=.0005)
                self.assertLess(result['rms'], .0011)
                self.assertLessEqual(result['iterations'], 512)
                self.assert_consistent(result, source, target, .003)

    def test_planar_points_are_observable(self):
        source, _, transform, _ = fixture(n=80)
        source[:, 2] = .2
        target = source @ transform[:3, :3].T + transform[:3, 3]
        target[:20] += 4
        result = estimate_rigid_transform_robust(source, target, threshold=1e-6)
        self.assertTrue(result['success'], result['reason'])
        np.testing.assert_allclose(result['transform'], transform, atol=1e-12)

    def test_collinear_repeated_and_near_collinear_fail(self):
        rng = np.random.default_rng(9)
        line = np.column_stack((np.linspace(-1, 1, 30), np.zeros((30, 2))))
        almost_line = line + rng.normal(0, 1e-12, line.shape)
        for source in (line, almost_line, np.ones((20, 3))):
            result = estimate_rigid_transform_robust(source, source + [1, 2, 3], max_iterations=32)
            self.assertFalse(result['success'])
            self.assertIsNone(result['transform'])
            self.assertIn(result['reason'], ('degenerate_geometry', 'degenerate_inliers'))

    def test_low_support_fails_and_keeps_candidate_separate(self):
        source, target, _, _ = fixture(outlier_fraction=.6)
        result = estimate_rigid_transform_robust(source, target, min_inlier_ratio=.8)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'insufficient_inliers')
        self.assertIsNone(result['transform'])
        self.assertIsNotNone(result['candidate_transform'])
        self.assert_consistent(result, source, target, .01)

    def test_empty_and_too_few_points(self):
        for n in (0, 1, 2, 5):
            source = np.ones((n, 3))
            result = estimate_rigid_transform_robust(source, source)
            self.assertFalse(result['success'])
            self.assertEqual(result['reason'], 'insufficient_points')
            self.assertEqual(result['iterations'], 0)
            self.assertEqual(result['inliers'].shape, (n,))
            self.assertTrue(np.isinf(result['residuals']).all())

    def test_three_noncollinear_points_can_be_requested(self):
        source, target, transform, _ = fixture(n=3)
        result = estimate_rigid_transform_robust(source, target, min_inliers=3)
        self.assertTrue(result['success'], result['reason'])
        np.testing.assert_allclose(result['transform'], transform, atol=1e-12)

    def test_reflection_is_not_returned_as_rotation(self):
        source, _, _, _ = fixture(n=90)
        target = source * [-1, 1, 1]
        result = estimate_rigid_transform_robust(source, target, threshold=1e-5, min_inlier_ratio=.9)
        self.assertFalse(result['success'])
        self.assertIsNone(result['transform'])
        self.assert_consistent(result, source, target, 1e-5)

    def test_scale_change_is_not_rigid_motion(self):
        source, _, _, _ = fixture(n=90)
        result = estimate_rigid_transform_robust(source, 1.1 * source, threshold=.001,
                                                min_inlier_ratio=.9)
        self.assertFalse(result['success'])

    def test_reproducible_noncontiguous_read_only_inputs(self):
        source, target, _, _ = fixture(noise=.0005, outlier_fraction=.55)
        source, target = source[::-2], target[::-2]
        saved_source, saved_target = source.copy(), target.copy()
        source.flags.writeable = False
        target.flags.writeable = False
        a = estimate_rigid_transform_robust(source, target, min_inlier_ratio=.3, seed=73)
        b = estimate_rigid_transform_robust(source, target, min_inlier_ratio=.3, seed=73)
        self.assertTrue(a['success'])
        np.testing.assert_array_equal(a['transform'], b['transform'])
        np.testing.assert_array_equal(a['inliers'], b['inliers'])
        np.testing.assert_array_equal(source, saved_source)
        np.testing.assert_array_equal(target, saved_target)
        self.assertEqual(a['iterations'], b['iterations'])

    def test_units_and_large_world_origin(self):
        for scale, origin in ((1e-8, 0.), (1e8, 0.), (1., 1e9)):
            with self.subTest(scale=scale, origin=origin):
                source, target, transform, _ = fixture(outlier_fraction=.3)
                source, target = source * scale + origin, target * scale + origin
                threshold = .003 * scale
                result = estimate_rigid_transform_robust(source, target, threshold=threshold)
                self.assertTrue(result['success'], result['reason'])
                np.testing.assert_allclose(result['transform'][:3, :3], transform[:3, :3], atol=2e-7)
                self.assert_consistent(result, source, target, threshold)

    def test_iteration_budget(self):
        source, target, _, _ = fixture(outlier_fraction=.8)
        result = estimate_rigid_transform_robust(source, target, max_iterations=1)
        self.assertLessEqual(result['iterations'], 1)

    def test_good_initial_consensus_uses_adaptive_budget(self):
        rng = np.random.default_rng(608)
        half = rng.uniform(-3, 3, (35, 3))
        good = np.concatenate((half, -half))
        bad = np.tile(np.concatenate((np.eye(3), -np.eye(3))), (5, 1))
        source = np.concatenate((good, bad))
        target = np.concatenate((good, -bad)) + [1, 2, 3]
        result = estimate_rigid_transform_robust(source, target, threshold=.001)
        self.assertTrue(result['success'], result['reason'])
        self.assertEqual(result['inlier_count'], 70)
        self.assertGreater(result['iterations'], 0)
        self.assertLessEqual(result['iterations'], 18)
        self.assert_consistent(result, source, target, .001)

    def test_resource_limits_reject_before_copying_oversized_array(self):
        points = np.broadcast_to(np.zeros(3), (2_000_001, 3))
        with self.assertRaisesRegex(ValueError, 'point limit'):
            estimate_rigid_transform_robust(points, points)
        source, target, _, _ = fixture(n=10)
        with self.assertRaisesRegex(ValueError, 'max_iterations'):
            estimate_rigid_transform_robust(source, target, max_iterations=10_001)

    def test_invalid_input_and_parameters(self):
        source, target, _, _ = fixture(n=10)
        for value in (None, [1, 2, 3], np.ones((3, 2)), source.astype(complex),
                      np.full((3, 3), np.inf), np.full((3, 3), np.nan),
                      np.ones((3, 3), bool), [['1', '2', '3']]):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    estimate_rigid_transform_robust(value, target)
        with self.assertRaises(ValueError):
            estimate_rigid_transform_robust(source, target[:-1])
        invalid = {'threshold': (0, -1, np.inf, np.nan, True, '1'),
                   'confidence': (0, 1, -.1, np.inf, True),
                   'min_inlier_ratio': (0, 1.1, np.nan, True),
                   'max_iterations': (0, -1, 1.5, True),
                   'min_inliers': (2, 3.5, True), 'seed': (-1, 1.5, True, None)}
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        estimate_rigid_transform_robust(source, target, **{name: value})


if __name__ == '__main__':
    unittest.main()
