import math
import unittest

import numpy as np

from operators.cylinder_ops import fit_cylinder_known_axis


def cylinder_points(n=500, radius=.23, length=.8, axis=(.3, -.4, .7),
                    center=(.4, -.8, 1.2), coverage=360., noise=0., seed=14):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    basis = np.eye(3)[np.argmin(np.abs(axis))]
    u = basis - axis * (basis @ axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    rng = np.random.default_rng(seed)
    theta = np.linspace(-coverage / 2, coverage / 2, n, endpoint=False) * math.pi / 180
    axial = rng.uniform(-length / 2, length / 2, n)
    axial[0], axial[-1] = -length / 2, length / 2
    points = (np.asarray(center) + (radius * np.cos(theta))[:, None] * u
              + (radius * np.sin(theta))[:, None] * v + axial[:, None] * axis)
    points += rng.normal(0, noise, points.shape)
    return points, axis


class CylinderTests(unittest.TestCase):
    def test_arbitrary_axis_center_and_radius(self):
        points, axis = cylinder_points()
        result = fit_cylinder_known_axis(points, axis * 8)
        self.assertTrue(result['success'], result['reason'])
        np.testing.assert_allclose(result['axis_direction'], axis, atol=1e-15)
        np.testing.assert_allclose(result['axis_point'], [.4, -.8, 1.2], atol=1e-12)
        self.assertAlmostEqual(result['radius'], .23, delta=1e-12)
        self.assertAlmostEqual(result['length_observed'], .8, delta=1e-12)
        np.testing.assert_allclose(result['segment_xyz'][1] - result['segment_xyz'][0],
                                   .8 * axis, atol=1e-12)
        self.assertGreater(result['coverage_deg'], 359.)
        self.assertIsNone(result['candidate'])

    def test_noisy_surface_with_thirty_percent_outliers(self):
        surface, axis = cylinder_points(n=700, noise=.0006)
        rng = np.random.default_rng(502)
        outliers = rng.uniform(-1, 1, (300, 3)) + [.4, -.8, 1.2]
        points = np.vstack((surface, outliers))
        result = fit_cylinder_known_axis(points, axis, threshold=.0025, seed=19)
        self.assertTrue(result['success'], result['reason'])
        self.assertGreaterEqual(result['inlier_count'], 690)
        self.assertLess(result['inlier_count'], 715)
        self.assertAlmostEqual(result['radius'], .23, delta=.0002)
        delta = result['axis_point'] - [.4, -.8, 1.2]
        self.assertLess(np.linalg.norm(delta - (delta @ axis) * axis), .0002)
        self.assertLess(result['rms'], .0008)

    def test_final_residuals_mask_and_rms_match_independent_geometry(self):
        points, axis = cylinder_points(noise=.0008)
        points = np.vstack((points, [[2., 4., -3.]]))
        result = fit_cylinder_known_axis(points, axis, threshold=.0015)
        self.assertTrue(result['success'], result['reason'])
        difference = points - result['axis_point']
        distances = np.linalg.norm(np.cross(difference, result['axis_direction']), axis=1)
        residuals = abs(distances - result['radius'])
        np.testing.assert_allclose(result['residuals'], residuals, atol=1e-14)
        np.testing.assert_array_equal(result['inliers'], residuals <= .0015)
        self.assertAlmostEqual(result['rms'], np.sqrt(np.mean(residuals[result['inliers']] ** 2)), delta=1e-14)
        self.assertEqual(result['inlier_count'], int(result['inliers'].sum()))
        self.assertAlmostEqual(result['inlier_ratio'], result['inlier_count'] / len(points))

    def test_large_translation_and_axis_reversal(self):
        points, axis = cylinder_points(center=(1e8, -2e8, 3e8), noise=.0001)
        positive = fit_cylinder_known_axis(points, axis)
        negative = fit_cylinder_known_axis(points, -axis)
        self.assertTrue(positive['success'], positive['reason'])
        self.assertTrue(negative['success'], negative['reason'])
        np.testing.assert_allclose(positive['axis_point'], negative['axis_point'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(positive['segment_xyz'], negative['segment_xyz'][::-1], atol=1e-8, rtol=0)
        np.testing.assert_allclose(positive['axis_direction'], -negative['axis_direction'], atol=1e-15)
        self.assertAlmostEqual(positive['radius'], .23, delta=3e-5)
        self.assertAlmostEqual(positive['radius'], negative['radius'], delta=1e-12)
        delta = positive['axis_point'] - [1e8, -2e8, 3e8]
        self.assertLess(np.linalg.norm(delta - (delta @ axis) * axis), 3e-5)
        np.testing.assert_array_equal(positive['inliers'], negative['inliers'])

    def test_partial_coverage_acceptance_and_short_arc_rejection(self):
        points, axis = cylinder_points(coverage=180., noise=.0002)
        result = fit_cylinder_known_axis(points, axis)
        self.assertTrue(result['success'], result['reason'])
        self.assertAlmostEqual(result['radius'], .23, delta=.0002)
        self.assertGreater(result['coverage_deg'], 175.)
        self.assertLess(result['coverage_deg'], 185.)
        short, axis = cylinder_points(coverage=30., noise=.00005)
        result = fit_cylinder_known_axis(short, axis)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'insufficient_angular_coverage')
        self.assertIsNone(result['radius'])
        self.assertIsNotNone(result['candidate'])
        np.testing.assert_array_equal(result['inliers'], result['residuals'] <= .005)

    def test_ring_and_degenerate_points_are_rejected(self):
        ring, axis = cylinder_points(length=0.)
        result = fit_cylinder_known_axis(ring, axis)
        self.assertEqual(result['reason'], 'insufficient_axial_span')
        self.assertIsNone(result['axis_point'])
        self.assertLess(result['candidate']['length_observed'], 1e-12)
        for points in (np.zeros((50, 3)), np.column_stack((np.zeros((50, 2)), np.arange(50)))):
            result = fit_cylinder_known_axis(points, [0, 0, 1])
            self.assertFalse(result['success'])
            self.assertEqual(result['reason'], 'degenerate_cross_section')
        result = fit_cylinder_known_axis(np.zeros((0, 3)), [0, 0, 1])
        self.assertEqual(result['reason'], 'insufficient_points')
        self.assertEqual(result['inliers'].shape, (0,))
        result = fit_cylinder_known_axis(np.zeros((10, 3)), [0, 0, 1])
        self.assertEqual(result['reason'], 'insufficient_points')

    def test_radius_bounds_and_inlier_support_are_enforced(self):
        points, axis = cylinder_points()
        for options in ({'min_radius': .3}, {'max_radius': .2}):
            result = fit_cylinder_known_axis(points, axis, **options)
            self.assertEqual(result['reason'], 'radius_out_of_bounds')
            self.assertIsNone(result['radius'])
            self.assertAlmostEqual(result['candidate']['radius'], .23, delta=1e-12)
        rng = np.random.default_rng(77)
        mixed = np.vstack((points, rng.uniform(-4, 4, (500, 3))))
        result = fit_cylinder_known_axis(mixed, axis, min_inlier_ratio=.8)
        self.assertEqual(result['reason'], 'insufficient_support')
        self.assertLess(result['inlier_ratio'], .8)

    def test_axial_outlier_is_not_falsely_claimed_to_be_rejected(self):
        points, axis = cylinder_points()
        extended = np.vstack((points, points[10] + 100 * axis))
        result = fit_cylinder_known_axis(extended, axis)
        self.assertTrue(result['success'], result['reason'])
        self.assertTrue(result['inliers'][-1])
        self.assertGreater(result['length_observed'], 99.)

    def test_radius_constraints_search_past_larger_excluded_surface(self):
        valid, axis = cylinder_points(n=350, radius=.23, noise=.0001)
        excluded, _ = cylinder_points(n=650, radius=.51, noise=.0001, seed=96)
        points = np.vstack((excluded, valid))
        result = fit_cylinder_known_axis(points, axis, threshold=.001,
                                         min_radius=.2, max_radius=.3,
                                         min_inlier_ratio=.3, seed=2)
        self.assertTrue(result['success'], result['reason'])
        self.assertAlmostEqual(result['radius'], .23, delta=.0001)
        self.assertEqual(result['inlier_count'], 350)
        self.assertFalse(result['inliers'][:650].any())
        self.assertTrue(result['inliers'][650:].all())

    def test_minimum_count_and_coverage_are_independent_requirements(self):
        points, axis = cylinder_points(n=40, coverage=80.)
        accepted = fit_cylinder_known_axis(points, axis, min_coverage_deg=0.)
        self.assertTrue(accepted['success'], accepted['reason'])
        rejected = fit_cylinder_known_axis(points, axis, min_inliers=41)
        self.assertEqual(rejected['reason'], 'insufficient_points')
        ring, axis = cylinder_points(n=40, length=0.)
        ring_fit = fit_cylinder_known_axis(ring, axis, min_axial_span=0.)
        self.assertTrue(ring_fit['success'], ring_fit['reason'])

    def test_deterministic_no_mutation_and_scaled_directions(self):
        points, axis = cylinder_points(noise=.0003)
        original, original_axis = points.copy(), axis.copy()
        points.setflags(write=False)
        axis.setflags(write=False)
        first = fit_cylinder_known_axis(points, axis, seed=24)
        second = fit_cylinder_known_axis(points, axis, seed=24)
        np.testing.assert_array_equal(points, original)
        np.testing.assert_array_equal(axis, original_axis)
        np.testing.assert_array_equal(first['axis_point'], second['axis_point'])
        np.testing.assert_array_equal(first['inliers'], second['inliers'])
        for factor in (1e-200, 1e200):
            result = fit_cylinder_known_axis(points, axis * factor)
            self.assertTrue(result['success'])
            np.testing.assert_allclose(result['axis_direction'], axis, atol=1e-15)

    def test_invalid_input_and_parameters(self):
        points, axis = cylinder_points()
        for bad in ([[0, 0]], np.ones((3, 4)), [[1j, 0, 0]], [[True, False, True]],
                    [[np.nan, 0, 0]], [[np.inf, 0, 0]], [['1', '2', '3']]):
            with self.subTest(points=bad), self.assertRaises(ValueError):
                fit_cylinder_known_axis(bad, axis)
        for bad_axis in ([0, 0, 0], [1, 0], [np.inf, 0, 1], [0, np.nan, 1],
                         [1j, 0, 1], ['1', '0', '0']):
            with self.subTest(axis=bad_axis), self.assertRaises(ValueError):
                fit_cylinder_known_axis(points, bad_axis)
        bad_options = ({'threshold': 0}, {'threshold': np.nan}, {'threshold': True},
                       {'threshold': '1'}, {'threshold': [1]}, {'min_radius': -1},
                       {'max_radius': .1, 'min_radius': .1}, {'max_radius': -math.inf},
                       {'min_inlier_ratio': 0}, {'min_inlier_ratio': 1.1},
                       {'min_coverage_deg': -1}, {'min_coverage_deg': 361},
                       {'min_axial_span': -1}, {'min_axial_span': np.inf},
                       {'min_inliers': 2}, {'min_inliers': 3.1}, {'min_inliers': True},
                       {'max_iterations': 0}, {'max_iterations': 10001},
                       {'seed': -1}, {'seed': 1.2}, {'seed': False})
        for options in bad_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit_cylinder_known_axis(points, axis, **options)


if __name__ == '__main__':
    unittest.main()
