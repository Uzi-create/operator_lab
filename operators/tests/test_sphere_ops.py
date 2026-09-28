import unittest

import numpy as np

from operators.sphere_ops import fit_sphere


def make_sphere(seed=827, count=1000, outliers=200, noise=.003):
    rng = np.random.default_rng(seed)
    center = np.array([.2, -.3, .5])
    radius = .7
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    points = center+directions*(radius+rng.normal(0, noise, count))[:, None]
    points = np.vstack((points, rng.uniform(-2., 2., (outliers, 3))))
    return points, center, radius


class SphereTests(unittest.TestCase):
    def check_final(self, result, points, threshold):
        self.assertTrue(result['success'], result['reason'])
        residuals = np.linalg.norm(points-result['center'], axis=1)-result['radius']
        np.testing.assert_allclose(result['residuals'], residuals, rtol=0, atol=0)
        np.testing.assert_array_equal(result['inliers'], abs(residuals) <= threshold)
        self.assertEqual(result['inlier_count'], int(result['inliers'].sum()))
        self.assertAlmostEqual(result['rms'], np.sqrt(np.mean(residuals[result['inliers']]**2)), delta=1e-15)

    def test_noisy_sphere_outliers_and_final_residuals(self):
        points, truth, radius = make_sphere()
        result = fit_sphere(points, threshold=.015, min_inliers=900,
                            min_inlier_ratio=.7, seed=41)
        self.check_final(result, points, .015)
        self.assertGreaterEqual(result['inliers'][:1000].sum(), 998)
        self.assertLessEqual(result['inliers'][1000:].sum(), 1)
        self.assertLess(np.linalg.norm(result['center']-truth), .001)
        self.assertLess(abs(result['radius']-radius), .001)
        self.assertLess(result['iterations'], 1024)
        self.assertLess(result['rms'], .004)

    def test_exact_and_partial_hemisphere(self):
        points, truth, radius = make_sphere(count=500, outliers=0, noise=0.)
        exact = fit_sphere(points, threshold=1e-8, min_inlier_ratio=.9, seed=31)
        self.check_final(exact, points, 1e-8)
        np.testing.assert_allclose(exact['center'], truth, atol=1e-12)
        self.assertAlmostEqual(exact['radius'], radius, delta=1e-12)
        half = points[points[:, 2] > truth[2]]
        partial = fit_sphere(half, threshold=1e-8, min_inliers=100, seed=11)
        self.check_final(partial, half, 1e-8)
        np.testing.assert_allclose(partial['center'], truth, atol=1e-10)

    def test_plane_collinear_and_wrong_radius_rejected(self):
        rng = np.random.default_rng(91)
        plane = np.column_stack((rng.uniform(-1, 1, (500, 2)), np.zeros(500)))
        result = fit_sphere(plane, threshold=.01, max_iterations=80)
        self.assertFalse(result['success'])
        self.assertIsNone(result['center'])
        self.assertFalse(result['inliers'].any())
        line = np.column_stack((np.arange(100.), np.zeros(100), np.zeros(100)))
        self.assertFalse(fit_sphere(line, threshold=.01, max_iterations=20)['success'])
        points, _, _ = make_sphere()
        limited = fit_sphere(points, threshold=.015, max_radius=.5, max_iterations=60)
        self.assertFalse(limited['success'])

    def test_invalid_inputs(self):
        for bad in ([[1, 2]], [[np.nan, 0, 0]], [[1j, 2, 3]], np.ones((4, 4))):
            with self.assertRaises(ValueError):
                fit_sphere(bad)
        points = np.zeros((4, 3))
        for options in ({'threshold': 0}, {'min_radius': -1}, {'max_radius': 0},
                        {'min_inlier_ratio': 0}, {'min_spread_ratio': 0},
                        {'confidence': 1}, {'max_iterations': 0}, {'seed': -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit_sphere(points, **options)
        self.assertEqual(fit_sphere(np.zeros((3, 3)))['reason'], 'insufficient_points')


if __name__ == '__main__':
    unittest.main()
