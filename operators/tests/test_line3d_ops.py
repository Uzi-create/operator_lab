import unittest

import numpy as np

from operators.line3d_ops import fit_line_3d


def make_line(seed=268, count=1000, outliers=200):
    rng = np.random.default_rng(seed)
    point = np.array([.2, -.3, .5])
    direction = np.array([1., .4, -.2])
    direction /= np.linalg.norm(direction)
    t = rng.uniform(-1., 1., count)
    noise = rng.normal(0., .003, (count, 3))
    noise -= (noise@direction)[:, None]*direction
    points = point+t[:, None]*direction+noise
    points = np.vstack((points, rng.uniform(-2., 2., (outliers, 3))))
    return dict(points=points, point=point, direction=direction,
                count=count, outliers=outliers)


class Line3DTests(unittest.TestCase):
    def test_noisy_line_outliers_and_final_residuals(self):
        data = make_line()
        points = data['points']
        result = fit_line_3d(points, threshold=.012, min_inliers=900,
                             min_inlier_ratio=.7, min_span=1.8, seed=71)
        self.assertTrue(result['success'], result['reason'])
        delta = points-result['point']
        residuals = np.linalg.norm(delta-(delta@result['direction'])[:, None]*result['direction'], axis=1)
        np.testing.assert_array_equal(result['inliers'], residuals <= .012)
        np.testing.assert_allclose(result['distances'], residuals, atol=0, rtol=0)
        self.assertAlmostEqual(result['rms'], np.sqrt(np.mean(residuals[result['inliers']]**2)), delta=1e-15)
        self.assertEqual(result['inlier_count'], int(result['inliers'].sum()))
        self.assertGreater(result['inliers'][:1000].mean(), .99)
        self.assertLess(result['inliers'][1000:].sum(), 3)
        angle = np.degrees(np.arccos(np.clip(result['direction']@data['direction'], -1, 1)))
        self.assertLess(angle, .1)
        cross = np.linalg.norm(np.cross(result['point']-data['point'], data['direction']))
        self.assertLess(cross, .001)
        self.assertGreater(result['span'], 1.95)

    def test_exact_line_and_arbitrary_frame(self):
        data = make_line(count=100, outliers=0)
        t = np.linspace(-1, 1, 100)
        points = data['point']+t[:, None]*data['direction']
        result = fit_line_3d(points, threshold=1e-8, min_inliers=100,
                             min_inlier_ratio=1., min_span=1.99, seed=8)
        self.assertTrue(result['success'], result['reason'])
        self.assertAlmostEqual(result['span'], 2., delta=1e-12)
        np.testing.assert_allclose(result['segment'][0], points[0], atol=1e-12)
        np.testing.assert_allclose(result['segment'][1], points[-1], atol=1e-12)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        rotated = points@rotation.T+[1., -.2, .3]
        second = fit_line_3d(rotated, threshold=1e-8, min_inliers=100,
                             min_inlier_ratio=1., seed=8)
        self.assertTrue(second['success'])
        np.testing.assert_allclose(np.sort(second['segment'], axis=0),
                                   np.sort(result['segment']@rotation.T+[1., -.2, .3], axis=0), atol=1e-12)

    def test_non_line_degenerate_and_restrictions(self):
        rng = np.random.default_rng(984)
        blob = rng.normal(size=(300, 3))
        no_line = fit_line_3d(blob, threshold=.005, min_inlier_ratio=.7, max_iterations=60)
        self.assertFalse(no_line['success'])
        self.assertFalse(no_line['inliers'].any())
        coincident = fit_line_3d(np.ones((100, 3)))
        self.assertEqual(coincident['reason'], 'degenerate_point_cloud')
        data = make_line()
        too_short = fit_line_3d(data['points'], threshold=.012, min_span=3., seed=71)
        self.assertFalse(too_short['success'])

    def test_invalid_inputs(self):
        for bad in ([[1, 2]], [[np.nan, 0, 0]], [[1j, 2, 3]], np.ones((4, 4))):
            with self.assertRaises(ValueError):
                fit_line_3d(bad)
        for options in ({'threshold': 0}, {'min_span': -1}, {'min_inlier_ratio': 0},
                        {'confidence': 1}, {'max_iterations': 0}, {'seed': -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit_line_3d(np.ones((10, 3)), **options)
        self.assertEqual(fit_line_3d(np.zeros((1, 3)))['reason'], 'insufficient_points')


if __name__ == '__main__':
    unittest.main()
