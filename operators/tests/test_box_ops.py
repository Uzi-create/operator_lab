import unittest

import cv2
import numpy as np

from operators.box_ops import gravity_aligned_box


def box_points(angle_deg=23., center=(.3, -.2, .5), size=(.8, .35, .4)):
    rotation = cv2.Rodrigues(np.array([0., 0., np.radians(angle_deg)]))[0]
    vertices = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    rng = np.random.default_rng(120)
    inside = rng.uniform(-.5, .5, (2000, 3))
    points = np.vstack((vertices, inside))*np.asarray(size)
    return points@rotation.T+np.asarray(center), rotation


class BoxTests(unittest.TestCase):
    def test_rotated_box_known_size_and_enclosure(self):
        points, _ = box_points()
        result = gravity_aligned_box(points)
        self.assertTrue(result['success'], result['reason'])
        np.testing.assert_allclose(result['size'], [.8, .35, .4], atol=1e-6)
        np.testing.assert_allclose(result['center'], [.3, -.2, .5], atol=1e-6)
        self.assertAlmostEqual(result['footprint_area'], .28, delta=1e-6)
        self.assertAlmostEqual(result['volume'], .112, delta=1e-6)
        self.assertFalse(result['orientation_ambiguous'])
        np.testing.assert_allclose(result['axes'].T@result['axes'], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(result['axes']), 1., delta=1e-12)
        local = (points-result['center'])@result['axes']
        self.assertTrue((abs(local) <= result['size']/2+1e-12).all())
        self.assertEqual(result['corners'].shape, (8, 3))

    def test_arbitrary_up_and_visible_surface_height(self):
        points, _ = box_points()
        rotation = cv2.Rodrigues(np.array([.15, -.22, .11]))[0]
        shifted = points@rotation.T+[1.1, -.6, .3]
        result = gravity_aligned_box(shifted, up=rotation@[0, 0, 1])
        self.assertTrue(result['success'])
        np.testing.assert_allclose(result['size'], [.8, .35, .4], atol=1e-6)
        np.testing.assert_allclose(result['center'], np.asarray([.3, -.2, .5])@rotation.T+[1.1, -.6, .3], atol=1e-6)
        # A single visible face has zero height; do not hallucinate object volume.
        top = points[points[:, 2] >= .7-1e-12]
        surface = gravity_aligned_box(top)
        self.assertTrue(surface['success'])
        self.assertAlmostEqual(surface['size'][2], 0., delta=1e-12)

    def test_outlier_and_square_orientation_are_explicit(self):
        points, _ = box_points()
        original = gravity_aligned_box(points)
        expanded = gravity_aligned_box(np.vstack((points, [2., 2., 2.])))
        self.assertGreater(expanded['volume'], original['volume'])
        square, _ = box_points(size=(.5, .5, .25))
        result = gravity_aligned_box(square)
        self.assertTrue(result['orientation_ambiguous'])

    def test_degenerate_and_invalid(self):
        self.assertEqual(gravity_aligned_box(np.zeros((2, 3)))['reason'], 'insufficient_points')
        line = np.column_stack((np.arange(10.), np.zeros(10), np.zeros(10)))
        self.assertEqual(gravity_aligned_box(line)['reason'], 'degenerate_footprint')
        for bad in ([[0, 0]], [[np.nan, 0, 0]], np.ones((3, 4)), [[1j, 0, 0]]):
            with self.assertRaises(ValueError):
                gravity_aligned_box(bad)
        for options in ({'up': [0, 0, 0]}, {'up': [1, 2]},
                        {'min_footprint': 0}, {'min_footprint': np.inf}):
            with self.assertRaises(ValueError):
                gravity_aligned_box(np.zeros((3, 3)), **options)


if __name__ == '__main__':
    unittest.main()
