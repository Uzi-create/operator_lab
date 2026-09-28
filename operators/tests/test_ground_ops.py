import unittest

import cv2
import numpy as np

from operators.ground_ops import segment_ground_obstacles


def make_scene(seed=211):
    rng = np.random.default_rng(seed)
    ground = np.column_stack((rng.uniform(-2, 2, (3000, 2)),
                              rng.normal(0., .003, 3000)))
    wall = np.column_stack((1.4+rng.normal(0., .003, 6000),
                            rng.uniform(-2., 2., 6000),
                            rng.uniform(.1, 1.5, 6000)))
    objects = rng.uniform([-.7, -.8, .2], [.7, .8, .85], (1000, 3))
    points = np.vstack((ground, wall, objects))
    return points, slice(0, 3000), slice(3000, 9000)


class GroundTests(unittest.TestCase):
    def test_wall_dominates_but_gravity_selects_ground(self):
        points, ground_ids, wall_ids = make_scene()
        result = segment_ground_obstacles(points, expected_ground_height=0.,
                                          height_tolerance=.08, max_tilt_deg=8.,
                                          distance_threshold=.012,
                                          min_ground_points=2700, min_ground_ratio=.25,
                                          seed=31)
        self.assertTrue(result['success'], result['reason'])
        self.assertGreater(result['ground'][ground_ids].mean(), .99)
        self.assertFalse(result['ground'][wall_ids].any())
        self.assertGreater(result['obstacles'][wall_ids].mean(), .99)
        self.assertGreater(result['obstacles'][9000:].mean(), .99)
        self.assertLess(np.linalg.norm(result['normal']-[0., 0., 1.]), .002)
        self.assertLess(abs(result['ground_height_at_origin']), .001)
        self.assertLess(result['rms'], .004)
        self.assertLess(result['iterations'], 1024)
        # Recompute every public diagnostic from the returned refined plane.
        signed = points@result['normal']+result['offset']
        expected = np.abs(signed) <= .012
        np.testing.assert_array_equal(result['ground'], expected)
        np.testing.assert_allclose(result['vertical_heights'], signed/(result['normal']@[0, 0, 1]))
        self.assertEqual(result['ground_count'], int(expected.sum()))
        self.assertAlmostEqual(result['rms'], np.sqrt(np.mean(signed[expected]**2)), delta=1e-14)
        self.assertTrue(np.all(result['ground'].astype(int)+result['obstacles'].astype(int)+
                               result['below'].astype(int)+result['unknown'].astype(int) == 1))

    def test_tilted_ground_arbitrary_up_and_below(self):
        rng = np.random.default_rng(818)
        xy = rng.uniform(-1, 1, (1100, 2))
        ground = np.column_stack((xy, .03*xy[:, 0]-.02*xy[:, 1]+rng.normal(0, .001, 1100)))
        elevated = np.column_stack((xy[:180], np.full(180, .25)))
        below = np.column_stack((xy[180:260], np.full(80, -.2)))
        points = np.vstack((ground, elevated, below))
        rotation = cv2.Rodrigues(np.array([.17, -.11, .09]))[0]
        shifted = points@rotation.T+[.25, -.13, .4]
        up = rotation@[0., 0., 1.]
        result = segment_ground_obstacles(shifted, up=up, max_tilt_deg=5.,
                                          distance_threshold=.006, min_ground_points=900,
                                          min_ground_ratio=.5, seed=39)
        self.assertTrue(result['success'], result['reason'])
        self.assertGreater(result['ground'][:1100].mean(), .99)
        self.assertTrue(result['obstacles'][1100:1280].all())
        self.assertTrue(result['below'][1280:].all())
        self.assertAlmostEqual(result['tilt_deg'], np.degrees(np.arctan(np.hypot(.03, .02))), delta=.05)
        expected_height = .25-.03*xy[:180, 0]+.02*xy[:180, 1]
        np.testing.assert_allclose(result['vertical_heights'][1100:1280], expected_height, atol=.004)

    def test_floor_prior_rejects_large_horizontal_table(self):
        rng = np.random.default_rng(71)
        floor = np.column_stack((rng.uniform(-1, 1, (1100, 2)), rng.normal(0, .002, 1100)))
        table = np.column_stack((rng.uniform(-1, 1, (2300, 2)), .6+rng.normal(0, .002, 2300)))
        points = np.vstack((floor, table))
        result = segment_ground_obstacles(points, expected_ground_height=0.,
                                          height_tolerance=.04, min_ground_points=1000,
                                          min_ground_ratio=.25, seed=17)
        self.assertTrue(result['success'], result['reason'])
        self.assertGreater(result['ground'][:1100].mean(), .99)
        self.assertFalse(result['ground'][1100:].any())
        self.assertAlmostEqual(result['ground_height_at_origin'], 0., delta=.002)
        unconstrained = segment_ground_obstacles(points, min_ground_points=1000,
                                                 min_ground_ratio=.25, seed=17)
        self.assertTrue(unconstrained['success'])
        # Without an external height prior, the larger horizontal table is
        # geometrically indistinguishable from a floor candidate.
        self.assertAlmostEqual(unconstrained['ground_height_at_origin'], .6, delta=.002)

    def test_failure_and_parameter_boundaries(self):
        wall = np.column_stack((np.ones(200), np.linspace(-1, 1, 200), np.linspace(0, 2, 200)))
        result = segment_ground_obstacles(wall, max_iterations=15, min_ground_points=50)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'insufficient_ground_support')
        self.assertIsNone(result['normal'])
        self.assertFalse(result['obstacles'].any())
        self.assertTrue(result['unknown'].all())
        tiny = segment_ground_obstacles(np.zeros((2, 3)), min_ground_points=3)
        self.assertEqual(tiny['reason'], 'insufficient_points')
        for bad in (np.zeros((4, 2)), [[np.nan, 0, 0]], [['a', 1, 0]]):
            with self.assertRaises(ValueError):
                segment_ground_obstacles(bad)
        for options in ({'up': [0, 0, 0]}, {'max_tilt_deg': 70},
                        {'distance_threshold': 0}, {'obstacle_max_height': 0},
                        {'expected_ground_height': np.nan}, {'seed': -1},
                        {'confidence': 1}, {'min_ground_ratio': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                segment_ground_obstacles(np.zeros((10, 3)), **options)


if __name__ == '__main__':
    unittest.main()
