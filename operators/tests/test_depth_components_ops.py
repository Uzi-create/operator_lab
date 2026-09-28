import unittest

import numpy as np

from operators.depth_components_ops import depth_components


class DepthComponentTests(unittest.TestCase):
    def compare(self, image, **kwargs):
        native = depth_components(image, backend='native', **kwargs)
        reference = depth_components(image, backend='numpy', **kwargs)
        np.testing.assert_array_equal(native['labels'], reference['labels'])
        self.assertEqual(native['region_count'], reference['region_count'])
        self.assertEqual(native['labeled_pixel_count'], reference['labeled_pixel_count'])
        for actual, expected in zip(native['regions'], reference['regions']):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                if isinstance(actual[key], float):
                    self.assertAlmostEqual(actual[key], expected[key], delta=2e-13)
                else:
                    self.assertEqual(actual[key], expected[key])
        return native

    def test_known_blocks_holes_and_bounding_boxes(self):
        image = np.array([[1., 1., 0., 2., 2.],
                          [1., np.nan, 0., 2., 2.],
                          [0., 0., 0., 0., 0.],
                          [3., 3., np.inf, 4., 4.]], np.float64)
        before = image.copy()
        found = self.compare(image, max_depth=5., absolute_jump=.01)
        np.testing.assert_array_equal(found['labels'], [[1, 1, 0, 2, 2],
                                                         [1, 0, 0, 2, 2],
                                                         [0, 0, 0, 0, 0],
                                                         [3, 3, 0, 4, 4]])
        self.assertEqual([r['area'] for r in found['regions']], [3, 4, 2, 2])
        self.assertEqual(found['regions'][0]['bbox_xywh'], (0, 0, 2, 2))
        np.testing.assert_array_equal(image, before)

    def test_inclusive_absolute_relative_thresholds_and_scale(self):
        image = np.array([[1., 1.125, 1.265625, 1.423828125]], np.float64)
        same = self.compare(image, absolute_jump=0., relative_jump=.125, max_depth=2.)
        self.assertEqual(same['region_count'], 1)
        self.assertEqual(same['regions'][0]['area'], 4)
        above = image.copy()
        above[0, 1] = np.nextafter(1.125, 2.)
        self.assertGreaterEqual(self.compare(above, absolute_jump=0., relative_jump=.125,
                                              max_depth=2.)['region_count'], 2)
        exact = self.compare(np.array([[1., 1.125]]), absolute_jump=.125, max_depth=2.)
        self.assertEqual(exact['region_count'], 1)
        scaled = self.compare(np.array([[1000, 1125]], np.uint16), depth_scale=.001,
                              absolute_jump=.125, max_depth=2.)
        np.testing.assert_array_equal(scaled['labels'], exact['labels'])
        self.assertAlmostEqual(scaled['regions'][0]['mean_depth'], 1.0625)

    def test_four_vs_eight_diagonals_and_small_filter(self):
        image = np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
        four = self.compare(image, connectivity=4, min_area=2)
        self.assertEqual(four['region_count'], 0)
        self.assertEqual(four['labeled_pixel_count'], 0)
        eight = self.compare(image, connectivity=8, min_area=2)
        self.assertEqual(eight['region_count'], 1)
        self.assertEqual(eight['regions'][0]['area'], 3)
        self.assertEqual(eight['regions'][0]['bbox_xywh'], (0, 0, 3, 3))

    def test_transitive_ramp_bridges_long_range(self):
        ramp = np.array([[1., 1.1, 1.2, 1.3, 1.4, 1.5]])
        found = self.compare(ramp, absolute_jump=.101, max_depth=2.)
        self.assertEqual(found['region_count'], 1)
        self.assertAlmostEqual(found['regions'][0]['max_depth']-found['regions'][0]['min_depth'], .5)
        split = self.compare(ramp, absolute_jump=.099, max_depth=2.)
        self.assertEqual(split['region_count'], 6)

    def test_random_views_and_many_components_capacity_growth(self):
        rng = np.random.default_rng(937)
        values = rng.choice([0., 1., 1.01, 2., 2.02, 5.], size=(45, 37)).astype(np.float64)
        values[4, 5] = np.nan
        values[7, 9] = np.inf
        view = values[:, ::-1]
        for connectivity in (4, 8):
            for area in (1, 4, 13):
                self.compare(view, absolute_jump=.021, relative_jump=.01,
                             connectivity=connectivity, min_area=area)
        checkerboard = np.indices((30, 30)).sum(axis=0) % 2
        output = self.compare(np.where(checkerboard, 1., 2.), absolute_jump=0.)
        self.assertEqual(output['region_count'], 900)

    def test_invalid_arguments(self):
        image = np.ones((3, 4), np.float32)
        for bad in (np.ones((3, 4), bool), np.ones((3, 4), complex), np.empty((0, 4)),
                    np.ones((3, 4, 1)), [[1, 1], [1, 1]]):
            with self.assertRaises(ValueError):
                depth_components(bad)
        for key, value in [('depth_scale', 0.), ('min_depth', -1.), ('max_depth', 0.),
                           ('absolute_jump', -1.), ('relative_jump', -1.),
                           ('relative_jump', np.inf), ('min_area', 0), ('min_area', 13),
                           ('connectivity', 6), ('backend', 'bad')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                depth_components(image, **{key: value})


if __name__ == '__main__':
    unittest.main()
