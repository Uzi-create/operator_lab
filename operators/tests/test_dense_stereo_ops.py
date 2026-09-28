import unittest

import cv2
import numpy as np

from operators.dense_stereo_ops import dense_stereo_depth
from operators.robot_ops import Intrinsics


def shifted_pair(seed=414):
    rng = np.random.default_rng(seed)
    left = cv2.GaussianBlur(rng.integers(0, 256, (240, 320), dtype=np.uint8), (3, 3), 0)
    right = np.zeros_like(left)
    right[:, :-12] = left[:, 12:]
    return left, right, Intrinsics(320, 240, 500., 500., 160., 120.)


class DenseStereoTests(unittest.TestCase):
    def test_known_pixel_shift_and_metric_depth(self):
        left, right, camera = shifted_pair()
        result = dense_stereo_depth(left, right, camera, .12)
        central = np.s_[:, 90:240]
        self.assertTrue(result['valid'][central].all())
        np.testing.assert_allclose(result['disparity_px'][central], 12., atol=.125)
        np.testing.assert_allclose(result['depth'][central], 5., atol=.06)
        self.assertTrue(np.isnan(result['depth'][~result['valid']]).all())
        self.assertEqual(result['valid_count'], int(result['valid'].sum()))
        self.assertEqual(result['raw_disparity_px'].shape, left.shape)

    def test_occluded_band_and_textureless_image(self):
        left, right, camera = shifted_pair()
        rng = np.random.default_rng(11)
        occluded = right.copy()
        occluded[:, 120:150] = rng.integers(0, 256, (240, 30), dtype=np.uint8)
        result = dense_stereo_depth(left, occluded, camera, .12)
        self.assertLess(result['valid'][:, 132:162].mean(), .2)
        self.assertGreater(result['valid'][:, 180:230].mean(), .95)
        blank = np.full(left.shape, 127, dtype=np.uint8)
        no_texture = dense_stereo_depth(blank, blank, camera, .12)
        self.assertEqual(no_texture['valid_count'], 0)

    def test_color_float_and_depth_gates(self):
        left, right, camera = shifted_pair()
        color = dense_stereo_depth(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
                                   cv2.cvtColor(right, cv2.COLOR_GRAY2BGR), camera, .12)
        floats = dense_stereo_depth(left.astype(np.float32)/255,
                                    right.astype(np.float32)/255, camera, .12)
        np.testing.assert_array_equal(color['valid'], floats['valid'])
        shallow = dense_stereo_depth(left, right, camera, .12, max_depth=4.)
        self.assertEqual(shallow['valid_count'], 0)
        no_reverse = dense_stereo_depth(left, right, camera, .12, left_right_check=False)
        self.assertGreaterEqual(no_reverse['valid_count'], color['valid_count'])

    def test_invalid_geometry_and_parameters(self):
        left, right, camera = shifted_pair()
        for arguments in ({'baseline': 0.}, {'num_disparities': 63}, {'block_size': 4},
                          {'left_right_check': 1}, {'max_depth': .01}, {'min_disparity': 300}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                baseline = arguments.pop('baseline', .12)
                dense_stereo_depth(left, right, camera, baseline, **arguments)
        with self.assertRaises(ValueError):
            dense_stereo_depth(left, right[:-1], camera, .12)
        with self.assertRaises(ValueError):
            dense_stereo_depth(left.astype(np.float32)*2, right, camera, .12)


if __name__ == '__main__':
    unittest.main()
