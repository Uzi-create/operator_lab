import unittest

import cv2
import numpy as np

from operators.measurement_ops import estimate_translation, measure_circle, measure_stripes


class TranslationTests(unittest.TestCase):
    def test_odd_dft_dimensions_have_no_half_pixel_bias(self):
        rng = np.random.default_rng(42)
        for shape in [(75, 75), (81, 125), (135, 225), (75, 128), (128, 75)]:
            reference = rng.random(shape, dtype=np.float32)
            for dx, dy in [(0, 0), (5, -3), (-7, 4)]:
                moving = np.roll(reference, (dy, dx), axis=(0, 1))
                result = estimate_translation(reference, moving, window=False)
                with self.subTest(shape=shape, shift=(dx, dy)):
                    self.assertTrue(result['success'])
                    np.testing.assert_allclose(result['shift_xy'], [dx, dy], atol=1e-4)

    def test_odd_image_with_even_dft_padding_is_not_overcorrected(self):
        rng = np.random.default_rng(84)
        for shape in [(77, 127), (79, 129), (71, 121)]:
            reference = rng.random(shape, dtype=np.float32)
            identity = estimate_translation(reference, reference)
            np.testing.assert_allclose(identity['shift_xy'], [0, 0], atol=1e-4)

    def test_integer_shift_sign_and_alignment(self):
        rng = np.random.default_rng(42)
        reference = rng.random((128, 160), dtype=np.float32)
        for dx, dy in [(11, -7), (-18, 9), (0, 0)]:
            moving = np.roll(reference, (dy, dx), axis=(0, 1))
            out = estimate_translation(reference, moving, window=False)
            self.assertTrue(out['success'])
            np.testing.assert_allclose(out['shift_xy'], [dx, dy], atol=1e-4)
            np.testing.assert_allclose(out['alignment_matrix'][:, 2], [-dx, -dy], atol=1e-4)

    def test_subpixel_window_does_not_modify_views(self):
        rng = np.random.default_rng(8)
        reference = cv2.GaussianBlur(rng.random((240, 320), dtype=np.float32), (0, 0), 1.5)
        moving = cv2.warpAffine(reference, np.array([[1., 0., 3.3], [0., 1., -2.2]]),
                                (320, 240), borderMode=cv2.BORDER_REFLECT101)
        originals = reference.copy(), moving.copy()
        out = estimate_translation(reference.T, moving.T)
        self.assertTrue(out['success'])
        np.testing.assert_allclose(out['shift_xy'], [-2.2, 3.3], atol=.3)
        np.testing.assert_array_equal(reference, originals[0])
        np.testing.assert_array_equal(moving, originals[1])

    def test_no_texture_limit_and_invalid(self):
        gray = np.zeros((20, 20), np.float32)
        self.assertEqual(estimate_translation(gray, gray)['reason'], 'insufficient_texture')
        rng = np.random.default_rng(4)
        gray = rng.random((64, 64), dtype=np.float32)
        out = estimate_translation(gray, np.roll(gray, 10, axis=1), window=False, max_shift=(5, 5))
        self.assertFalse(out['success'])
        self.assertEqual(out['reason'], 'shift_limit')
        self.assertIsNone(out['alignment_matrix'])
        for kwargs in [dict(min_response=-1), dict(max_shift=(1, np.nan)), dict(window=1)]:
            with self.assertRaises(ValueError):
                estimate_translation(gray, gray, **kwargs)
        with self.assertRaises(ValueError):
            estimate_translation(gray, gray[:-1])


class StripeTests(unittest.TestCase):
    @staticmethod
    def scene():
        x = np.arange(200, dtype=np.float32)
        profile = .1 + .4 * (np.tanh((x - 45.3) / 1.5) - np.tanh((x - 75.7) / 1.5))
        profile += .4 * (np.tanh((x - 115.2) / 1.5) - np.tanh((x - 140.4) / 1.5))
        return np.tile(profile, (40, 1))

    def test_multiple_widths_subpixel_and_reverse(self):
        gray = self.scene()
        for start, end in [((5, 20), (190, 20)), ((190, 20), (5, 20))]:
            out = measure_stripes(gray, start, end)
            widths = sorted(item['width_px'] for item in out['stripes'])
            np.testing.assert_allclose(widths, [25.2, 30.4], atol=.2)

    def test_dark_width_filter_and_incomplete(self):
        gray = 1 - self.scene()
        out = measure_stripes(gray, (5, 20), (190, 20), polarity='dark', min_width=28, max_width=35)
        self.assertEqual(len(out['stripes']), 1)
        self.assertAlmostEqual(out['stripes'][0]['width_px'], 30.4, delta=.2)
        # The first edge of this stripe is outside the caliper; no invented pair.
        self.assertFalse(measure_stripes(gray, (60, 20), (100, 20), polarity='dark')['stripes'])
        with self.assertRaises(ValueError):
            measure_stripes(gray, (5, 20), (190, 20), min_width=5, max_width=3)


class CircleCaliperTests(unittest.TestCase):
    @staticmethod
    def scene(center=(120.3, 105.6), radius=48.7):
        y, x = np.indices((220, 250), dtype=np.float32)
        distance = np.hypot(x - center[0], y - center[1])
        return (.5 + .4 * np.tanh((radius - distance) / 1.2)).astype(np.float32)

    def test_circle_subpixel_and_polarity(self):
        gray = self.scene()
        for image, polarity in [(gray, 'dark'), (1 - gray, 'bright')]:
            out = measure_circle(image, (119, 107), 49, polarity=polarity)
            self.assertTrue(out['success'], out['reason'])
            np.testing.assert_allclose(out['center'], (120.3, 105.6), atol=.15)
            self.assertAlmostEqual(out['radius'], 48.7, delta=.15)
            self.assertLess(out['rms'], .15)
            self.assertGreater(out['angular_coverage_deg'], 350)
            self.assertEqual(len(out['inliers']), len(out['edge_points']))

    def test_outlier_patch_and_partial_frame(self):
        gray = self.scene()
        gray[70:110, 153:175] = .5
        out = measure_circle(gray, (120, 106), 49, fit_threshold=.5, min_inliers=70)
        self.assertTrue(out['success'], out['reason'])
        np.testing.assert_allclose(out['center'], (120.3, 105.6), atol=.2)
        clipped = self.scene((20.3, 105.6))
        out = measure_circle(clipped, (20, 106), 49)
        self.assertTrue(out['success'], out['reason'])
        self.assertLess(np.count_nonzero(out['supported_rays']), 128)
        np.testing.assert_allclose(out['center'], (20.3, 105.6), atol=.2)
        out = measure_circle(clipped, (20, 106), 49, min_coverage_deg=330)
        self.assertFalse(out['success'])
        self.assertEqual(out['reason'], 'insufficient_coverage')

    def test_empty_wrong_polarity_and_parameters(self):
        gray = self.scene()
        self.assertFalse(measure_circle(np.zeros_like(gray), (120, 106), 49)['success'])
        self.assertFalse(measure_circle(gray, (120, 106), 49, polarity='bright')['success'])
        for kwargs in [dict(num_rays=3), dict(radial_range=50), dict(threshold=0),
                       dict(min_inliers=129), dict(sigma=np.nan), dict(seed=-1),
                       dict(min_inlier_ratio=0)]:
            with self.assertRaises(ValueError):
                measure_circle(gray, (120, 106), 49, **kwargs)

    def test_random_texture_requires_consensus(self):
        gray = np.random.default_rng(61).random((220, 250), dtype=np.float32)
        out = measure_circle(gray, (120, 106), 49, threshold=.01, min_inlier_ratio=.8)
        self.assertFalse(out['success'])
        self.assertEqual(out['reason'], 'insufficient_inlier_ratio')


if __name__ == '__main__':
    unittest.main()
