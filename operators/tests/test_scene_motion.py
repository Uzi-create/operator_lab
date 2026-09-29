"""Regression cases for the live camera's conservative image-motion estimate.

These synthetic pairs check the reported *image* displacement. They do not
establish physical camera pose, or accuracy on a particular camera feed.
"""
import unittest

import cv2
import numpy as np

from operators.measurement_ops import estimate_scene_motion


def texture(shape=(240, 320), seed=20260929):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.random(shape, dtype=np.float32), (0, 0), .8)


def translate(image, dx, dy=0, *, border=cv2.BORDER_CONSTANT):
    height, width = image.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (width, height), borderMode=border)


class SceneMotionTests(unittest.TestCase):
    def test_signed_image_shift_and_inverse_alignment(self):
        reference = texture()
        for dx, dy in [(12, -7), (-12, 0), (0, 0), (-9, 6)]:
            with self.subTest(expected=(dx, dy)):
                result = estimate_scene_motion(reference, translate(reference, dx, dy))
                self.assertTrue(result['success'], result)
                np.testing.assert_allclose(result['shift_xy'], (dx, dy), atol=.5)
                np.testing.assert_allclose(result['alignment_matrix'][:, 2],
                                           -np.asarray(result['shift_xy']), atol=1e-8)

    def test_subpixel_nonwrapping_translation_and_input_views(self):
        reference = texture((240, 320), seed=9)
        moving = translate(reference, 3.25, -2.5, border=cv2.BORDER_REFLECT101)
        original = reference.copy(), moving.copy()
        result = estimate_scene_motion(reference.T, moving.T)
        self.assertTrue(result['success'], result)
        np.testing.assert_allclose(result['shift_xy'], (-2.5, 3.25), atol=.8)
        np.testing.assert_array_equal(reference, original[0])
        np.testing.assert_array_equal(moving, original[1])

    def test_cropped_shift_beyond_half_width_is_not_wrapped(self):
        reference = texture(seed=13)
        for dx in (190, -190):
            with self.subTest(dx=dx):
                result = estimate_scene_motion(reference, translate(reference, dx))
                self.assertTrue(result['success'], result)
                np.testing.assert_allclose(result['shift_xy'], (dx, 0), atol=1.)

    def test_downsampled_analysis_reports_original_image_pixels(self):
        reference = texture((720, 1280), seed=14)
        result = estimate_scene_motion(reference, translate(reference, 12, -7))
        self.assertTrue(result['success'], result)
        self.assertLessEqual(max(result['analysis_shape_hw']), 480)
        np.testing.assert_allclose(result['shift_xy'], (12, -7), atol=.75)

    def test_circular_alias_is_rejected(self):
        reference = texture((128, 160), seed=17)
        # These pixels are consistent with both +90 and -70 px. A high FFT
        # response cannot identify the true direction without a motion prior.
        result = estimate_scene_motion(reference, np.roll(reference, 90, axis=1))
        self.assertFalse(result['success'], result)
        self.assertEqual(result['reason'], 'ambiguous_wrap')
        self.assertIsNone(result['alignment_matrix'])

    def test_periodic_texture_is_rejected(self):
        x = np.arange(320, dtype=np.float32)
        reference = np.broadcast_to(.5 + .4 * np.sin(2 * np.pi * x / 20),
                                    (240, 320)).copy()
        result = estimate_scene_motion(reference, np.roll(reference, 60, axis=1))
        self.assertFalse(result['success'], result)
        self.assertEqual(result['reason'], 'ambiguous_texture')
        self.assertIsNone(result['alignment_matrix'])

    def test_conflicting_foreground_and_background_rejected(self):
        background = .45 + .03 * texture(seed=23)
        foreground = texture((150, 240), seed=24)
        first = background.copy()
        first[45:195, 40:280] = foreground
        second = translate(background, -8, border=cv2.BORDER_REFLECT101)
        second[45:195, 65:305] = foreground
        result = estimate_scene_motion(first, second)
        self.assertFalse(result['success'], result)
        self.assertEqual(result['reason'], 'inconsistent_motion')
        self.assertIsNone(result['alignment_matrix'])

    def test_dominant_moving_foreground_does_not_override_thin_background(self):
        # A large bright object moves +15 while the outer scene moves -8.
        # Coarse global/local tiles can all vote for the object, so check the
        # actual image-motion API rejects this one-direction ambiguity.
        shape = (240, 320)
        for coverage in (.65, .8):
            pw = int(shape[1] * np.sqrt(coverage))
            ph = int(shape[0] * np.sqrt(coverage))
            x, y = (shape[1] - pw) // 2, (shape[0] - ph) // 2
            foreground = texture((ph, pw), seed=3456 + int(coverage * 100))
            for contrast in (.03, .3):
                with self.subTest(coverage=coverage, contrast=contrast):
                    background = .45 + contrast * (texture(shape, seed=4567) - .5)
                    first = background.copy()
                    first[y:y + ph, x:x + pw] = foreground
                    second = translate(background, -8, border=cv2.BORDER_REFLECT101)
                    fx = min(shape[1] - pw, x + 15)
                    second[y:y + ph, fx:fx + pw] = foreground
                    result = estimate_scene_motion(first, second)
                    self.assertFalse(result['success'], result)
                    self.assertIsNone(result['alignment_matrix'])

    def test_unrelated_tiny_frames_do_not_get_a_direction(self):
        # With too few pixels, a weak random FFT peak can exceed min_response.
        for shape, seed in [((16, 16), 2), ((24, 32), 2), ((32, 40), 3)]:
            with self.subTest(shape=shape, seed=seed):
                first = texture(shape, seed=1000 + seed)
                second = texture(shape, seed=2000 + seed)
                result = estimate_scene_motion(first, second)
                self.assertFalse(result['success'], result)
                self.assertIsNone(result['alignment_matrix'])

    def test_rotation_and_scale_do_not_claim_one_global_translation(self):
        reference = texture((240, 320), seed=1234)
        for angle, scale in ((1., 1.), (0., 1.03)):
            with self.subTest(angle=angle, scale=scale):
                transform = cv2.getRotationMatrix2D((160, 120), angle, scale)
                moving = cv2.warpAffine(reference, transform, (320, 240),
                                        borderMode=cv2.BORDER_REFLECT101)
                result = estimate_scene_motion(reference, moving)
                self.assertFalse(result['success'], result)
                self.assertIsNone(result['alignment_matrix'])

    def test_tiny_frame_rejects_an_inaccurate_shift(self):
        first = texture((16, 16), seed=6789)
        second = translate(first, 2, -1)
        result = estimate_scene_motion(first, second)
        if result['success']:
            np.testing.assert_allclose(result['shift_xy'], (2, -1), atol=1.)
        else:
            self.assertIsNone(result['alignment_matrix'])

    def test_blank_frames_do_not_fabricate_zero_motion(self):
        blank = np.zeros((240, 320), np.float32)
        result = estimate_scene_motion(blank, blank)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'insufficient_texture')
        self.assertIsNone(result['shift_xy'])

    def test_invalid_inputs_and_limits(self):
        reference = texture((128, 160), seed=31)
        with self.assertRaises(ValueError):
            estimate_scene_motion(reference, reference[:-1])
        for kwargs in [dict(max_analysis_width=0), dict(max_analysis_width=1.5),
                       dict(min_response=-.1), dict(max_shift=(np.nan, 4)),
                       dict(max_shift=(-1, 4))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate_scene_motion(reference, reference, **kwargs)
        limited = estimate_scene_motion(reference, translate(reference, 12),
                                        max_shift=(5, 5))
        self.assertFalse(limited['success'], limited)
        self.assertEqual(limited['reason'], 'shift_limit')
        self.assertIsNone(limited['alignment_matrix'])


if __name__ == '__main__':
    unittest.main()
