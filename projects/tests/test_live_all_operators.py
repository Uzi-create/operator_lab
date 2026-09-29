import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from projects.camera import live_all_operators as live


class AllOperatorViewerTests(unittest.TestCase):
    def test_native_view_copies_pixels_instead_of_downscaling(self):
        image = np.random.default_rng(18).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        view = live.image_view(image, (480, 400), zoom=1.)
        np.testing.assert_array_equal(view, image[340:740, 720:1200])
        canvas = live.panel(image, image, 2, [], 0., 'test', False)
        np.testing.assert_array_equal(canvas[90:490, 280:760], view)

    def test_zoom_preserves_pixel_blocks_and_pan_reaches_edges(self):
        image = np.arange(4*4*3, dtype=np.uint8).reshape(4, 4, 3)
        view = live.image_view(image, (4, 4), zoom=2.)
        np.testing.assert_array_equal(view, image[1:3, 1:3].repeat(2, axis=0).repeat(2, axis=1))
        np.testing.assert_array_equal(live.image_view(image, (2, 2), center=(0., 0.)), image[:2, :2])
        np.testing.assert_array_equal(live.image_view(image, (2, 2), center=(1., 1.)), image[-2:, -2:])

    def test_fit_view_keeps_aspect_ratio_with_letterboxing(self):
        image = np.full((100, 200, 3), 255, np.uint8)
        view = live.image_view(image, (100, 100), zoom=0.)
        np.testing.assert_array_equal(view[25:75], 255)
        np.testing.assert_array_equal(view[:25], 16)
        np.testing.assert_array_equal(view[75:], 16)

    def test_saved_original_preserves_full_resolution_and_pixels(self):
        frame = np.random.default_rng(8).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = live.save_original(frame, Path(directory))
            decoded = live.cv2.imread(str(path))
        np.testing.assert_array_equal(decoded, frame)

    def test_all_cases_render_without_mutating_input(self):
        frame = np.random.default_rng(72).integers(0, 256, (192, 256, 3), dtype=np.uint8)
        original = frame.copy()
        gray = live.cv2.cvtColor(frame, live.cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        self.assertEqual(tuple(n for n, _, _ in live.check.cases(gray, gray)), live.NAMES)
        self.assertEqual(len(live.NAMES), 38)
        for index, name in enumerate(live.NAMES):
            with self.subTest(operator=name):
                result, info, elapsed, backend = live.process(frame, gray, name)
                self.assertEqual(result.shape, frame.shape)
                self.assertEqual(result.dtype, np.uint8)
                canvas = live.panel(frame, result, index, info, elapsed, backend, False)
                self.assertEqual(canvas.shape, (760, 1260, 3))
                np.testing.assert_array_equal(frame, original)

    def test_rejected_circle_fit_is_not_drawn_as_accepted(self):
        frame = np.zeros((96, 128, 3), np.uint8)
        value = {'success': False, 'reason': 'insufficient_coverage',
                 'center': np.array([60, 40]), 'radius': 12., 'edge_points': np.empty((0, 2))}
        result, info = live.render_result(frame, 'measure_circle', value, {})
        self.assertTrue(any('NO DETECTION' in text for text in info))
        self.assertFalse(np.any(np.all(result == (0, 255, 0), axis=2)))

    def test_camera_released_on_read_failure(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        with patch.object(live.cv2, 'VideoCapture', return_value=cap), \
             patch.object(live.sys, 'argv', ['live', '--headless', '--frames', '1']):
            with self.assertRaises(RuntimeError): live.main()
        cap.release.assert_called_once()
