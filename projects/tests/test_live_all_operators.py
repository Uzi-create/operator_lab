import unittest
import tempfile
import re
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
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

    def test_scene_motion_first_frame_has_no_arrow_or_all_operator_preparation(self):
        frame = np.zeros((192, 256, 3), np.uint8)
        with patch.object(live.check, 'cases', side_effect=AssertionError('unrelated preparation ran')):
            result, info, _, _ = live.process(frame, None, 'estimate_translation')
        np.testing.assert_array_equal(result, frame)
        self.assertIn('no_previous_frame', ' '.join(info))

    def test_scene_motion_rejection_never_draws_a_green_arrow(self):
        frame = np.zeros((192, 256, 3), np.uint8)
        value = {'success': False, 'reason': 'inconsistent_motion',
                 'shift_xy': np.array([25., -8.]), 'response': .95}
        result, info = live.render_result(frame, 'estimate_translation', value, {})
        np.testing.assert_array_equal(result, frame)
        self.assertIn('inconsistent_motion', ' '.join(info))

    def test_scene_motion_live_arrow_matches_synthetic_image_shift(self):
        previous = np.random.default_rng(12).integers(0, 256, (192, 256), dtype=np.uint8)
        current = cv2.warpAffine(previous, np.float32([[1, 0, 12], [0, 1, -7]]),
                                 (256, 192), borderMode=cv2.BORDER_CONSTANT)
        frame = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
        result, info, _, _ = live.process(frame, previous.astype(np.float32)/255,
                                          'estimate_translation')
        details = ' '.join(info)
        self.assertIn('scene/image shift', details)
        self.assertIn('DETECTED', details)
        match = re.search(r'dx=([-\d.]+), dy=([-\d.]+)', details)
        self.assertIsNotNone(match)
        self.assertAlmostEqual(float(match.group(1)), 12., delta=.5)
        self.assertAlmostEqual(float(match.group(2)), -7., delta=.5)
        self.assertTrue(np.any(np.all(result == (0, 255, 0), axis=2)))

    def test_headless_camera_starts_without_a_previous_frame(self):
        previous = np.random.default_rng(12).integers(0, 256, (192, 256), dtype=np.uint8)
        current = cv2.warpAffine(previous, np.float32([[1, 0, 12], [0, 1, -7]]),
                                 (256, 192), borderMode=cv2.BORDER_CONSTANT)
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
                                for image in (previous, current)]
        output = io.StringIO()
        with patch.object(live.cv2, 'VideoCapture', return_value=cap), \
             patch.object(live.sys, 'argv', ['live', '--headless', '--frames', '2',
                                            '--operator', 'estimate_translation']), \
             redirect_stdout(output):
            self.assertEqual(live.main(), 0)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertIn('no_previous_frame', ' '.join(records[0]['details']))
        self.assertIn('DETECTED', ' '.join(records[1]['details']))
        cap.release.assert_called_once()

    def test_camera_released_on_read_failure(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        with patch.object(live.cv2, 'VideoCapture', return_value=cap), \
             patch.object(live.sys, 'argv', ['live', '--headless', '--frames', '1']):
            with self.assertRaises(RuntimeError): live.main()
        cap.release.assert_called_once()
