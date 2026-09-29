from unittest.mock import patch, MagicMock
import unittest

import numpy as np
from projects.camera import live_operators as live


class LiveCameraTests(unittest.TestCase):
    def test_operator_outputs_preserve_shape_and_input(self):
        frame = np.random.default_rng(9).integers(0, 256, (25, 33, 3), dtype=np.uint8)
        original = frame.copy()
        for name in live.OPERATORS:
            with self.subTest(operator=name):
                out = live.process_frame(frame, name)
                self.assertEqual(out.shape, (25, 33))
                self.assertEqual(out.dtype, np.uint8)
                np.testing.assert_array_equal(frame, original)

    def test_constant_scene_has_no_candidate_edges(self):
        frame = np.full((18, 23, 3), 128, np.uint8)
        for name in ('canny', 'dynamic', 'ridge', 'open_close'):
            np.testing.assert_array_equal(live.process_frame(frame, name), 0)
        for name in ('gray', 'adaptive', 'mean'):
            np.testing.assert_array_equal(live.process_frame(frame, name), 128)

    def test_input_contract(self):
        for frame in (np.zeros((2, 2), np.uint8), np.zeros((2, 2, 3), np.float32),
                      np.zeros((0, 2, 3), np.uint8)):
            with self.assertRaises(ValueError):
                live.process_frame(frame, 'adaptive')
        with self.assertRaises(ValueError):
            live.process_frame(np.zeros((2, 2, 3), np.uint8), 'unknown')

    def test_camera_released_after_open_or_read_failure(self):
        for opened in (False, True):
            cap = MagicMock()
            cap.isOpened.return_value = opened
            cap.read.return_value = (False, None)
            with patch.object(live.cv2, 'VideoCapture', return_value=cap), \
                 patch.object(live.sys, 'argv', ['live', '--headless', '--frames', '1']):
                with self.assertRaises(RuntimeError):
                    live.main()
            cap.release.assert_called_once()
