import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from projects.camera import focus_meter as focus


class FocusMeterTests(unittest.TestCase):
    def test_blur_reduces_metrics_for_same_pattern(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(frame, 'FOCUS 123 ABC', (25, 230), cv2.FONT_HERSHEY_SIMPLEX, 2., (255, 255, 255), 3)
        before = frame.copy()
        crop, sharp = focus.measure(frame)
        _, blurred = focus.measure(cv2.GaussianBlur(frame, (0, 0), 3.))
        self.assertGreater(sharp['laplacian'], blurred['laplacian'])
        self.assertGreater(sharp['gradient_energy'], blurred['gradient_energy'])
        np.testing.assert_array_equal(frame, before)
        np.testing.assert_array_equal(crop, frame)

    def test_constant_input_is_zero_not_a_fake_quality_label(self):
        frame = np.full((1080, 1920, 3), 128, np.uint8)
        crop, stats = focus.measure(frame)
        self.assertEqual(crop.shape, (480, 640, 3))
        self.assertEqual(stats['laplacian'], 0.)
        self.assertEqual(stats['gradient_energy'], 0.)
        self.assertEqual(focus.draw(frame, crop, stats, [], 1).shape, (800, 1100, 3))

    def test_read_failure_releases_camera(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        with patch.object(focus.cv2, 'VideoCapture', return_value=cap), \
             patch.object(focus.sys, 'argv', ['focus', '--headless', '--frames', '1']):
            with self.assertRaises(RuntimeError): focus.main()
        cap.release.assert_called_once()
