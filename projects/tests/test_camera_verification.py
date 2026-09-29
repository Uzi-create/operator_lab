from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from projects.camera import verify_camera_operators as check


class CameraVerificationTests(unittest.TestCase):
    def test_no_detection_is_distinct_from_error_and_warmup(self):
        frame = np.zeros((96, 128, 3), np.uint8)
        def rejected(): raise ValueError('insufficient scene texture')
        def failed(): raise RuntimeError('native failed')
        jobs = [('no_target', 'test', lambda: {'success': False, 'reason': 'no_edges'}),
                ('reject', 'test', rejected), ('error', 'test', failed)]
        with tempfile.TemporaryDirectory() as directory, patch.object(check, 'cases', return_value=jobs):
            result = check.verify_frames([frame, frame], Path(directory))
        self.assertEqual(result['no_target']['status'], 'completed')
        self.assertEqual(result['no_target']['no_detection_calls'], 2)
        self.assertEqual(len(result['no_target']['samples_ms']), 1)
        self.assertEqual(result['reject']['errors'][0]['kind'], 'scene_or_input_rejected')
        self.assertEqual(result['error']['errors'][0]['kind'], 'execution_error')

    def test_distance_infinity_is_valid_but_nan_is_error(self):
        value = np.array([np.inf, -np.inf, 1.], np.float32)
        self.assertEqual(check.summarize(value)['infinite'], 2)
        self.assertIsNotNone(check.preview(value.reshape(1, 3), (1, 3)))
        with self.assertRaises(AssertionError): check.summarize(np.array([np.nan]))

    def test_real_input_reference_paths_agree(self):
        frame = np.random.default_rng(13).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        result = check.check_references(frame)
        self.assertEqual(len(result), 16)
        self.assertTrue(all(item['status'] == 'passed' for item in result.values()))

    def test_camera_released_when_read_fails(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        with patch.object(check.cv2, 'VideoCapture', return_value=cap), \
             patch.object(check.sys, 'argv', ['verify', '--frames', '2']):
            with self.assertRaises(RuntimeError): check.main()
        cap.release.assert_called_once()
