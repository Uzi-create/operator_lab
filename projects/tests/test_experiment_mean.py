import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from projects.camera import experiment_mean as experiment


class MeanExperimentTests(unittest.TestCase):
    def test_black_capture_rejected_before_report(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'almost black'):
                experiment.benchmark(np.zeros((96, 128, 3), np.uint8), [1], 2, Path(directory))
            self.assertFalse((Path(directory)/'report.json').exists())

    def test_comparison_detects_changed_algorithm_results(self):
        frame = np.random.default_rng(12).integers(0, 256, (20, 24, 3), dtype=np.uint8)
        gray, pixels = experiment.prepare(frame)
        before = pixels[:]
        outputs, times, errors = experiment.compare(pixels, 24, 20, 3)
        self.assertLessEqual(errors['max_abs_error'], 1e-6)
        self.assertEqual(pixels, before)
        self.assertEqual(experiment.panel(gray, outputs, times, errors, 3).shape, (830, 1000, 3))
        original = experiment.run
        def broken(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs['mode'] == 'mean_naive': result[0] += .1
            return result
        with patch.object(experiment, 'run', side_effect=broken):
            with self.assertRaises(AssertionError): experiment.compare(pixels, 24, 20, 3)

    def test_benchmark_saves_readable_report_and_images(self):
        frame = np.random.default_rng(8).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            report = experiment.benchmark(frame, [1, 3], 2, Path(directory))
            self.assertEqual([r['radius'] for r in report['results']], [1, 3])
            self.assertTrue((Path(directory)/'RESULTS.md').exists())
            saved = experiment.cv2.imread(str(Path(directory)/'original.png'))
            np.testing.assert_array_equal(frame, saved)

    def test_camera_released_on_read_failure(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        with patch.object(experiment.cv2, 'VideoCapture', return_value=cap), \
             patch.object(experiment.sys, 'argv', ['experiment', '--benchmark']):
            with self.assertRaises(RuntimeError): experiment.main()
        cap.release.assert_called_once()
