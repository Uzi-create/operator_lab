"""Recording tests use the actual MJPG encoder/decoder, including Unicode paths."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from projects.camera import motion_recording as recording


def record(index=0, **values):
    result = {'frame_index': index, 'timestamp_s': index * .04, 'success': True,
              'reason': 'accepted', 'shift_xy': [2., -1.], 'api_ms': 4.,
              'processing_ms': 5., 'frame_ms': 7., 'segment_id': 0}
    result.update(values)
    return result


def decode(path):
    capture = cv2.VideoCapture(str(path))
    images = []
    try:
        if not capture.isOpened():
            raise AssertionError(f'Cannot decode {path}')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            images.append(frame)
    finally:
        capture.release()
    return images


class MotionRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / '相机运动测试'
        self.frame = np.full((65, 97, 3), (30, 100, 180), np.uint8)
        self.overlay = np.full((90, 160, 3), (150, 50, 80), np.uint8)

    def test_real_unicode_video_records_all_frames_and_pads_odd_dimensions(self):
        recorder = recording.MotionRecorder(self.root, metadata={'source': 'synthetic'}, fps=25)
        with recorder:
            for index in range(4):
                before = record(index, error_px=.15)
                original = dict(before)
                elapsed = recorder.append(self.frame, self.overlay, before)
                self.assertGreaterEqual(elapsed, 0.)
                self.assertEqual(before, original)
        summary = recorder.close()
        self.assertEqual(summary['status'], 'ok', summary['errors'])
        self.assertEqual(summary['frames'], 4)
        self.assertEqual(summary['synthetic_error_px']['count'], 4)
        rows = [json.loads(line) for line in
                (recorder.directory / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
        with (recorder.directory / 'frames.csv').open(encoding='utf-8-sig', newline='') as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(csv_rows))
        self.assertEqual([row['timestamp_s'] for row in rows], [0., .04, .08, .12])
        self.assertGreater(rows[0]['recorded_frame_ms'], rows[0]['frame_ms'])
        self.assertEqual(summary['source_timeline_frame_rate_hz'], 25.)
        for video in summary['videos']:
            images = decode(recorder.directory / video['path'])
            self.assertEqual(len(images), 4)
            self.assertEqual(video['validation']['mode'], 'all_frames')
            self.assertEqual(video['validation']['decoded_frames'], 4)
            self.assertFalse(Path(video['path']).is_absolute())
            self.assertTrue(video['path'].isascii())
            expected = self.frame if video['kind'] == 'raw' else self.overlay
            expected_shape = [66, 98] if video['kind'] == 'raw' else [90, 160]
            self.assertEqual(list(images[0].shape[:2]), expected_shape)
            self.assertLess(float(np.abs(images[0][20, 20].astype(float)
                                        - expected[20, 20].astype(float)).max()), 5.)
        self.assertEqual(summary['videos'][0]['padding_bottom_right'], [1, 1])
        self.assertFalse((recorder.directory / 'summary.json.tmp').exists())
        self.assertIn('fixed nominal FPS', (recorder.directory / 'RESULTS.md').read_text('utf-8'))

    def test_source_size_changes_create_explicit_parts_without_losing_frames(self):
        recorder = recording.MotionRecorder(self.root)
        second = np.full((80, 110, 3), 100, np.uint8)
        with recorder:
            recorder.append(self.frame, self.overlay, record(0))
            recorder.append(second, self.overlay, record(1))
            recorder.append(second, self.overlay, record(2))
        summary = recorder.close()
        self.assertFalse(summary['errors'], summary)
        raw = [item for item in summary['videos'] if item['kind'] == 'raw']
        overlays = [item for item in summary['videos'] if item['kind'] == 'overlay']
        self.assertEqual([item['frames'] for item in raw], [1, 2])
        self.assertEqual([item['frames'] for item in overlays], [3])
        self.assertEqual([len(decode(recorder.directory / item['path'])) for item in raw], [1, 2])
        self.assertEqual(raw[1]['source_shape_hw'], [80, 110])
        self.assertEqual(raw[1]['first_record_index'], 1)
        rows = [json.loads(line) for line in
                (recorder.directory / 'frames.jsonl').read_text('utf-8').splitlines()]
        self.assertNotEqual(rows[0]['raw_video'], rows[1]['raw_video'])
        self.assertEqual(rows[0]['overlay_video'], rows[1]['overlay_video'])

    def test_baselines_are_excluded_from_acceptance_and_api_timing(self):
        recorder = recording.MotionRecorder(self.root, save_video=False)
        reasons = ['no_previous_frame', 'manual_reset', 'roi_changed',
                   'frame_size_changed', 'resumed']
        with recorder:
            for index, reason in enumerate(reasons):
                recorder.append(self.frame, None, record(index, success=False, reason=reason, api_ms=0.))
            recorder.append(self.frame, None, record(5, baseline=True, reason='new_baseline', api_ms=0.))
            recorder.append(self.frame, None, record(6, api_ms=8.))
            recorder.append(self.frame, None, record(7, success=False, reason='low_texture', api_ms=12.))
        summary = recorder.close()
        self.assertEqual(summary['baseline_frames'], 6)
        self.assertEqual(summary['evaluated_frames'], 2)
        self.assertEqual(summary['accepted_frames'], 1)
        self.assertEqual(summary['rejected_frames'], 1)
        self.assertEqual(summary['acceptance_rate'], .5)
        self.assertEqual(summary['timing_ms']['api_ms']['count'], 2)
        self.assertEqual(summary['timing_ms']['api_ms']['mean'], 10.)
        self.assertEqual(summary['timing_ms']['processing_ms']['count'], 8)
        self.assertFalse(list(recorder.directory.glob('*.avi')))

    def test_nonfinite_numpy_values_become_json_null_with_field_diagnostics(self):
        recorder = recording.MotionRecorder(self.root, metadata={'nested': {'value': np.inf}},
                                            save_video=False)
        with recorder:
            recorder.append(self.frame, None, record(
                response=np.float32(np.nan), success=np.bool_(False), reason='ambiguous',
                shift_xy=np.array([np.inf, 1.]), candidate_shift_xy=np.array([np.inf, 2.]),
                diagnostics={'score': np.float64(-np.inf)}))
        summary = recorder.close()
        row = json.loads((recorder.directory / 'frames.jsonl').read_text('utf-8'))
        self.assertIsNone(row['response'])
        self.assertIsNone(row['shift_xy'])
        self.assertEqual(row['candidate_shift_xy'], [None, 2.])
        self.assertIsNone(row['diagnostics']['score'])
        self.assertIn('candidate_shift_xy[0]', row['nonfinite_fields'])
        self.assertEqual(summary['nonfinite_values_replaced_with_null'], 5)
        for filename in ('session.json', 'summary.json', 'frames.jsonl'):
            text = (recorder.directory / filename).read_text('utf-8')
            self.assertNotIn('NaN', text)
            self.assertNotIn('Infinity', text)
            json.loads(text)

    def test_real_input_never_reports_accuracy_even_if_error_field_is_present(self):
        recorder = recording.MotionRecorder(self.root, metadata={'source': 'camera'}, save_video=False)
        with recorder:
            recorder.append(self.frame, None, record(error_px=0.))
        self.assertIsNone(recorder.close()['synthetic_error_px'])
        self.assertIn('field accuracy is not measured',
                      (recorder.directory / 'RESULTS.md').read_text('utf-8'))

    def test_unique_session_keeps_previous_data_and_close_is_idempotent(self):
        first = recording.MotionRecorder(self.root, save_video=False)
        first.append(self.frame, None, record())
        summary = first.close()
        text = (first.directory / 'frames.jsonl').read_bytes()
        second = recording.MotionRecorder(self.root, save_video=False)
        second.close()
        self.assertNotEqual(first.directory, second.directory)
        self.assertEqual(text, (first.directory / 'frames.jsonl').read_bytes())
        self.assertIs(first.close(), summary)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            first.append(self.frame, None, record())

    def test_unopened_writer_fails_releases_and_leaves_error_report(self):
        writer = MagicMock()
        writer.isOpened.return_value = False
        recorder = recording.MotionRecorder(self.root)
        with patch.object(recording.cv2, 'VideoWriter', return_value=writer):
            with self.assertRaisesRegex(RuntimeError, 'Cannot open MJPG'):
                with recorder:
                    recorder.append(self.frame, self.overlay, record())
        writer.release.assert_called_once()
        self.assertTrue(recorder._jsonl.closed)
        self.assertTrue(recorder._csv_file.closed)
        summary = json.loads((recorder.directory / 'summary.json').read_text('utf-8'))
        self.assertEqual(summary['status'], 'error')
        self.assertTrue(summary['errors'])
        self.assertTrue((recorder.directory / 'RESULTS.md').is_file())

    def test_silent_video_write_failure_is_detected_by_decode(self):
        writer = MagicMock()
        writer.isOpened.return_value = True
        recorder = recording.MotionRecorder(self.root)
        with patch.object(recording.cv2, 'VideoWriter', return_value=writer):
            recorder.append(self.frame, self.overlay, record())
            summary = recorder.close()
        self.assertEqual(writer.release.call_count, 2)
        self.assertEqual(summary['status'], 'error')
        self.assertTrue(all(item['validation']['status'] == 'failed' for item in summary['videos']))
        self.assertTrue(any('Video validation failed' in error for error in summary['errors']))

    def test_write_exception_releases_resources_and_reports_original_error(self):
        writer = MagicMock()
        writer.isOpened.return_value = True
        writer.write.side_effect = RuntimeError('simulated disk failure')
        recorder = recording.MotionRecorder(self.root)
        with patch.object(recording.cv2, 'VideoWriter', return_value=writer):
            with self.assertRaisesRegex(RuntimeError, 'disk failure'):
                with recorder:
                    recorder.append(self.frame, self.overlay, record())
        writer.release.assert_called_once()
        self.assertTrue(recorder._jsonl.closed)
        self.assertIn('simulated disk failure', '\n'.join(recorder.close()['errors']))

    def test_external_failure_keeps_successfully_written_videos_and_records(self):
        recorder = recording.MotionRecorder(self.root)
        with self.assertRaisesRegex(RuntimeError, 'source disconnected'):
            with recorder:
                recorder.append(self.frame, self.overlay, record())
                raise RuntimeError('source disconnected')
        summary = recorder.close()
        self.assertEqual(summary['frames'], 1)
        self.assertEqual(summary['status'], 'error')
        self.assertTrue(all(video['validation']['status'] == 'passed' for video in summary['videos']))
        self.assertEqual(len(decode(recorder.directory / 'raw_000.avi')), 1)

    def test_validation_capture_failure_still_writes_summary(self):
        recorder = recording.MotionRecorder(self.root)
        recorder.append(self.frame, self.overlay, record())
        with patch.object(recording.cv2, 'VideoCapture', side_effect=RuntimeError('decoder unavailable')):
            summary = recorder.close()
        self.assertEqual(summary['status'], 'error')
        self.assertTrue((recorder.directory / 'summary.json').is_file())
        self.assertTrue(all(video['validation']['status'] == 'failed' for video in summary['videos']))

    def test_no_frames_has_no_misleading_rates(self):
        recorder = recording.MotionRecorder(self.root, save_video=False)
        summary = recorder.close()
        self.assertEqual(summary['frames'], 0)
        self.assertIsNone(summary['acceptance_rate'])
        self.assertIsNone(summary['source_timeline_frame_rate_hz'])
        self.assertIsNone(summary['timing_ms']['api_ms']['median'])

    def test_source_configuration_updates_session_and_actual_video_fps(self):
        recorder = recording.MotionRecorder(self.root)
        recorder.configure_source(24., {'source_fps': 24., 'timestamp_source': 'video_nominal_frame_index'})
        configured = json.loads((recorder.directory / 'session.json').read_text('utf-8'))
        self.assertEqual(configured['nominal_video_fps'], 24.)
        self.assertEqual(configured['metadata']['source_fps'], 24.)
        with recorder:
            recorder.append(self.frame, self.overlay, record())
            with self.assertRaises(RuntimeError):
                recorder.configure_source(30.)
        capture = cv2.VideoCapture(str(recorder.directory / 'raw_000.avi'))
        try:
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 24.)
        finally:
            capture.release()

    def test_discontinuous_timestamps_do_not_fabricate_a_frame_rate(self):
        recorder = recording.MotionRecorder(self.root, save_video=False)
        with recorder:
            for timestamp in (0., .1, .1, .2):
                recorder.append(self.frame, None, record(timestamp_s=timestamp))
        self.assertEqual(recorder.close()['timestamp_discontinuities'], 1)
        self.assertIsNone(recorder.close()['source_timeline_frame_rate_hz'])

    def test_failed_report_write_can_be_retried_after_resources_are_released(self):
        recorder = recording.MotionRecorder(self.root)
        recorder.append(self.frame, self.overlay, record())
        with patch.object(recording, '_atomic_text', side_effect=OSError('report unavailable')):
            with self.assertRaisesRegex(OSError, 'report unavailable'):
                recorder.close()
        self.assertTrue(recorder._jsonl.closed)
        with patch.object(recorder, '_validate_video') as validate:
            summary = recorder.close()
        validate.assert_not_called()
        self.assertEqual(summary['status'], 'ok')
        self.assertTrue((recorder.directory / 'summary.json').is_file())
        self.assertTrue((recorder.directory / 'RESULTS.md').is_file())

    def test_invalid_nominal_fps_rejected_before_session_creation(self):
        for fps in (0., -1., float('nan'), float('inf'), True):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                recording.MotionRecorder(self.root, fps=fps)
        self.assertFalse(self.root.exists())


if __name__ == '__main__':
    unittest.main()
