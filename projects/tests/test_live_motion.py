"""User-visible viewer behavior and real headless recording/replay checks."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from projects.camera import live_motion as live
from projects.tests.test_motion_session import textured_frame


ROOT = Path(__file__).resolve().parents[2]


def arguments(directory, *extra):
    return live.build_parser().parse_args(['--output', str(directory), *extra])


def saved_session(directory):
    files = list(Path(directory).glob('*/summary.json'))
    if len(files) != 1:
        raise AssertionError(f'Expected one saved session, found {files}')
    folder = files[0].parent
    return folder, json.loads(files[0].read_text(encoding='utf-8'))


def gui_mocks(stack, capture, keys=(-1,), visible=1.):
    stack.enter_context(patch.object(live.cv2, 'VideoCapture', return_value=capture))
    for name in ('namedWindow', 'setMouseCallback', 'imshow', 'destroyWindow'):
        stack.enter_context(patch.object(live.cv2, name))
    stack.enter_context(patch.object(live.cv2, 'waitKey', side_effect=keys))
    stack.enter_context(patch.object(live.cv2, 'getWindowProperty', return_value=visible))
    stack.enter_context(redirect_stdout(io.StringIO()))


class MotionViewerTests(unittest.TestCase):
    def test_letterbox_mapping_uses_source_pixels_and_ignores_bars(self):
        tf = live.ViewTransform.fit((1080, 1920), (10, 20, 800, 600))
        self.assertEqual((tf.left, tf.top, tf.display_width, tf.display_height),
                         (10, 95, 800, 450))
        self.assertIsNone(tf.canvas_to_image(400, 50))
        self.assertIsNone(tf.canvas_to_image(400, 570))
        for point in ((0, 0), (480, 270), (1918, 1078)):
            np.testing.assert_allclose(tf.canvas_to_image(*tf.image_to_canvas(*point)), point)
        np.testing.assert_allclose(tf.canvas_to_image(-30, 700, clip=True), (0, 1080))

    def test_dragged_roi_is_independent_of_drag_direction(self):
        tf = live.ViewTransform.fit((1080, 1920), (20, 200, 960, 640))
        callbacks = []
        controller = live.ROIController(on_roi=callbacks.append)
        controller.set_transform(tf)
        start = tuple(map(round, tf.image_to_canvas(320, 180)))
        end = tuple(map(round, tf.image_to_canvas(1280, 900)))
        for first, last in ((start, end), (end, start)):
            controller.handle_mouse(cv2.EVENT_LBUTTONDOWN, *first)
            controller.handle_mouse(cv2.EVENT_MOUSEMOVE, *last)
            self.assertEqual(controller.selection, (320, 180, 960, 720))
            controller.handle_mouse(cv2.EVENT_LBUTTONUP, *last)
        self.assertEqual(callbacks, [(320, 180, 960, 720)] * 2)
        self.assertIsNone(controller.selection)

    def test_letterbox_and_tiny_drags_cannot_install_a_roi(self):
        callbacks = []
        controller = live.ROIController(on_roi=callbacks.append)
        controller.set_transform(live.ViewTransform.fit((1080, 1920), (20, 200, 960, 640)))
        controller.handle_mouse(cv2.EVENT_LBUTTONDOWN, 100, 210)
        controller.handle_mouse(cv2.EVENT_LBUTTONUP, 500, 600)
        controller.handle_mouse(cv2.EVENT_LBUTTONDOWN, 200, 400)
        controller.handle_mouse(cv2.EVENT_LBUTTONUP, 210, 410)
        self.assertEqual(callbacks, [])
        self.assertTrue(controller.message)

    def test_rejected_and_paused_frames_never_draw_a_green_motion_marker(self):
        frame = np.zeros((192, 256, 3), np.uint8)
        original = frame.copy()
        candidate = dict(success=False, reason='inconsistent_motion', shift_xy=(12, -7),
                         candidate_shift_xy=(12, -7), response=.95, accumulated_xy=(12, -7))
        for success, paused in ((False, False), (True, True), (True, False)):
            with self.subTest(success=success, paused=paused):
                record = dict(candidate, success=success)
                canvas, tf = live.render_motion(frame, record, [(0, 0), (12, -7)],
                                                roi=(30, 20, 180, 144), paused=paused)
                viewport = canvas[tf.top:tf.top + tf.display_height,
                                  tf.left:tf.left + tf.display_width]
                b, g, r = [viewport[:, :, c].astype(int) for c in range(3)]
                has_green = np.any((g > b + 80) & (g > r + 80))
                self.assertEqual(bool(has_green), success and not paused)
                np.testing.assert_array_equal(frame, original)

    def test_window_close_releases_capture_and_finalizes_recording(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30.
        capture.read.return_value = (True, textured_frame())
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            gui_mocks(stack, capture, visible=0.)
            result = live.run(arguments(directory, '--camera', '0', '--no-video'))
            _, persisted = saved_session(directory)
        capture.release.assert_called_once()
        self.assertEqual(result['frames'], 1)
        self.assertEqual(persisted['status'], 'ok')

    def test_large_arrow_stays_inside_preview_without_changing_measurement(self):
        frame = np.zeros((192, 256, 3), np.uint8)
        record = dict(success=True, reason='ok', shift_xy=(400., -300.))
        canvas, tf = live.render_motion(frame, record, [])
        b, g, r = [canvas[:, :, c].astype(int) for c in range(3)]
        green = (g > b + 80) & (g > r + 80)
        self.assertTrue(green.any())
        # Anti-aliased strokes can extend one pixel past the clipped centerline.
        green[max(0, tf.top-2):tf.top+tf.display_height+2,
              max(0, tf.left-2):tf.left+tf.display_width+2] = False
        self.assertFalse(green.any())
        self.assertEqual(record['shift_xy'], (400., -300.))

    def test_unopened_source_preserves_error_report_and_releases_capture(self):
        capture = MagicMock()
        capture.isOpened.return_value = False
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(live.cv2, 'VideoCapture', return_value=capture), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Could not open'):
                live.run(arguments(directory, '--camera', '0', '--headless', '--frames', '1', '--no-video'))
            _, summary = saved_session(directory)
        capture.release.assert_called_once()
        self.assertEqual(summary['status'], 'error')
        self.assertEqual(summary['frames'], 0)

    def test_video_eof_distinguishes_declared_truncation_from_complete_input(self):
        for declared in (0, 1, 5):
            with self.subTest(declared=declared):
                capture = MagicMock()
                capture.isOpened.return_value = True
                capture.get.side_effect = lambda key: 24. if key == cv2.CAP_PROP_FPS else declared
                capture.read.side_effect = [(True, textured_frame()), (False, None)]
                with tempfile.TemporaryDirectory() as directory, \
                     patch.object(live.cv2, 'VideoCapture', return_value=capture), \
                     redirect_stdout(io.StringIO()):
                    args = arguments(directory, '--video', 'input.avi', '--headless', '--no-video')
                    if declared == 5:
                        with self.assertRaisesRegex(RuntimeError, 'ended early'):
                            live.run(args)
                    else:
                        live.run(args)
                    folder, summary = saved_session(directory)
                    config = json.loads((folder / 'session.json').read_text('utf-8'))
                capture.release.assert_called_once()
                self.assertEqual(summary['status'], 'error' if declared == 5 else 'ok')
                self.assertEqual(config['nominal_video_fps'], 24.)
                self.assertEqual(summary['nominal_video_fps'], 24.)

    def test_pause_resume_adds_no_fake_samples_and_restarts_baseline(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30.
        capture.read.side_effect = [(True, textured_frame()), (True, textured_frame(seed=735))]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            gui_mocks(stack, capture, keys=(ord(' '), ord(' '), ord('q')))
            result = live.run(arguments(directory, '--camera', '0', '--frames', '3', '--no-video'))
            folder, _ = saved_session(directory)
            rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
        capture.release.assert_called_once()
        self.assertEqual(capture.read.call_count, 2)
        self.assertEqual(result['frames'], 2)
        self.assertEqual(result['baseline_frames'], 2)
        self.assertTrue(all(row['baseline'] and row['shift_xy'] is None for row in rows))

    def test_read_failure_releases_source_and_saves_error_report(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30.
        capture.read.return_value = (False, None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(live.cv2, 'VideoCapture', return_value=capture), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'frames'):
                live.run(arguments(directory, '--camera', '0', '--headless', '--frames', '1', '--no-video'))
            _, summary = saved_session(directory)
        capture.release.assert_called_once()
        self.assertEqual(summary['status'], 'error')
        self.assertEqual(summary['frames'], 0)
        self.assertTrue(summary['errors'])

    def test_demo_recording_and_video_replay_make_decodable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            demo_dir = Path(directory) / 'demo'
            result = live.run(arguments(demo_dir, '--demo', '--headless', '--frames', '6',
                                        '--width', '256', '--height', '192', '--analysis-width', '240'))
            folder, saved = saved_session(demo_dir)
            self.assertEqual(result, saved)
            self.assertEqual(saved['status'], 'ok')
            self.assertEqual(saved['frames'], 6)
            self.assertEqual(saved['baseline_frames'], 1)
            self.assertGreater(saved['accepted_frames'], 0)
            self.assertGreater(saved['synthetic_error_px']['count'], 0)
            self.assertLess(saved['synthetic_error_px']['max'], 1.)
            self.assertEqual(saved['timing_ms']['api_ms']['count'], 5)
            rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
            with (folder / 'frames.csv').open(encoding='utf-8', newline='') as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(csv_rows), 6)
            self.assertEqual([row['frame_index'] for row in rows], list(range(6)))
            for row in rows:
                for key in ('read_ms', 'api_ms', 'processing_ms', 'render_ms',
                            'frame_ms', 'video_write_ms', 'loop_ms', 'recorded_frame_ms'):
                    self.assertIsInstance(row[key], (int, float), key)
                    self.assertGreaterEqual(row[key], 0., key)
                self.assertGreaterEqual(row['loop_ms'], row['frame_ms'])
            self.assertTrue((folder / 'RESULTS.md').is_file())
            self.assertEqual(len(saved['videos']), 2)
            for video in saved['videos']:
                self.assertEqual(video['validation']['status'], 'passed')
                cap = cv2.VideoCapture(str(folder / video['path']))
                try:
                    decoded = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        self.assertEqual(tuple(frame.shape[:2]), tuple(video['encoded_shape_hw']))
                        decoded += 1
                finally:
                    cap.release()
                self.assertEqual(decoded, 6)
            raw = next(item for item in saved['videos'] if item['path'].startswith('raw_'))
            replay_dir = Path(directory) / 'replay'
            replay = live.run(arguments(replay_dir, '--video', str(folder / raw['path']), '--headless'))
            _, saved_replay = saved_session(replay_dir)
            self.assertEqual(replay['frames'], 6)
            self.assertEqual(saved_replay['status'], 'ok')
            self.assertIsNone(saved_replay['synthetic_error_px'])
            self.assertTrue(all(video['validation']['status'] == 'passed' for video in saved_replay['videos']))

    def test_main_reports_invalid_options_without_secondary_exception(self):
        with redirect_stderr(io.StringIO()):
            result = live.main(['--demo', '--headless', '--width', '20'])
        self.assertNotEqual(result, 0)

    def test_standalone_help_works_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'projects/camera/live_motion.py'), '--help'],
                                    cwd=directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--roi', result.stdout)


if __name__ == '__main__':
    unittest.main()
