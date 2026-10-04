"""Session-level checks for trustworthy image displacement recording."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from projects.camera.motion_session import MotionSession


def textured_frame(height=240, width=320, seed=734):
    gray = np.random.default_rng(seed).integers(0, 256, (height, width), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (0, 0), .8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def displaced(frame, dx, dy):
    height, width = frame.shape[:2]
    return cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]),
                          (width, height), borderMode=cv2.BORDER_CONSTANT)


def estimate(success, shift=(3., -2.), reason='ok'):
    return dict(success=success, reason=reason, shift_xy=np.asarray(shift),
                response=.9, analysis_shape_hw=(120, 160))


class MotionSessionTests(unittest.TestCase):
    def test_first_frame_is_only_a_baseline(self):
        frame = textured_frame()
        session = MotionSession()
        with patch('projects.camera.motion_session.estimate_scene_motion') as operator:
            record = session.process(frame, timestamp_s=10.)
        operator.assert_not_called()
        self.assertEqual(record['frame_index'], 0)
        self.assertFalse(record['success'])
        self.assertTrue(record['baseline'])
        self.assertIsNone(record['shift_xy'])
        self.assertIsNone(record['accumulated_xy'])
        self.assertIsNone(record['pair_interval_s'])
        self.assertEqual(len(session.path), 0)

    def test_roi_preserves_signed_original_pixel_units_and_inputs(self):
        original = textured_frame(480, 640)
        for dx, dy in ((12, -7), (-9, 6)):
            with self.subTest(shift=(dx, dy)):
                current = displaced(original, dx, dy)
                saved = original.copy(), current.copy()
                # ROI origin and analysis scaling must not change dx/dy units.
                session = MotionSession(roi=(93, 61, 480, 360), max_analysis_width=240)
                session.process(original, timestamp_s=1.)
                record = session.process(current, timestamp_s=1.05,
                                         expected_shift=(dx, dy))
                self.assertTrue(record['success'], record)
                np.testing.assert_allclose(record['shift_xy'], (dx, dy), atol=.9)
                np.testing.assert_allclose(record['accumulated_xy'], (dx, dy), atol=.9)
                self.assertEqual(tuple(record['roi_xywh']), (93, 61, 480, 360))
                self.assertEqual(tuple(record['frame_shape_hw']), (480, 640))
                self.assertLess(record['error_px'], .9)
                self.assertAlmostEqual(record['pair_interval_s'], .05)
                self.assertGreaterEqual(record['processing_ms'], record['api_ms'])
                np.testing.assert_array_equal(original, saved[0])
                np.testing.assert_array_equal(current, saved[1])

    def test_reused_capture_buffer_does_not_overwrite_previous_frame(self):
        buffer = textured_frame()
        moved = displaced(buffer, 10, -5)
        session = MotionSession()
        session.process(buffer, timestamp_s=0.)
        buffer[:] = moved
        record = session.process(buffer, timestamp_s=.04)
        self.assertTrue(record['success'], record)
        np.testing.assert_allclose(record['shift_xy'], (10, -5), atol=.6)

    def test_rejection_breaks_path_and_next_pair_starts_a_new_segment(self):
        session = MotionSession()
        frame = textured_frame()
        session.process(frame, timestamp_s=0.)
        with patch('projects.camera.motion_session.estimate_scene_motion', side_effect=[
                estimate(True, (3., -2.)),
                estimate(False, (99., 0.), 'inconsistent_motion'),
                estimate(True, (-1., 4.))]):
            first = session.process(frame, timestamp_s=.04)
            rejected = session.process(frame, timestamp_s=.08)
            self.assertFalse(rejected['success'])
            self.assertIsNone(rejected['shift_xy'])
            self.assertIsNone(rejected['accumulated_xy'])
            np.testing.assert_allclose(rejected['candidate_shift_xy'], (99, 0))
            self.assertEqual(len(session.path), 0)
            following = session.process(frame, timestamp_s=.12)
        self.assertTrue(following['success'])
        self.assertNotEqual(following['segment_id'], first['segment_id'])
        np.testing.assert_allclose(following['accumulated_xy'], (-1, 4))
        np.testing.assert_allclose(session.path[-1], (-1, 4))
        self.assertAlmostEqual(following['pair_interval_s'], .04)

    def test_blank_pair_does_not_claim_zero_motion(self):
        session = MotionSession()
        blank = np.zeros((192, 256, 3), np.uint8)
        session.process(blank, timestamp_s=0.)
        record = session.process(blank, timestamp_s=.04)
        self.assertFalse(record['success'])
        self.assertEqual(record['reason'], 'insufficient_texture')
        self.assertIsNone(record['shift_xy'])
        self.assertIsNone(record['error_px'])

    def test_roi_change_and_explicit_reset_require_a_new_baseline(self):
        session = MotionSession()
        frame = textured_frame()
        session.process(frame, timestamp_s=0.)
        session.set_roi((31, 17, 192, 160))
        with patch('projects.camera.motion_session.estimate_scene_motion') as operator:
            changed = session.process(frame, timestamp_s=.04)
            session.reset('resume')
            resumed = session.process(frame, timestamp_s=.08)
        operator.assert_not_called()
        for record in (changed, resumed):
            self.assertFalse(record['success'])
            self.assertIsNone(record['shift_xy'])
            self.assertIsNone(record['accumulated_xy'])
        self.assertEqual(len(session.path), 0)

    def test_invalid_roi_is_rejected_without_silently_clipping(self):
        frame = textured_frame()
        for roi in ((-1, 0, 96, 96), (0, 0, 95, 96), (0, 0, 96, 0),
                    (0.5, 0, 96, 96), (230, 0, 96, 96), (0, 150, 96, 96),
                    (0, 0, 96), (True, 0, 96, 96)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                session = MotionSession(roi=roi)
                session.process(frame, timestamp_s=0.)

    def test_frame_size_change_resets_baseline_and_cancels_old_roi(self):
        session = MotionSession(roi=(160, 120, 128, 112))
        session.process(textured_frame(), timestamp_s=0.)
        with patch('projects.camera.motion_session.estimate_scene_motion') as operator:
            record = session.process(textured_frame(128, 160), timestamp_s=.04)
        operator.assert_not_called()
        self.assertFalse(record['success'])
        self.assertIsNone(record['shift_xy'])
        self.assertIsNone(session.roi)
        self.assertEqual(tuple(record['roi_xywh']), (0, 0, 160, 128))

    def test_duplicate_or_reversed_timestamp_breaks_pairing(self):
        frame = textured_frame()
        for timestamp in (10., 9.):
            with self.subTest(timestamp=timestamp):
                session = MotionSession()
                session.process(frame, timestamp_s=10.)
                with patch('projects.camera.motion_session.estimate_scene_motion') as operator:
                    record = session.process(frame, timestamp_s=timestamp)
                operator.assert_not_called()
                self.assertFalse(record['success'])
                self.assertIsNone(record['shift_xy'])
                self.assertIsNone(record['pair_interval_s'])

    def test_invalid_frames_and_nonfinite_timestamps_raise(self):
        for frame in (np.zeros((96, 96), np.uint8), np.zeros((96, 96, 4), np.uint8),
                      np.zeros((95, 96, 3), np.uint8), np.zeros((96, 96, 3), np.float32)):
            with self.subTest(shape=frame.shape, dtype=frame.dtype), self.assertRaises(ValueError):
                MotionSession().process(frame)
        for timestamp in (float('nan'), float('inf')):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                MotionSession().process(textured_frame(), timestamp_s=timestamp)

    def test_rejected_pair_still_advances_previous_frame(self):
        session = MotionSession()
        frames = [np.full((96, 96, 3), level, np.uint8) for level in (20, 40, 60)]
        session.process(frames[0], timestamp_s=0.)
        with patch('projects.camera.motion_session.estimate_scene_motion', side_effect=[
                estimate(False, reason='low_response'), estimate(True)]) as operator:
            session.process(frames[1], timestamp_s=.04)
            session.process(frames[2], timestamp_s=.08)
        previous, current = operator.call_args.args
        np.testing.assert_allclose(previous, 40. / 255.)
        np.testing.assert_allclose(current, 60. / 255.)

    def test_invalid_success_candidate_does_not_enter_path(self):
        session = MotionSession()
        frame = textured_frame()
        session.process(frame, timestamp_s=0.)
        with patch('projects.camera.motion_session.estimate_scene_motion',
                   return_value=estimate(True, (float('nan'), 3.))):
            record = session.process(frame, timestamp_s=.04)
        self.assertFalse(record['success'])
        self.assertIsNone(record['shift_xy'])
        self.assertIsNone(record['accumulated_xy'])
        self.assertEqual(len(session.path), 0)

    def test_long_trace_is_bounded_without_resetting_accumulation(self):
        with patch.object(MotionSession, 'MAX_PATH_POINTS', 4):
            session = MotionSession()
        frame = textured_frame()
        session.process(frame, timestamp_s=0.)
        with patch('projects.camera.motion_session.estimate_scene_motion',
                   return_value=estimate(True, (2., -1.))):
            for index in range(1, 9):
                record = session.process(frame, timestamp_s=index * .04)
        self.assertLessEqual(len(session.path), 4)
        np.testing.assert_allclose(record['accumulated_xy'], (16., -8.))
        np.testing.assert_allclose(session.path[-1], (16., -8.))


if __name__ == '__main__':
    unittest.main()
