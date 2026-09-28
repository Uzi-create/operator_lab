import unittest

import cv2
import numpy as np

from operators.calibration_ops import calibrate_planar_camera, detect_chessboard_corners


def calibration_scene(seed=702, count=24, noise=.15, distortion=None):
    rng = np.random.default_rng(seed)
    xy = np.indices((6, 9)).transpose(1, 2, 0).reshape(-1, 2)[:, ::-1] * .05
    board = np.column_stack((xy, np.zeros(len(xy))))
    matrix = np.array([[800., 0., 640.], [0., 820., 480.], [0., 0., 1.]])
    if distortion is None:
        distortion = np.array([-.12, .025, .001, -.002, .004])
    distortion = np.asarray(distortion, dtype=np.float64)
    views = []
    attempts = 0
    while len(views) < count:
        attempts += 1
        if attempts > count*100:
            raise RuntimeError('synthetic board coverage could not be generated')
        rvec = rng.uniform([-.5, -.55, -.2], [.5, .55, .2])
        tvec = rng.uniform([-.4, -.33, .65], [.28, .25, 1.25])
        pixels = cv2.projectPoints(board, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
        if (pixels[:, 0].min() < 20 or pixels[:, 0].max() > 1259 or
                pixels[:, 1].min() < 20 or pixels[:, 1].max() > 939):
            continue
        views.append(pixels + rng.normal(0, noise, pixels.shape))
    return board, views, matrix, np.asarray(distortion)


class CalibrationTests(unittest.TestCase):
    def test_known_camera_and_independent_final_residual(self):
        board, views, truth, _ = calibration_scene()
        result = calibrate_planar_camera(board, views, (1280, 960))
        self.assertTrue(result['success'], result['reason'])
        camera = result['camera']
        for estimated, actual in zip((camera.fx, camera.fy, camera.cx, camera.cy),
                                     (truth[0, 0], truth[1, 1], truth[0, 2], truth[1, 2])):
            self.assertLess(abs(estimated-actual), 8.)
        self.assertEqual(result['distortion'].shape, (5,))
        self.assertLess(result['rms_px'], .25)
        self.assertGreater(result['normal_spread_deg'], 20.)
        matrix = np.array([[camera.fx, 0., camera.cx], [0., camera.fy, camera.cy], [0., 0., 1.]])
        errors = []
        for pixels, pose, reported in zip(views, result['T_camera_from_board'], result['per_view_rms_px']):
            rvec = cv2.Rodrigues(pose[:3, :3])[0]
            projection = cv2.projectPoints(board, rvec, pose[:3, 3], matrix,
                                           result['distortion'])[0].reshape(-1, 2)
            distances = np.linalg.norm(projection-pixels, axis=1)
            errors.extend(distances)
            self.assertAlmostEqual(reported, np.sqrt(np.mean(distances**2)), delta=1e-9)
            self.assertTrue(((board @ pose[:3, :3].T + pose[:3, 3])[:, 2] > 0).all())
        self.assertAlmostEqual(result['rms_px'], np.sqrt(np.mean(np.square(errors))), delta=1e-9)

    def test_rational_model_and_failure_diagnostics(self):
        board, views, _, _ = calibration_scene(distortion=[-.12, .025, .001, -.002, .004,
                                                            .02, -.01, .003])
        result = calibrate_planar_camera(board, views, (1280, 960), distortion_model='rational8')
        self.assertTrue(result['success'], result['reason'])
        self.assertEqual(result['distortion'].shape, (8,))
        self.assertLess(result['rms_px'], .25)
        too_few = calibrate_planar_camera(board, views[:3], (1280, 960))
        self.assertEqual(too_few['reason'], 'insufficient_views')
        self.assertIsNone(too_few['camera'])
        tilted = calibrate_planar_camera(board, [views[0].copy() for _ in range(8)], (1280, 960))
        self.assertFalse(tilted['success'])
        self.assertIn(tilted['reason'], ('insufficient_view_tilt_diversity', 'invalid_calibration',
                                        'opencv_calibration_failed'))
        swapped = [v.copy() for v in views]
        swapped[0][[0, 1]] = swapped[0][[1, 0]]
        bad = calibrate_planar_camera(board, swapped, (1280, 960))
        self.assertFalse(bad['success'])
        self.assertEqual(bad['reason'], 'reprojection_quality_failed')

    def test_invalid_observations_and_geometry(self):
        board, views, _, _ = calibration_scene(count=8)
        for invalid in (np.full_like(views[0], np.nan),
                        np.full_like(views[0], -1.), views[0][:-1]):
            with self.subTest(kind=str(invalid.shape)), self.assertRaises(ValueError):
                calibrate_planar_camera(board, [invalid]+views[1:], (1280, 960))
        with self.assertRaises(ValueError):
            calibrate_planar_camera(board+np.array([0, 0, 1]), views, (1280, 960))
        with self.assertRaises(ValueError):
            calibrate_planar_camera(board, views, (700, 400))
        with self.assertRaises(ValueError):
            calibrate_planar_camera(board, views, (1280, 960), distortion_model='unknown')

    def test_checkerboard_detection_and_missing_pattern(self):
        image = np.zeros((560, 740), np.uint8)
        for row in range(7):
            for col in range(10):
                if (row+col) % 2 == 0:
                    image[50+row*60:50+(row+1)*60, 50+col*60:50+(col+1)*60] = 255
        result = detect_chessboard_corners(image, (9, 6))
        self.assertTrue(result['success'])
        self.assertEqual(result['corners_xy'].shape, (54, 2))
        expected = np.array([(110+col*60, 110+row*60) for row in range(6) for col in range(9)])
        for corner in result['corners_xy']:
            self.assertLess(np.min(np.linalg.norm(expected-corner, axis=1)), 1.)
        self.assertFalse(detect_chessboard_corners(np.zeros_like(image), (9, 6))['success'])
        with self.assertRaises(ValueError):
            detect_chessboard_corners(image.astype(np.float32)*2, (9, 6))
        with self.assertRaises(ValueError):
            detect_chessboard_corners(image, (2, 6))


if __name__ == '__main__':
    unittest.main()
