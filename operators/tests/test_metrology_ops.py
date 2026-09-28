import unittest

import cv2
import numpy as np

from operators.metrology_ops import measure_line, measure_rectangle
from operators.vision_ops import measure_edges


def line_scene(shape=(240, 320), angle=17., point=(80., 100.)):
    angle = np.radians(angle)
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    y, x = np.indices(shape, dtype=np.float64)
    distance = (x-point[0])*normal[0]+(y-point[1])*normal[1]
    gray = (.5+.4*np.tanh(distance/1.2)).astype(np.float32)
    return gray, direction, normal


def rectangle_scene(center=(160.3, 120.6), size=(140.4, 78.7), angle=23.):
    y, x = np.indices((260, 340), dtype=np.float64)
    angle = np.radians(angle)
    u = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-u[1], u[0]])
    p = np.stack((x-center[0], y-center[1]), axis=-1)
    distances = np.minimum(size[0]/2-np.abs(p@u), size[1]/2-np.abs(p@v))
    return (.5+.4*np.tanh(distances/1.1)).astype(np.float32)


class MultiCaliperLineTests(unittest.TestCase):
    def test_known_line_subpixel_and_scan_reversal(self):
        image, direction, normal = line_scene()
        start = np.array([80., 101.])-40*direction
        end = np.array([80., 101.])+170*direction
        for first, second, polarity in [(start, end, 'bright'), (end, start, 'dark')]:
            out = measure_line(image, first, second, polarity=polarity)
            self.assertTrue(out['success'], out['reason'])
            self.assertLess(abs((out['point']-[80, 100])@normal), .05)
            self.assertGreater(abs(out['direction']@direction), .99999)
            self.assertLess(out['rms'], .05)
            self.assertGreater(out['coverage'], .9)
            residuals = abs((out['edge_points']-out['point'])@out['normal'])
            np.testing.assert_array_equal(out['inliers'], residuals <= .5)
            self.assertAlmostEqual(out['rms'], float(np.sqrt(np.mean(residuals[out['inliers']]**2))), places=12)

    def test_batch_sampling_against_independent_calipers(self):
        image, tangent, normal = line_scene()
        start = np.array([80., 101.])-30*tangent
        end = np.array([80., 101.])+150*tangent
        out = measure_line(image, start, end, num_calipers=12, min_inliers=6,
                           search_half_length=8., polarity='bright')
        self.assertTrue(out['success'])
        expected = []
        for center in np.linspace(start, end, 12):
            reference = measure_edges(image, center-8*normal, center+8*normal,
                                      width=5, sigma=1, polarity='bright', threshold=.03)
            edge = min(reference['edges'], key=lambda edge: abs(edge['distance']-8))
            expected.append(edge['xy'])
        np.testing.assert_allclose(out['edge_points'], expected, atol=.002)

    def test_noisy_occlusion_and_missing_border_calipers(self):
        image, direction, normal = line_scene()
        rng = np.random.default_rng(62)
        image = np.clip(image + rng.normal(0, .012, image.shape), 0, 1).astype(np.float32)
        image[:, 130:160] = .5
        start = np.array([80., 101.])-80*direction
        end = np.array([80., 101.])+260*direction
        out = measure_line(image, start, end, min_inliers=12, min_coverage=.6)
        self.assertTrue(out['success'], out['reason'])
        self.assertLess(abs((out['point']-[80, 100])@normal), .1)
        self.assertLess(out['supported_calipers'].sum(), 32)
        self.assertLess(len(out['edge_points']), out['supported_calipers'].sum())

    def test_blank_wrong_polarity_coverage_and_invalid(self):
        image, direction, _ = line_scene()
        start, end = np.array([80., 101.])-30*direction, np.array([80., 101.])+150*direction
        for image_arg, kwargs in [(np.zeros_like(image), {}), (image, {'polarity': 'dark'})]:
            self.assertFalse(measure_line(image_arg, start, end, **kwargs)['success'])
        limited = np.full_like(image, .5)
        limited[:, 130:170] = image[:, 130:170]
        out = measure_line(limited, start, end, min_inliers=3, min_coverage=.8)
        self.assertFalse(out['success'])
        for kwargs in [dict(width=0), dict(num_calipers=3), dict(threshold=0),
                       dict(min_inlier_ratio=0), dict(min_coverage=2), dict(seed=-1),
                       dict(selection='random'), dict(search_half_length=2)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                measure_line(image, start, end, **kwargs)


class RectangleMetrologyTests(unittest.TestCase):
    def test_rotated_rectangle_known_geometry(self):
        for angle in [0., 23., -38.]:
            image = rectangle_scene(angle=angle)
            out = measure_rectangle(image, (161., 120.), (141., 78.), angle+1.)
            self.assertTrue(out['success'], out['reason'])
            np.testing.assert_allclose(out['center_xy'], [160.3, 120.6], atol=.1)
            self.assertAlmostEqual(out['width_px'], 140.4, delta=.12)
            self.assertAlmostEqual(out['height_px'], 78.7, delta=.12)
            self.assertAlmostEqual(out['angle_deg'], angle, delta=.06)
            self.assertLess(out['orthogonality_error_deg'], .1)

    def test_occlusion_and_contrast_reversal(self):
        image = rectangle_scene()
        image[68:78, 130:160] = .5
        for source in [image, 1-image]:
            out = measure_rectangle(source, (160, 121), (140, 79), 23.)
            self.assertTrue(out['success'], out['reason'])
            self.assertAlmostEqual(out['width_px'], 140.4, delta=.2)

    def test_blank_missing_side_and_invalid(self):
        image = rectangle_scene()
        self.assertFalse(measure_rectangle(np.zeros_like(image), (160, 120), (140, 79), 23.)['success'])
        missing = image.copy()
        missing[:, 210:] = .5
        self.assertFalse(measure_rectangle(missing, (160, 120), (140, 79), 23.)['success'])
        for kwargs in [dict(size=(5, 5)), dict(center=(np.nan, 120)),
                       dict(search_half_length=45), dict(max_angle_error=0), dict(num_calipers=3)]:
            arguments = dict(center=(160, 120), size=(140, 79), angle_deg=23.)
            arguments.update(kwargs)
            with self.assertRaises(ValueError):
                measure_rectangle(image, **arguments)


if __name__ == '__main__':
    unittest.main()
