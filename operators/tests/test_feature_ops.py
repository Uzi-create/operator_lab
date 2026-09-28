import unittest
from unittest.mock import patch

import cv2
import numpy as np

import operators.feature_ops as f
def textured_image(seed=83):
    rng = np.random.default_rng(seed)
    image = np.full((260, 320), 90, np.uint8)
    for _ in range(150):
        x, y = rng.integers([16, 16], [304, 244])
        color = int(rng.integers(15, 245))
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 10)), color, -1)
    for _ in range(28):
        start, end = rng.integers([15, 15], [305, 245], (2, 2))
        cv2.line(image, tuple(start), tuple(end), int(rng.integers(20, 235)), 1)
    cv2.putText(image, 'VISION 42', (28, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, 245, 2)
    return image.astype(np.float32) / 255


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = textured_image()
        cls.model = f.create_orb_template(cls.template)
        cls.source_corners = np.array([[0, 0], [319, 0], [319, 259], [0, 259]], np.float32)

    def test_perspective_known_corners_and_diagnostics(self):
        target = np.array([[104, 68], [421, 82], [400, 354], [76, 326]], np.float32)
        matrix = cv2.getPerspectiveTransform(self.source_corners, target)
        scene = cv2.warpPerspective(self.template, matrix, (520, 440))
        result = f.locate_planar_template(scene, self.model)
        self.assertTrue(result['success'], result['reason'])
        np.testing.assert_allclose(result['corners_xy'], target, atol=3)
        self.assertGreater(result['inlier_count'], 70)
        self.assertLess(result['reprojection_rms'], 1.5)
        self.assertEqual(result['inliers'].dtype, bool)
        self.assertEqual(len(result['inliers']), result['match_count'])
        self.assertGreater(result['template_coverage'], .2)

    def test_rotation_scale_and_partial_occlusion(self):
        matrix = cv2.getRotationMatrix2D((160, 130), 24, .9)
        matrix[:, 2] += [95, 65]
        homography = np.vstack((matrix, [0, 0, 1]))
        scene = cv2.warpPerspective(self.template, homography, (520, 440))
        scene[130:190, 230:310] = .3
        result = f.locate_planar_template(scene, self.model, mutual=False)
        self.assertTrue(result['success'], result['reason'])
        expected = cv2.perspectiveTransform(self.source_corners[None], homography)[0]
        np.testing.assert_allclose(result['corners_xy'], expected, atol=4)

    def test_blank_and_no_match(self):
        result = f.locate_planar_template(np.zeros((300, 350), np.float32), self.model)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'insufficient_scene_features')
        rng = np.random.default_rng(1)
        result = f.locate_planar_template(rng.random((360, 420), dtype=np.float32), self.model)
        self.assertFalse(result['success'])
        self.assertIsNone(result['homography'])
        self.assertIsNone(result['corners_xy'])
        with self.assertRaises(ValueError):
            f.create_orb_template(np.ones((200, 200), np.float32))

    def test_cached_model_and_views(self):
        source = self.template.astype(np.float64)[:, ::-1]
        model = f.create_orb_template(source)
        points, descriptors = model.points_xy.copy(), model.descriptors.copy()
        source[:] = 0
        np.testing.assert_array_equal(model.points_xy, points)
        np.testing.assert_array_equal(model.descriptors, descriptors)
        self.assertFalse(model.points_xy.flags.writeable)
        self.assertFalse(model.descriptors.flags.writeable)
        with self.assertRaises(ValueError):
            model.points_xy.setflags(write=True)
        with self.assertRaises(ValueError):
            model.descriptors.setflags(write=True)
        # Only the scene is detected; the template's cached features are reused.
        with patch.object(f, '_features', wraps=f._features) as extract:
            result = f.locate_planar_template(self.template, self.model)
        self.assertTrue(result['success'])
        self.assertEqual(extract.call_count, 1)

    def test_invalid_images_and_parameters(self):
        for image in [self.template.astype(np.uint8), self.template[None],
                      np.zeros((0, 2), np.float32), np.full((10, 10), np.nan, np.float32),
                      np.full((10, 10), 1.1, np.float32)]:
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaises(ValueError):
                    f.locate_planar_template(image, self.model)
        for args in [dict(ratio=1), dict(ransac_threshold=0), dict(min_matches=3),
                     dict(min_inliers=3), dict(min_inlier_ratio=0), dict(confidence=1),
                     dict(min_template_coverage=0), dict(max_iterations=0), dict(mutual=1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                f.locate_planar_template(self.template, self.model, **args)
        for args in [dict(max_features=20), dict(scale_factor=1), dict(levels=0),
                     dict(fast_threshold=-1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                f.create_orb_template(self.template, **args)

    def test_collinear_and_duplicate_correspondences_rejected(self):
        # Native matching with exact distinct descriptors; isolate geometric failure
        # independently of feature detector responses on a particular OpenCV build.
        rng = np.random.default_rng(55)
        descriptors = rng.integers(0, 256, (20, 32), dtype=np.uint8)
        points = np.column_stack((np.arange(20) * 6 + 20, np.arange(20) * 4 + 30)).astype(np.float32)
        model = f.OrbTemplate((260, 320), points, descriptors, 1500, 1.2, 8, 15)
        with patch.object(f, '_features', return_value=(points + [5, 9], descriptors.copy())):
            result = f.locate_planar_template(self.template, model)
        self.assertEqual(result['reason'], 'degenerate_matches')
        duplicate = np.tile([30., 40.], (20, 1)).astype(np.float32)
        with patch.object(f, '_features', return_value=(duplicate, descriptors.copy())):
            result = f.locate_planar_template(self.template, model)
        self.assertEqual(result['reason'], 'insufficient_matches')
        self.assertEqual(result['match_count'], 1)

    def test_pole_and_collapsed_projection_rejected(self):
        horizon = np.eye(3)
        horizon[2] = [-.01, 0, 1]
        self.assertIsNone(f._corners(horizon, (100, 200)))
        collapsed = np.diag([1e-6, 1e-6, 1])
        self.assertIsNone(f._corners(collapsed, (100, 200)))

    def test_filtered_reverse_queries_equal_full_queries_with_ties(self):
        # Many duplicated descriptors deliberately create equal-distance ties.
        # A sparse reverse query must keep the same nearest template index.
        rng = np.random.default_rng(90)
        base = rng.integers(0, 256, (60, 32), dtype=np.uint8)
        template = np.repeat(base, 3, axis=0)
        scene = np.vstack((base[::-1], base[::2], rng.integers(0, 256, (40, 32), dtype=np.uint8)))
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        full = matcher.match(scene, template)
        selected = sorted(rng.choice(len(scene), 25, replace=False))
        sparse = matcher.match(scene[selected], template)
        for match in sparse:
            expected = full[selected[match.queryIdx]]
            self.assertEqual(match.trainIdx, expected.trainIdx)
            self.assertEqual(match.distance, expected.distance)


if __name__ == '__main__':
    unittest.main()
