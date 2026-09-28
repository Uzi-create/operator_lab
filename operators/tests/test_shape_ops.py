import unittest
from unittest.mock import patch

import cv2
import numpy as np

from operators import shape_ops as shape


def plain_shape():
    image = np.zeros((64, 72), np.float32)
    polygon = np.array([[10, 9], [48, 9], [48, 24], [62, 24], [62, 51], [10, 51]], np.int32)
    cv2.fillPoly(image, [polygon], .85)
    cv2.circle(image, (25, 29), 8, 0., -1)
    cv2.rectangle(image, (39, 37), (49, 45), 0., -1)
    return image


def render(template, center, angle=0., scale=1., size=(260, 220)):
    pivot = ((template.shape[1]-1)/2, (template.shape[0]-1)/2)
    matrix = cv2.getRotationMatrix2D(pivot, angle, scale)
    matrix[:, 2] += np.asarray(center)-pivot
    return cv2.warpAffine(template, matrix, size), np.vstack((matrix, (0, 0, 1)))


class ShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = plain_shape()
        cls.model = shape.create_shape_template(cls.template, angles=(0., 15., 25., 35.),
                                                scales=(.9, 1., 1.1))

    def test_rotation_scale_weak_texture_and_coordinate_transform(self):
        scene, expected = render(self.template, (143., 96.), 25., 1.1)
        result = shape.match_shape(scene, self.model, candidates_per_pose=8)
        self.assertTrue(result['success'], result['reason'])
        match = result['matches'][0]
        self.assertEqual(match['angle_degrees'], 25.)
        self.assertEqual(match['scale'], 1.1)
        np.testing.assert_allclose(match['center_xy'], [143, 96], atol=.8)
        np.testing.assert_allclose(match['matrix'][:2, :2], expected[:2, :2], atol=1e-12)
        np.testing.assert_allclose(match['matrix'][:2, 2], expected[:2, 2], atol=.8)
        self.assertGreater(match['support_fraction'], .97)
        self.assertLess(match['mean_distance_px'], .7)
        center_h = np.r_[self.model.center_xy, 1.]
        np.testing.assert_allclose((match['matrix']@center_h)[:2], match['center_xy'], atol=1e-12)
        corners = np.array([[0., 0.], [71., 0.], [71., 63.], [0., 63.]])
        np.testing.assert_allclose(match['corners_xy'],
                                   corners@match['matrix'][:2, :2].T+match['matrix'][:2, 2])

    def test_multiple_targets_and_nms(self):
        a, _ = render(self.template, (73, 69), 0., 1., (320, 230))
        b, _ = render(self.template, (236, 153), 25., 1.1, (320, 230))
        result = shape.match_shape(np.maximum(a, b), self.model, max_matches=2,
                                   candidates_per_pose=10)
        self.assertTrue(result['success'])
        self.assertEqual(len(result['matches']), 2)
        matches = sorted(result['matches'], key=lambda m: m['center_xy'][0])
        for match, center in zip(matches, ([73, 69], [236, 153])):
            np.testing.assert_allclose(match['center_xy'], center, atol=.85)
        self.assertEqual([(m['angle_degrees'], m['scale']) for m in matches], [(0., 1.), (25., 1.1)])

    def test_noise_and_partial_occlusion(self):
        scene, _ = render(self.template, (143, 96), 25., 1.1)
        scene[75:88, 137:172] = .15
        rng = np.random.default_rng(224)
        scene = np.clip(scene+rng.normal(0, .012, scene.shape), 0, 1).astype(np.float32)
        result = shape.match_shape(scene, self.model, min_support=.65,
                                   max_mean_distance=1.8, candidates_per_pose=10)
        self.assertTrue(result['success'], result['reason'])
        match = result['matches'][0]
        np.testing.assert_allclose(match['center_xy'], [143, 96], atol=1.2)
        self.assertEqual((match['angle_degrees'], match['scale']), (25., 1.1))
        self.assertGreater(match['support_fraction'], .7)
        self.assertLess(match['support_fraction'], 1.)

    def test_roi_is_center_constraint_in_scene_coordinates(self):
        scene, _ = render(self.template, (143, 96), 25., 1.1)
        result = shape.match_shape(scene, self.model, roi=(136, 89, 15, 15), candidates_per_pose=4)
        self.assertTrue(result['success'])
        np.testing.assert_allclose(result['matches'][0]['center_xy'], [143, 96], atol=.8)
        result = shape.match_shape(scene, self.model, roi=(10, 10, 15, 15), candidates_per_pose=4)
        self.assertFalse(result['success'])

    def test_no_edges_wrong_shape_and_too_small_scene(self):
        result = shape.match_shape(np.zeros((160, 190), np.float32), self.model)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'no_scene_edges')
        self.assertEqual(result['matches'], [])
        wrong = np.zeros((160, 190), np.float32)
        cv2.circle(wrong, (90, 80), 24, .85, -1)
        result = shape.match_shape(wrong, self.model, min_support=.85, max_mean_distance=.9)
        self.assertFalse(result['success'])
        tiny = np.zeros((12, 12), np.float32)
        tiny[3:9, 3:9] = 1.
        result = shape.match_shape(tiny, self.model)
        self.assertFalse(result['success'])

    def test_coarse_mode_final_statistics_use_all_full_resolution_points(self):
        scene, _ = render(self.template, (143, 96), 25., 1.1)
        result = shape.match_shape(scene, self.model, coarse_step=3, candidates_per_pose=5)
        self.assertTrue(result['success'])
        match = result['matches'][0]
        np.testing.assert_allclose(match['center_xy'], [143, 96], atol=.8)
        scene_edges = shape._edges(shape._image_u8(scene), self.model.canny_low,
                                  self.model.canny_high, self.model.blur_sigma)
        distance = cv2.distanceTransform((scene_edges == 0).astype(np.uint8),
                                        cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        points = (self.model.points_xy@match['matrix'][:2, :2].T+match['matrix'][:2, 2])
        values = shape._sample(distance, points)
        self.assertEqual(match['edge_count'], len(values))
        self.assertAlmostEqual(match['mean_distance_px'], float(values.mean()), places=12)
        self.assertAlmostEqual(match['clipped_mean_distance_px'], float(np.minimum(values, 6).mean()), places=12)
        self.assertEqual(match['support_fraction'], float(np.mean(values <= 2.)))

    def test_cached_arrays_own_immutable_storage_and_views_supported(self):
        source = self.template[:, ::-1].copy()[:, ::-1]
        model = shape.create_shape_template(source, angles=(0,), scales=(1,))
        points = model.points_xy.copy()
        kernel = model.poses[0].kernel.copy()
        source[:] = 0
        np.testing.assert_array_equal(model.points_xy, points)
        np.testing.assert_array_equal(model.poses[0].kernel, kernel)
        for array in (model.points_xy, model.poses[0].offsets_xy, model.poses[0].kernel):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        scene, _ = render(self.template, (119, 95))
        scene.flags.writeable = False
        self.assertTrue(shape.match_shape(scene, model, candidates_per_pose=4)['success'])

    def test_mask_excludes_edges_without_adding_mask_boundary(self):
        mask = np.zeros_like(self.template, dtype=bool)
        mask[15:44, 12:37] = True
        model = shape.create_shape_template(self.template, angles=(0,), scales=(1,), mask=mask)
        xy = model.points_xy.astype(int)
        self.assertTrue(mask[xy[:, 1], xy[:, 0]].all())
        original = shape._edges(shape._image_u8(self.template), 40., 100., .7)
        self.assertTrue((original[xy[:, 1], xy[:, 0]] > 0).all())
        self.assertLess(len(model.points_xy), len(self.model.points_xy))

    def test_kernel_preserves_bilinear_sampling_and_edge_multiplicity(self):
        offsets = np.array([[-1.2, -2.1], [0., 0.], [0., 0.], [3.7, 2.3]])
        kernel, origin = shape._kernel(offsets)
        field = np.random.default_rng(71).random((20, 22), dtype=np.float32)
        score = cv2.matchTemplate(field, kernel, cv2.TM_CCORR)
        center = np.array([10, 9])
        value = score[center[1]+origin[1], center[0]+origin[0]]
        expected = shape._sample(field, offsets+center).mean()
        self.assertAlmostEqual(float(value), float(expected), places=6)
        self.assertAlmostEqual(float(kernel.sum()), 1., places=6)

    def test_discrete_pose_grid_and_budget_are_explicit(self):
        model = shape.create_shape_template(self.template, angles=(0, 360, -360), scales=(1., 1.))
        self.assertEqual(len(model.poses), 1)
        scene, _ = render(self.template, (119, 95))
        result = shape.match_shape(scene, model, candidates_per_pose=1, refine_translation=False)
        self.assertTrue(result['success'])
        match = result['matches'][0]
        self.assertEqual(match['angle_degrees'], 0.)
        self.assertEqual(match['scale'], 1.)
        np.testing.assert_array_equal(match['center_xy'], np.rint(match['center_xy']))
        self.assertEqual(result['candidates_verified'], 1)
        self.assertTrue(result['candidate_budget_hit'])

    def test_invalid_images_models_and_parameters(self):
        invalid_images = [self.template.astype(np.uint8), self.template[None],
                          np.zeros((0, 4), np.float32), np.full((8, 8), np.nan, np.float32),
                          np.full((8, 8), 1.1, np.float64)]
        for image in invalid_images:
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                shape.create_shape_template(image)
            with self.assertRaises(ValueError):
                shape.match_shape(image, self.model)
        for args in (dict(angles=[]), dict(scales=[]), dict(angles=[np.inf]),
                     dict(scales=[0]), dict(canny_low=120), dict(blur_sigma=-1),
                     dict(min_edges=3), dict(mask=np.ones(self.template.shape, np.uint8)),
                     dict(angles=list(range(721)), scales=(.9, 1., 1.1))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                shape.create_shape_template(self.template, **args)
        with self.assertRaises(ValueError):
            shape.create_shape_template(np.zeros((40, 40), np.float32))
        with self.assertRaises(ValueError):
            shape.match_shape(self.template, None)
        for args in (dict(distance_tolerance=0), dict(min_support=0), dict(max_mean_distance=6),
                     dict(clip_distance=1), dict(candidates_per_pose=0), dict(coarse_step=0),
                     dict(refine_translation=1), dict(nms_iou=1), dict(max_matches=0),
                     dict(roi=(0, 0, 0, 3)), dict(roi=(0., 0., 2., 2.)),
                     dict(roi=(-1, 0, 3, 3)), dict(roi=(0, 0, 999, 999))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                shape.match_shape(self.template, self.model, **args)

    def test_positive_and_negative_angles_follow_opencv(self):
        model = shape.create_shape_template(self.template, angles=(-90, 0, 90), scales=(1,))
        for angle in (-90, 90):
            with self.subTest(angle=angle):
                scene, expected = render(self.template, (143, 96), angle)
                result = shape.match_shape(scene, model, candidates_per_pose=5)
                self.assertTrue(result['success'])
                match = result['matches'][0]
                self.assertEqual(match['angle_degrees'], angle)
                np.testing.assert_allclose(match['matrix'], expected, atol=.8)

    def test_collinear_template_rejected(self):
        edges = shape._edges(shape._image_u8(self.template), 40., 100., .7)
        counts = np.count_nonzero(edges, axis=1)
        mask = np.zeros_like(self.template, dtype=bool)
        mask[int(counts.argmax())] = True
        self.assertGreaterEqual(int(counts.max()), 12)
        with self.assertRaisesRegex(ValueError, 'collinear'):
            shape.create_shape_template(self.template, mask=mask)

    def test_candidate_budget_can_limit_recall_and_is_reported(self):
        model = shape.create_shape_template(self.template, angles=(0,), scales=(1,))
        a, _ = render(self.template, (73, 69), size=(320, 230))
        b, _ = render(self.template, (236, 153), size=(320, 230))
        scene = np.maximum(a, b)
        short = shape.match_shape(scene, model, max_matches=2, candidates_per_pose=1)
        full = shape.match_shape(scene, model, max_matches=2, candidates_per_pose=64)
        self.assertEqual(len(short['matches']), 1)
        self.assertTrue(short['candidate_budget_hit'])
        self.assertEqual(len(full['matches']), 2)

    def test_batched_refinement_is_exact_against_scalar_reference(self):
        rng = np.random.default_rng(803)
        distance = rng.uniform(0, 8, (100, 120)).astype(np.float32)
        centers = np.array([[0, 0], [1, 1], [42., 37.], [60.125, 70.5],
                            [79.875, 66.25], [90, 80], [100, 80], [119, 99]])
        for pose in self.model.poses:
            for roi in ((0, 0, 120, 100), (40, 35, 40, 45)):
                actual = shape._verify_many(centers, pose, distance, 6., 2., roi)
                for center, result in zip(centers, actual):
                    expected = shape._verify(center, pose, distance, 6., 2., roi)
                    if expected is None:
                        self.assertIsNone(result)
                    else:
                        np.testing.assert_array_equal(result['center_xy'], expected['center_xy'])
                        for key in expected.keys()-{'center_xy'}:
                            self.assertEqual(result[key], expected[key], key)

    def test_optimized_match_is_exact_against_scalar_reference(self):
        def reference(centers, pose, distance, clip_distance, tolerance, roi):
            return [shape._verify(center, pose, distance, clip_distance, tolerance, roi)
                    for center in centers]

        for angle, scale in ((0, .9), (25, 1.1), (35, 1)):
            scene, _ = render(self.template, (143.25, 96.375), angle, scale)
            fast = shape.match_shape(scene, self.model, candidates_per_pose=5)
            with patch.object(shape, '_verify_many', side_effect=reference), \
                    patch.object(shape, '_refinement_improvement_bound', return_value=np.inf):
                scalar = shape.match_shape(scene, self.model, candidates_per_pose=5)
            self.assertTrue(fast['success'])
            for key in fast.keys()-{'matches'}:
                self.assertEqual(fast[key], scalar[key], key)
            self.assertEqual(len(fast['matches']), len(scalar['matches']))
            for actual, expected in zip(fast['matches'], scalar['matches']):
                for key in expected:
                    if isinstance(expected[key], np.ndarray):
                        np.testing.assert_array_equal(actual[key], expected[key])
                    else:
                        self.assertEqual(actual[key], expected[key], key)

    def test_clipping_statistics_use_interpolated_distances(self):
        # The proposal surrogate clips before sampling, but accepted metrics
        # must sample the true distance first. These differ at a clip boundary.
        field = np.array([[0., 10.], [0., 10.]], np.float32)
        points = np.array([[.5, .5], [.25, .5]])
        values = shape._sample(field, points)
        clipped_after = np.minimum(values, 6.)
        clipped_before = shape._sample(np.minimum(field, 6.), points)
        np.testing.assert_array_equal(clipped_after, [5., 2.5])
        np.testing.assert_array_equal(clipped_before, [3., 1.5])
        self.assertTrue(np.all(clipped_before < clipped_after))

    def test_refinement_bound_covers_bilinear_field_changes(self):
        rng = np.random.default_rng(442)
        # Use an arbitrary field, not only unit-gradient distance transforms.
        field = rng.uniform(0, 50, (70, 80)).astype(np.float32)
        points = rng.uniform((2, 2), (77, 67), (1000, 2))
        bound = shape._refinement_improvement_bound(field)
        initial = shape._sample(field, points)
        for delta in ((-.875, -.875), (.875, -.875), (-.875, .875), (.875, .875)):
            shifted = shape._sample(field, points+delta)
            self.assertTrue(np.all(np.abs(initial-shifted) <= bound))
            self.assertLessEqual(abs(np.minimum(initial, 6).mean()-
                                     np.minimum(shifted, 6).mean()), bound)

    def test_pruning_preserves_noisy_occluded_multitarget_results(self):
        def scalar_reference(centers, pose, distance, clip_distance, tolerance, roi):
            return [shape._verify(center, pose, distance, clip_distance, tolerance, roi)
                    for center in centers]

        rng = np.random.default_rng(929)
        a, _ = render(self.template, (73, 69), 0., 1., (320, 230))
        b, _ = render(self.template, (236, 153), 25., 1.1, (320, 230))
        scene = np.maximum(a, b)
        scene[136:147, 229:258] = .15
        scene = np.clip(scene+rng.normal(0, .015, scene.shape), 0, 1).astype(np.float32)
        for step, support, mean_distance in ((1, .65, 1.8), (3, .65, 1.8),
                                             (1, .9, .5), (1, 1., .1)):
            arguments = dict(max_matches=2, candidates_per_pose=5, coarse_step=step,
                             min_support=support, max_mean_distance=mean_distance)
            fast = shape.match_shape(scene, self.model, **arguments)
            with patch.object(shape, '_refinement_improvement_bound', return_value=np.inf), \
                    patch.object(shape, '_verify_many', side_effect=scalar_reference):
                full = shape.match_shape(scene, self.model, **arguments)
            if support == .65:
                self.assertTrue(fast['success'])
            for key in fast.keys()-{'matches'}:
                self.assertEqual(fast[key], full[key], key)
            self.assertEqual(len(fast['matches']), len(full['matches']))
            for actual, expected in zip(fast['matches'], full['matches']):
                for key in expected:
                    if isinstance(expected[key], np.ndarray):
                        np.testing.assert_array_equal(actual[key], expected[key])
                    else:
                        self.assertEqual(actual[key], expected[key], key)


if __name__ == '__main__':
    unittest.main()
