import unittest

import cv2
import numpy as np

from operators.handeye_ops import calibrate_eye_in_hand, calibrate_eye_in_hand_robust


def make_scene(seed=23, count=20, noise_rotation=.0005, noise_translation=.0005):
    rng = np.random.default_rng(seed)
    camera_to_gripper = np.eye(4)
    camera_to_gripper[:3, :3] = cv2.Rodrigues(np.array([.08, -.12, .05]))[0]
    camera_to_gripper[:3, 3] = [.05, -.02, .08]
    target_to_base = np.eye(4)
    target_to_base[:3, :3] = cv2.Rodrigues(np.array([.03, .05, -.02]))[0]
    target_to_base[:3, 3] = [.1, -.05, 1.4]
    robot, observations = [], []
    for _ in range(count):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(rng.uniform(-.5, .5, 3))[0]
        pose[:3, 3] = rng.uniform([-.25, -.25, .1], [.25, .25, .4])
        observed = np.linalg.inv(pose @ camera_to_gripper) @ target_to_base
        observed[:3, :3] = cv2.Rodrigues(rng.normal(0, noise_rotation, 3))[0] @ observed[:3, :3]
        observed[:3, 3] += rng.normal(0, noise_translation, 3)
        robot.append(pose)
        observations.append(observed)
    return robot, observations, camera_to_gripper, target_to_base


class HandEyeTests(unittest.TestCase):
    def test_known_transform_and_independent_target_consistency(self):
        robot, observations, true_camera, true_target = make_scene()
        result = calibrate_eye_in_hand(robot, observations)
        self.assertTrue(result['success'], result['reason'])
        estimate = result['T_gripper_from_camera']
        rotation_error = np.degrees(np.linalg.norm(cv2.Rodrigues(
            estimate[:3, :3] @ true_camera[:3, :3].T)[0]))
        self.assertLess(rotation_error, .1)
        self.assertLess(np.linalg.norm(estimate[:3, 3]-true_camera[:3, 3]), .002)
        self.assertLess(np.linalg.norm(result['T_base_from_target'][:3, 3]-true_target[:3, 3]), .002)
        self.assertLess(result['translation_rms'], .002)
        self.assertLess(result['rotation_rms_deg'], .1)
        translations = np.array([(g @ estimate @ t)[:3, 3] for g, t in zip(robot, observations)])
        errors = np.linalg.norm(translations-result['T_base_from_target'][:3, 3], axis=1)
        np.testing.assert_allclose(result['per_view_translation_error'], errors, atol=1e-12)
        self.assertAlmostEqual(result['translation_rms'], np.sqrt(np.mean(errors**2)), delta=1e-12)

    def test_five_opencv_methods_on_exact_geometry(self):
        robot, observations, truth, _ = make_scene(noise_rotation=0., noise_translation=0.)
        for method in ('park', 'tsai', 'horaud', 'andreff', 'daniilidis'):
            with self.subTest(method=method):
                result = calibrate_eye_in_hand(robot, observations, method=method)
                self.assertTrue(result['success'], result['reason'])
                np.testing.assert_allclose(result['T_gripper_from_camera'], truth, atol=1e-8)

    def test_motion_degeneracy_and_inconsistent_target(self):
        robot, observations, _, _ = make_scene()
        too_few = calibrate_eye_in_hand(robot[:3], observations[:3])
        self.assertEqual(too_few['reason'], 'insufficient_views')
        same_pose = calibrate_eye_in_hand([robot[0]]*8, [observations[0]]*8)
        self.assertEqual(same_pose['reason'], 'insufficient_motion_diversity')
        one_axis = []
        for i, pose in enumerate(robot):
            copy = pose.copy()
            copy[:3, :3] = cv2.Rodrigues(np.array([0., 0., i*.06]))[0]
            one_axis.append(copy)
        degenerate = calibrate_eye_in_hand(one_axis, observations)
        self.assertEqual(degenerate['reason'], 'insufficient_motion_diversity')
        corrupt = [pose.copy() for pose in observations]
        corrupt[0][:3, 3] += [.05, 0., 0.]
        bad = calibrate_eye_in_hand(robot, corrupt)
        self.assertEqual(bad['reason'], 'inconsistent_target_poses')
        self.assertFalse(bad['success'])

    def test_invalid_pose_and_options(self):
        robot, observations, _, _ = make_scene()
        with self.assertRaises(ValueError):
            calibrate_eye_in_hand(robot, observations[:-1])
        with self.assertRaises(ValueError):
            calibrate_eye_in_hand(robot, observations, method='unknown')
        with self.assertRaises(ValueError):
            calibrate_eye_in_hand(robot, observations, min_rotation_deg=180.)
        bad_pose = [pose.copy() for pose in robot]
        bad_pose[0][3, 3] = 2.
        with self.assertRaises(ValueError):
            calibrate_eye_in_hand(bad_pose, observations)

    def test_robust_recovers_four_corrupt_camera_poses(self):
        robot, observations, truth, _ = make_scene()
        damaged = [pose.copy() for pose in observations]
        outliers = (1, 5, 11, 17)
        for j, index in enumerate(outliers):
            damaged[index][:3, 3] += [.04+.01*j, -.03, .02]
            damaged[index][:3, :3] = cv2.Rodrigues(np.array([.04, .02, -.03]))[0] @ damaged[index][:3, :3]
        ordinary = calibrate_eye_in_hand(robot, damaged)
        self.assertFalse(ordinary['success'])
        result = calibrate_eye_in_hand_robust(robot, damaged, min_inliers=12, seed=42)
        self.assertTrue(result['success'], result['reason'])
        self.assertEqual(result['inlier_count'], 16)
        self.assertFalse(result['inliers'][list(outliers)].any())
        self.assertTrue(result['inliers'][[i for i in range(20) if i not in outliers]].all())
        self.assertLess(np.linalg.norm(result['T_gripper_from_camera'][:3, 3]-truth[:3, 3]), .002)
        self.assertGreater(result['per_view_translation_error'][list(outliers)].min(), .02)
        fixed = calibrate_eye_in_hand_robust(robot, damaged, min_inliers=12, seed=42,
                                             adaptive=False)
        self.assertTrue(fixed['success'])
        np.testing.assert_array_equal(result['inliers'], fixed['inliers'])
        self.assertLess(result['trials_run'], fixed['trials_run'])
        np.testing.assert_allclose(result['T_gripper_from_camera'], fixed['T_gripper_from_camera'],
                                   atol=1e-10)

    def test_robust_clean_input_and_invalid_settings(self):
        robot, observations, truth, _ = make_scene()
        result = calibrate_eye_in_hand_robust(robot, observations)
        self.assertTrue(result['success'], result['reason'])
        self.assertEqual(result['inlier_count'], 20)
        self.assertEqual(result['trials_run'], 1)
        self.assertLess(np.linalg.norm(result['T_gripper_from_camera'][:3, 3]-truth[:3, 3]), .002)
        for options in ({'min_inliers': 5}, {'max_trials': 0}, {'seed': -1},
                        {'max_translation_error': -1.}, {'adaptive': 1},
                        {'confidence': 1.}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                calibrate_eye_in_hand_robust(robot, observations, **options)
        damaged = [pose.copy() for pose in observations]
        for index in range(10):
            damaged[index][:3, 3] += [.03*index, .02*index, -.01*index]
        rejected = calibrate_eye_in_hand_robust(robot, damaged, min_inliers=15,
                                                max_trials=24, seed=19)
        self.assertFalse(rejected['success'])
        self.assertEqual(rejected['reason'], 'insufficient_consensus')


if __name__ == '__main__':
    unittest.main()
