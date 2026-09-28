import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from operators.image_io import write_image
from projects.vision_robot.demo_dense_stereo import make_scene
from projects.vision_robot.run_stereo_pair import load_calibration, run_pair


class StereoFileWorkflowTests(unittest.TestCase):
    def test_saved_depth_and_xyz_match_known_scale(self):
        scene = make_scene()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            left, right, calibration = root/'left.png', root/'right.png', root/'calibration.json'
            write_image(left, scene['left'])
            write_image(right, scene['right'])
            camera = scene['camera']
            intrinsics = dict(width=camera.width, height=camera.height, fx=camera.fx,
                              fy=camera.fy, cx=camera.cx, cy=camera.cy)
            transform = np.eye(4)
            transform[0, 3] = -.12
            calibration.write_text(json.dumps(dict(left_camera=intrinsics, right_camera=intrinsics,
                                                   T_right_from_left=transform.tolist())), encoding='utf-8')
            report = run_pair(left, right, calibration, root/'result')
            self.assertTrue(report['success'])
            self.assertAlmostEqual(report['median_depth'], 5., delta=.06)
            depth = np.load(root/'result/depth.npy')
            cloud = np.load(root/'result/point_cloud_xyz.npy')
            self.assertEqual(report['valid_pixels'], len(cloud))
            self.assertEqual(report['valid_pixels'], int(np.isfinite(depth).sum()))
            self.assertAlmostEqual(float(np.median(cloud[:, 2])), 5., delta=.06)
            self.assertEqual(len(report['outputs']), 6)
            missing = json.loads(calibration.read_text(encoding='utf-8'))
            del missing['right_camera']
            calibration.write_text(json.dumps(missing), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_calibration(calibration)

    def test_wrong_image_dimensions_fail_before_outputs(self):
        scene = make_scene()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            left, right, calibration = root/'left.png', root/'right.png', root/'calibration.json'
            write_image(left, scene['left'][:-1])
            write_image(right, scene['right'])
            camera = scene['camera']
            intrinsics = dict(width=camera.width, height=camera.height, fx=camera.fx,
                              fy=camera.fy, cx=camera.cx, cy=camera.cy)
            transform = np.eye(4)
            transform[0, 3] = -.12
            calibration.write_text(json.dumps(dict(left_camera=intrinsics, right_camera=intrinsics,
                                                   T_right_from_left=transform.tolist())), encoding='utf-8')
            with self.assertRaises(ValueError):
                run_pair(left, right, calibration, root/'result')
            self.assertFalse((root/'result').exists())


if __name__ == '__main__':
    unittest.main()
