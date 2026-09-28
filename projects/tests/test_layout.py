"""Layout regression checks: reusable library boundaries and standalone entry points."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class LayoutTests(unittest.TestCase):
    def test_library_does_not_import_application_packages(self):
        for path in (ROOT/'operators').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                names = []
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or '']
                self.assertFalse(any(name == 'projects' or name.startswith('projects.') for name in names), str(path))

    def test_entry_points_from_unrelated_working_directory(self):
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as directory:
            for script in ['build.py', 'verify.py', 'projects/examples/demo.py',
                           'projects/metal/run_samples.py', 'projects/vision_robot/demo_vision_robot.py',
                           'projects/vision_robot/demo_metrology.py', 'projects/vision_robot/demo_advanced_perception.py',
                           'projects/vision_robot/demo_depth_components.py', 'projects/benchmarks/benchmark_depth_components.py',
                           'projects/vision_robot/demo_ray_plane.py', 'projects/benchmarks/benchmark_ray_plane.py',
                           'projects/vision_robot/demo_ground_obstacles.py', 'projects/benchmarks/benchmark_ground_obstacles.py',
                           'projects/vision_robot/demo_rgbd_scene.py', 'projects/benchmarks/benchmark_rgbd_scene.py',
                           'projects/vision_robot/demo_box.py', 'projects/benchmarks/benchmark_box.py',
                           'projects/vision_robot/demo_sphere.py', 'projects/benchmarks/benchmark_sphere.py',
                           'projects/vision_robot/demo_line3d.py', 'projects/benchmarks/benchmark_line3d.py',
                           'projects/vision_robot/demo_voxel_clusters.py', 'projects/benchmarks/benchmark_voxel_clusters.py',
                           'projects/vision_robot/demo_icp_plane.py', 'projects/benchmarks/benchmark_icp_plane.py',
                           'projects/vision_robot/demo_calibration.py', 'projects/benchmarks/benchmark_calibration.py',
                           'projects/vision_robot/demo_stereo.py', 'projects/benchmarks/benchmark_stereo.py',
                           'projects/vision_robot/demo_handeye.py', 'projects/benchmarks/benchmark_handeye.py',
                           'projects/vision_robot/demo_dense_stereo.py', 'projects/benchmarks/benchmark_dense_stereo.py',
                           'projects/vision_robot/demo_stereo_rectify.py', 'projects/benchmarks/benchmark_stereo_rectify.py',
                           'projects/vision_robot/demo_stereo_file_workflow.py', 'projects/vision_robot/run_stereo_pair.py',
                           'projects/benchmarks/benchmark_advanced_perception.py',
                           'projects/benchmarks/benchmark_vision_robot.py',
                           'projects/benchmarks/benchmark_accuracy_first.py']:
                run = subprocess.run([sys.executable, '-B', str(ROOT/script), '--help'],
                                     cwd=directory, env=environment, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, script + '\n' + run.stderr)


if __name__ == '__main__':
    unittest.main()
