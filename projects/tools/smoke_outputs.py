"""Exercise real demo entry points and validate saved artifacts, not only kernels."""
from pathlib import Path
import json
import os
import subprocess
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def run_demos(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONUTF8'] = '1'
    cases = [(name, folder, []) for name, folder in [
        ('projects/examples/demo.py', 'basic'), ('projects/examples/demo_advanced.py', 'advanced'),
        ('projects/examples/demo_geometry.py', 'geometry'), ('projects/competition/demo_competition.py', 'competition'),
        ('projects/metal/run_samples.py', 'metal'), ('projects/metal/compare_versions.py', 'metal_v2'),
        ('projects/metal/demo_defect_ops.py', 'diagnostics'),
        ('projects/vision_robot/demo_vision_robot.py', 'vision_robot'),
        ('projects/vision_robot/demo_metrology.py', 'metrology'),
        ('projects/vision_robot/demo_advanced_perception.py', 'advanced_perception'),
        ('projects/vision_robot/demo_depth_components.py', 'depth_components'),
        ('projects/vision_robot/demo_ray_plane.py', 'ray_plane'),
        ('projects/vision_robot/demo_ground_obstacles.py', 'ground_obstacles'),
        ('projects/vision_robot/demo_rgbd_scene.py', 'rgbd_scene'),
        ('projects/vision_robot/demo_box.py', 'oriented_box'),
        ('projects/vision_robot/demo_sphere.py', 'sphere'),
        ('projects/vision_robot/demo_line3d.py', 'line3d'),
        ('projects/vision_robot/demo_voxel_clusters.py', 'voxel_clusters'),
        ('projects/vision_robot/demo_icp_plane.py', 'icp_plane'),
        ('projects/vision_robot/demo_calibration.py', 'calibration'),
        ('projects/vision_robot/demo_stereo.py', 'stereo'),
        ('projects/vision_robot/demo_handeye.py', 'handeye'),
        ('projects/vision_robot/demo_dense_stereo.py', 'dense_stereo'),
        ('projects/vision_robot/demo_stereo_rectify.py', 'stereo_rectify'),
        ('projects/vision_robot/demo_stereo_file_workflow.py', 'stereo_file_workflow')]]
    for script, folder in [('projects/metal/compare_versions.py', 'new_v2'),
                           ('projects/metal/demo_defect_ops.py', 'new_diagnostics')]:
        cases.append((script, folder, ['--manifest', str(ROOT/'projects/metal/new_sample_rois.json')]))
    sample_dir = ROOT/'projects/metal/samples'
    sample_manifests = [ROOT/'projects/metal/sample_rois.json',
                        ROOT/'projects/metal/new_sample_rois.json']
    if not (sample_dir.is_dir() and all(path.is_file() for path in sample_manifests)):
        cases = [case for case in cases if not case[0].startswith('projects/metal/')]
        print('Metal photo demos skipped: private samples are not part of the public source.', flush=True)
    for script, folder, extra in cases:
        result = subprocess.run([sys.executable, str(ROOT/script), '--output', str(output/folder)]+extra,
                                cwd=ROOT, env=environment, capture_output=True, text=True, encoding='utf-8')
        (output/(folder+'.log')).write_text(result.stdout+result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{script} failed: {result.stderr}')
        print('Demo passed:', script, folder, flush=True)
    count = 0
    for path in output.rglob('*.png'):
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise AssertionError(f'Invalid PNG: {path}')
        count += 1
    for path in output.rglob('*.html'):
        path.read_text(encoding='utf-8')
    report = json.loads((output/'competition/report.json').read_text(encoding='utf-8'))
    for case in report['cases']:
        path = output/'competition'/(case['name']+'_support.png')
        mask = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        assert mask.shape == (240, 320)
        expected = case['cluster_count'] if case['accepted'] else 0
        assert np.count_nonzero(mask) == expected, (case['name'], expected)
    (output/'verification.json').write_text(json.dumps({'pngs_decoded': count,
        'demo_runs': len(cases), 'support_masks_match_estimator': True}, indent=2), encoding='utf-8')
    print('Validated PNGs:', count, '; competition masks match estimator')
