"""Accuracy gate and alternating old/new timings for result-preserving changes."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

import operators.feature_ops as feature_ops
import operators.measurement_ops as measurement_ops
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_vision_robot import make_inputs

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    from projects.benchmarks.legacy_loader import execute_legacy
    execute_legacy(spec, module)
    return module


def assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        else:
            assert a[key] == b[key], (key, a[key], b[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=ROOT/'projects/output/accuracy_first/before')
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/accuracy_first')
    args = parser.parse_args()
    before = load('feature_accuracy_baseline', args.before/'feature_ops.py')
    old_phase = load('phase_accuracy_baseline', args.before/'measurement_ops.py')
    data = make_inputs()
    old_model = before.create_orb_template(data['template'])
    model = feature_ops.create_orb_template(data['template'])
    rng = np.random.default_rng(410)
    scenes = []
    for angle, scale in [(0, 1), (-35, .8), (27, 1.1), (45, 1), (-12, 1.2)]:
        transform = cv2.getRotationMatrix2D((159.5, 129.5), angle, scale)
        transform[:, 2] += [160, 110]
        image = cv2.warpAffine(data['template'], transform, (640, 480))
        # Fixed modest sensor perturbations, not tuned against returned results.
        image = np.clip(cv2.GaussianBlur(image, (0, 0), .5) * .85 + .05 +
                        rng.normal(0, .005, image.shape), 0, 1).astype(np.float32)
        original_corners = np.array([[0, 0], [319, 0], [319, 259], [0, 259]], np.float64)
        expected = original_corners @ transform[:, :2].T + transform[:, 2]
        scenes.append((f'angle{angle}_scale{scale}', image, expected))
    scenes += [('perspective_occlusion', data['scene'], data['target_corners']),
               ('blank', np.zeros_like(data['scene']), None),
               ('unrelated_noise', rng.random(data['scene'].shape, dtype=np.float32), None)]
    equality = []
    for name, image, expected in scenes:
        for mutual in [True, False]:
            old = before.locate_planar_template(image, old_model, mutual=mutual)
            new = feature_ops.locate_planar_template(image, model, mutual=mutual)
            assert_same(old, new)
            row = {'scene': name, 'mutual': mutual, 'all_outputs_exact_equal': True,
                   'success': new['success'], 'reason': new['reason']}
            if expected is None:
                assert not new['success'], name
            else:
                assert new['success'], (name, new['reason'])
                error = float(np.linalg.norm(new['corners_xy'] - expected, axis=1).max())
                old_error = float(np.linalg.norm(old['corners_xy'] - expected, axis=1).max())
                # Gate this optimization on accuracy preservation. Separately
                # report the absolute target: a pre-existing error must not be
                # hidden by calling exact baseline equality "accurate".
                assert error <= old_error + 1e-12, (name, old_error, error)
                row['before_max_corner_error_px'] = old_error
                row['max_corner_error_px'] = error
                row['meets_4px_example_target'] = error < 4.
            equality.append(row)
    phases = []
    for shape in [(75, 75), (81, 125), (135, 225), (75, 128), (128, 75), (128, 160)]:
        image = rng.random(shape, dtype=np.float32)
        moving = np.roll(image, (-3, 5), axis=(0, 1))
        old = old_phase.estimate_translation(image, moving, window=False)
        new = measurement_ops.estimate_translation(image, moving, window=False)
        assert new['success']
        np.testing.assert_allclose(new['shift_xy'], [5, -3], atol=1e-4)
        phases.append({'shape_hw': shape, 'before_error_xy': (old['shift_xy'] - [5, -3]).tolist(),
                       'after_error_xy': (new['shift_xy'] - [5, -3]).tolist()})
    timings = measure({
        'before_orb': lambda: before.locate_planar_template(data['scene'], old_model),
        'after_orb': lambda: feature_ops.locate_planar_template(data['scene'], model),
        'before_phase': lambda: old_phase.estimate_translation(data['reference'], data['moving']),
        'after_phase': lambda: measurement_ops.estimate_translation(data['reference'], data['moving']),
    }, 41)
    paths = [args.before/'feature_ops.py', args.before/'measurement_ops.py',
             ROOT/'operators/feature_ops.py', ROOT/'operators/measurement_ops.py', Path(__file__).resolve()]
    result = {'opencv': cv2.__version__, 'numpy': np.__version__, 'opencv_threads': cv2.getNumThreads(),
              'scope': 'Same-process alternating API timings; 3 warmups, 41 samples. Fixed descriptors, thresholds, input resolution and RANSAC settings.',
              'hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              'matching_equality': equality, 'phase_bias_regression': phases, 'timings': timings}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Matching scenes with exact output equality:', len(equality))
    print('Phase dimension cases:', len(phases))
    print('Pre-existing absolute accuracy limits:', [row['scene'] for row in equality
          if row.get('meets_4px_example_target') is False])
    for name, values in timings.items():
        print(name, 'median_ms', round(values['median_ms'], 4), 'p95_ms', round(values['p95_ms'], 4))


if __name__ == '__main__':
    main()
