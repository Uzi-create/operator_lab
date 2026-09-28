"""Exact old/new shape matching comparison with alternating cached-model timing."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from operators import shape_ops as updated
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_advanced_perception import make_inputs

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT/'projects/benchmarks/baselines/shape_before_batching.py'


def old_module():
    name = '_shape_before_batching'
    spec = importlib.util.spec_from_file_location(name, BEFORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model(module, template):
    return module.create_shape_template(template, angles=(-20., 0., 25.),
                                        scales=(.9, 1., 1.1))


def same(a, b, path='output'):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            same(a[key], b[key], path+'.'+key)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            same(x, y, path+f'[{index}]')
    else:
        assert a == b, (path, a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/continuous_batch6/shape_comparison')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    old = old_module()
    checks = []
    for seed in (71, 17, 106, 903):
        data = make_inputs(seed)
        first, second = model(old, data['template']), model(updated, data['template'])
        for options in ({'max_matches': 2, 'candidates_per_pose': 8},
                        {'max_matches': 2},
                        {'max_matches': 2, 'coarse_step': 3, 'candidates_per_pose': 5},
                        {'max_matches': 2, 'refine_translation': False}):
            before = old.match_shape(data['scene'], first, **options)
            after = updated.match_shape(data['scene'], second, **options)
            same(before, after)
            checks.append(dict(seed=seed, options=options, matches=len(after['matches']),
                               candidate_budget_hit=after['candidate_budget_hit']))
    data = make_inputs()
    first, second = model(old, data['template']), model(updated, data['template'])
    options = dict(max_matches=2)
    timings = measure({'before_default_24': lambda: old.match_shape(data['scene'], first, **options),
                       'after_default_24': lambda: updated.match_shape(data['scene'], second, **options)},
                      args.repeats)
    before = timings['before_default_24']['median_ms']
    after = timings['after_default_24']['median_ms']
    report = dict(repeats=args.repeats, warmups=3,
                  exact_output_agreement=True, checks=checks, timings=timings,
                  median_speedup=before/after,
                  scope='Same cached pose grid, all full-resolution edges, thresholds and candidate budget. Old/new APIs alternated; 16 fixed truth/parameter cases compared field-by-field. No changed search semantics.',
                  hashes=dict(before=hashlib.sha256(BEFORE.read_bytes()).hexdigest(),
                              after=hashlib.sha256((ROOT/'operators/shape_ops.py').read_bytes()).hexdigest()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Old {before:.4f} ms; new {after:.4f} ms; speedup {before/after:.2f}x')
    print('PASS: 16 fixed cases, every result field exactly equal.')


if __name__ == '__main__':
    main()
