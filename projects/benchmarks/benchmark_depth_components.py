"""Native and exact Python depth-component timing with label-truth checks."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from operators.depth_components_ops import depth_components
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_depth_components import make_depth, evaluate

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = dict(depth_scale=.001, max_depth=3., absolute_jump=.015, min_area=100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/depth_components_benchmark')
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(*make_depth(seed), backend='native')[1]) for seed in (982, 17, 246)]
    full, full_truth = make_depth()
    small, small_truth = make_depth(shape=(120, 160))
    a = depth_components(small, backend='native', **OPTIONS)
    b = depth_components(small, backend='numpy', **OPTIONS)
    np.testing.assert_array_equal(a['labels'], b['labels'])
    np.testing.assert_array_equal(a['labels'], small_truth)
    np.testing.assert_array_equal(depth_components(full, backend='native', **OPTIONS)['labels'], full_truth)
    cases = {
        'depth_640x480_native': lambda: depth_components(full, backend='native', **OPTIONS),
        'depth_160x120_native': lambda: depth_components(small, backend='native', **OPTIONS),
        'depth_160x120_numpy_reference': lambda: depth_components(small, backend='numpy', **OPTIONS),
    }
    timings = measure(cases, args.repeats)
    sources = ['operators/depth_components.hpp', 'operators/depth_components_ops.py',
               'projects/vision_robot/demo_depth_components.py', 'projects/benchmarks/benchmark_depth_components.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks,
                  native_numpy_exact_labels=True,
                  scope='Whole API including input validation, scaling, segmentation and region statistics. Fixed three-object synthetic metric scene. NumPy reference timed at 160x120 to keep runtimes practical.',
                  timings=timings,
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: exact known labels and native/Python parity.')


if __name__ == '__main__':
    main()
