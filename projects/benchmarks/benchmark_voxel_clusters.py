"""Native versus independent Python voxel graph clustering timings and truth."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from operators.voxel_cluster_ops import voxel_clusters
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_voxel_clusters import make_scene, evaluate

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = dict(connectivity=26, min_voxels=8, min_points=100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/voxel_clusters_benchmark')
    parser.add_argument('--repeats', type=int, default=11)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    checks = [dict(seed=seed, **evaluate(make_scene(seed))[1]) for seed in (613, 81, 207, 735)]
    large = make_scene()
    small = make_scene(per_object=700, outliers=80)
    native = voxel_clusters(small['points'], .09, backend='native', **OPTIONS)
    reference = voxel_clusters(small['points'], .09, backend='numpy', **OPTIONS)
    np.testing.assert_array_equal(native['labels'], reference['labels'])
    np.testing.assert_array_equal(native['voxel_labels'], reference['voxel_labels'])
    cases = {
        'voxel_18500_native_full_api': lambda: voxel_clusters(large['points'], .09, backend='native', **OPTIONS),
        'voxel_2180_native_full_api': lambda: voxel_clusters(small['points'], .09, backend='native', **OPTIONS),
        'voxel_2180_python_reference': lambda: voxel_clusters(small['points'], .09, backend='numpy', **OPTIONS),
    }
    timings = measure(cases, args.repeats)
    files = ['operators/voxel_components.hpp', 'operators/voxel_cluster_ops.py',
             'projects/vision_robot/demo_voxel_clusters.py', 'projects/benchmarks/benchmark_voxel_clusters.py']
    report = dict(repeats=args.repeats, warmups=3, checks=checks, timings=timings,
                  native_reference_exact_labels=True,
                  scope='Whole API includes voxel quantization, NumPy unique mapping, C++/Python connectivity and cluster statistics. Reference timed on 2180 points for practicality. Synthetic cluster truth.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four known-truth scenes and native/reference label parity.')


if __name__ == '__main__':
    main()
