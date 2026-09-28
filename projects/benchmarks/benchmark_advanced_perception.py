"""Full-API shape/ICP/PnP timings with known-truth and independent NN checks."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from operators.pose_ops import estimate_pose_pnp
from operators.registration_ops import NearestNeighborIndex, icp_point_to_point
from operators.shape_ops import match_shape
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_advanced_perception import make_inputs, make_model, evaluate

ROOT = Path(__file__).resolve().parents[2]


def build_index(points):
    with NearestNeighborIndex(points, backend='native') as index:
        return len(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/advanced_perception_benchmark')
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--opencv-threads', type=int)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('repeats must be 5..10000')
    if args.opencv_threads is not None:
        if not 1 <= args.opencv_threads <= 256:
            parser.error('opencv-threads must be 1..256')
        cv2.setNumThreads(args.opencv_threads)
    checks = []
    for seed in (71, 17, 106, 903):
        _, check = evaluate(make_inputs(seed))
        checks.append({'seed': seed, **check})
    data = make_inputs()
    model = make_model(data)
    rng = np.random.default_rng(567)
    target = rng.normal(0, 1, (8000, 3))
    query = rng.normal(0, 1, (2000, 3))
    with NearestNeighborIndex(target, backend='native') as native, \
            NearestNeighborIndex(target, backend='numpy') as reference, \
            NearestNeighborIndex(data['target'], backend='native') as icp_index:
        fast = native.query(query, max_distance=.3)
        slow = reference.query(query, max_distance=.3)
        np.testing.assert_array_equal(fast['indices'], slow['indices'])
        np.testing.assert_allclose(fast['squared_distances'], slow['squared_distances'], rtol=2e-15, atol=0.)
        icp_args = dict(max_distance=.09, trim_fraction=.9, min_overlap=.7)
        icp_native = icp_point_to_point(data['source'], icp_index, **icp_args)
        icp_reference = icp_point_to_point(data['source'], data['target'], backend='numpy', **icp_args)
        np.testing.assert_allclose(icp_native['transform'], icp_reference['transform'], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(icp_native['inliers'], icp_reference['inliers'])
        with NearestNeighborIndex(data['target'], backend='numpy') as independent:
            final = independent.query(icp_native['transformed_source'], max_distance=.09)
        np.testing.assert_array_equal(final['indices'], icp_native['target_indices'])
        np.testing.assert_allclose(final['distances'], icp_native['distances'], rtol=2e-15, atol=0.)
        cases = {
            'shape_build_9_poses': lambda: make_model(data),
            'shape_360x260_9poses_8candidates': lambda: match_shape(data['scene'], model, max_matches=2, candidates_per_pose=8),
            'shape_360x260_9poses_default24candidates': lambda: match_shape(data['scene'], model, max_matches=2),
            'nn_build_8000_native': lambda: build_index(target),
            'nn_query_2000_into_8000_native_cached': lambda: native.query(query, max_distance=.3),
            'nn_query_2000_into_8000_numpy_cached': lambda: reference.query(query, max_distance=.3),
            'icp_620_into_780_native_build_included': lambda: icp_point_to_point(data['source'], data['target'], backend='native', **icp_args),
            'icp_620_into_780_native_cached': lambda: icp_point_to_point(data['source'], icp_index, **icp_args),
            'icp_620_into_780_numpy': lambda: icp_point_to_point(data['source'], data['target'], backend='numpy', **icp_args),
            'pnp_160_25percent_outliers': lambda: estimate_pose_pnp(data['objects'], data['pixels'], data['camera'],
                distortion=data['distortion'], reprojection_threshold=1.5, min_inliers=110),
        }
        timings = measure(cases, args.repeats)
    paths = ['operators/shape_ops.py', 'operators/registration_ops.py', 'operators/nearest_neighbor.hpp',
             'operators/pose_ops.py', 'operators/operators.dll', 'projects/vision_robot/demo_advanced_perception.py',
             'projects/benchmarks/benchmark_advanced_perception.py']
    report = dict(platform=platform.platform(), cpu=platform.processor(), python=platform.python_version(),
                  numpy=np.__version__, opencv=cv2.__version__, opencv_threads=cv2.getNumThreads(),
                  repeats=args.repeats, warmups=3, timings=timings, checks=checks,
                  exact_nn_index_agreement=True, icp_final_residual_independently_verified=True,
                  native_numpy_icp_transform_max_difference=float(np.max(abs(icp_native['transform']-icp_reference['transform']))),
                  scope='Wall-clock public APIs including validation and allocation. NN target is cached; ICP build/cached cases separated. Alternating order. Synthetic precision does not establish field accuracy.',
                  hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths if (ROOT/name).exists()})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in timings.items():
        print(f"{name}: median={row['median_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms")
    print('PASS: four known-truth scenes, exact NN agreement, final ICP residuals.')


if __name__ == '__main__':
    main()
