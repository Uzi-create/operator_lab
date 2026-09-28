"""Batch calipers versus independent existing calipers, with known-geometry checks."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from operators.metrology_ops import measure_line, measure_rectangle
from operators.vision_ops import measure_edges, fit_line
from projects.benchmarks.benchmark_vision_robot import measure
from projects.vision_robot.demo_metrology import make_inputs, evaluate

ROOT = Path(__file__).resolve().parents[2]


def independent_line(data, n=48):
    tangent = data['line_end']-data['line_start']
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    points = []
    for center in np.linspace(data['line_start'], data['line_end'], n):
        output = measure_edges(data['line_image'], center-10*normal, center+10*normal,
                               width=5, sigma=1., threshold=.03)
        if output['edges']:
            points.append(min(output['edges'], key=lambda edge: abs(edge['distance']-10))['xy'])
    return fit_line(np.asarray(points), threshold=.5, min_inliers=8, seed=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/metrology_benchmark')
    parser.add_argument('--repeats', type=int, default=31)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 10000:
        parser.error('Use 5..10000 repetitions')
    data = make_inputs()
    line, rectangle, _ = evaluate(data)
    checks = []
    for seed, angle, noise in [(43, 23., .008), (11, -32., .012), (74, 0., .02), (8, 41., 0.)]:
        _, _, errors = evaluate(make_inputs(seed, noise, angle))
        checks.append({'seed': seed, 'angle': angle, 'noise_sigma': noise, **errors})
    independent = independent_line(data)
    agreement = float(abs((independent['point']-line['point'])@line['normal']))
    assert agreement < .02, agreement
    cases = {
        'line_48_independent_calipers': lambda: independent_line(data),
        'line_48_batched_calipers': lambda: measure_line(data['line_image'], data['line_start'], data['line_end'], num_calipers=48),
        'rectangle_4x24_batched_calipers': lambda: measure_rectangle(data['rectangle_image'], data['rectangle_center']+[1, -1],
                                        data['rectangle_size']+[1, -1], data['rectangle_angle']+1),
    }
    report = {'opencv': cv2.__version__, 'numpy': np.__version__, 'opencv_threads': cv2.getNumThreads(),
              'repeats': args.repeats, 'checks': checks, 'independent_line_agreement_px': agreement,
              'scope': '640x480 full API, validation included. Independent reference has its own floating sample-grid rounding; compare known truth as well as agreement.',
              'hashes': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                         ['operators/metrology_ops.py', 'projects/vision_robot/demo_metrology.py', 'projects/benchmarks/benchmark_metrology.py']},
              'timings': measure(cases, args.repeats)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in report['timings'].items():
        print(name, 'median_ms', round(row['median_ms'], 4), 'p95_ms', round(row['p95_ms'], 4))
    print('Independent fit agreement:', agreement)


if __name__ == '__main__':
    main()
