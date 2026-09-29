"""Paired, operator-only benchmark for the camera image-motion estimators.

Run ``python -B projects/tools/benchmark_scene_motion.py`` from any directory.
The deterministic synthetic inputs and a single OpenCV thread make old/new
results comparable on this machine. Camera capture, display and conversion
from BGR are intentionally outside the timed region.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from operators.measurement_ops import estimate_scene_motion, estimate_translation


def texture(shape, seed):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.random(shape, dtype=np.float32), (0, 0), .8)


def translate(image, dx, dy=0, *, border=cv2.BORDER_CONSTANT):
    h, w = image.shape
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]),
                          (w, h), borderMode=border)


def inputs():
    full = texture((1080, 1920), 77)
    small = texture((240, 320), 78)
    circular = texture((128, 160), 79)
    stripes = np.broadcast_to(
        .5 + .4 * np.sin(np.arange(320, dtype=np.float32) * 2 * np.pi / 20),
        (240, 320)).copy()
    background = .45 + .03 * texture((240, 320), 80)
    foreground = texture((150, 240), 81)
    first = background.copy()
    first[45:195, 40:280] = foreground
    second = translate(background, -8, border=cv2.BORDER_REFLECT101)
    second[45:195, 65:305] = foreground
    return {
        '1080p_small_shift': (full, translate(full, 12, -7), (12, -7)),
        'cropped_large_shift': (small, translate(small, 190), (190, 0)),
        'circular_alias': (circular, np.roll(circular, 90, axis=1), None),
        'periodic_texture': (stripes, np.roll(stripes, 60, axis=1), None),
        'two_motion_layers': (first, second, None),
    }


def summarize(result, truth):
    shift = result['shift_xy']
    return {
        'success': bool(result['success']),
        'reason': str(result['reason']),
        'shift_xy': None if shift is None else np.asarray(shift).tolist(),
        'error_px': None if shift is None or truth is None else
                    float(np.linalg.norm(np.asarray(shift) - truth)),
        'response': float(result['response']),
    }


def time_pair(reference, moving, truth, rounds):
    methods = {
        'old_periodic': estimate_translation,
        'new_scene': estimate_scene_motion,
    }
    samples = {name: [] for name in methods}
    results = {}
    for name, method in methods.items():
        for _ in range(3):
            method(reference, moving)
    # Alternate call order so gradual CPU clock/temperature changes do not
    # systematically favour one implementation.
    for repeat in range(rounds):
        order = list(methods) if repeat % 2 == 0 else list(reversed(methods))
        for name in order:
            start = perf_counter()
            result = methods[name](reference, moving)
            samples[name].append((perf_counter() - start) * 1000)
            results[name] = summarize(result, truth)
    report = {}
    for name in methods:
        values = np.asarray(samples[name])
        report[name] = {
            **results[name],
            'median_ms': float(np.median(values)),
            'p90_ms': float(np.percentile(values, 90)),
            'min_ms': float(values.min()),
            'max_ms': float(values.max()),
        }
    report['median_speedup_old_over_new'] = (
        report['old_periodic']['median_ms'] / report['new_scene']['median_ms'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=9,
                        help='timed runs per method and scene (default: 9)')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'projects/output/scene_motion_benchmark.json')
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error('--rounds must be >= 3')
    prior_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        scenes = inputs()
        report = {
            'opencv_version': cv2.__version__,
            'numpy_version': np.__version__,
            'opencv_threads': cv2.getNumThreads(),
            'runs_per_method': args.rounds,
            'timing_scope': 'operator call only; synthetic float32 grayscale input',
            'cases': {},
        }
        for name, (reference, moving, truth) in scenes.items():
            report['cases'][name] = {
                'shape_hw': list(reference.shape),
                'expected_shift_xy': truth,
                'results': time_pair(reference, moving, truth, args.rounds),
            }
    finally:
        cv2.setNumThreads(prior_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    print('case                       old_ms   new_ms   speedup  new_result')
    for name, case in report['cases'].items():
        old = case['results']['old_periodic']
        new = case['results']['new_scene']
        ratio = case['results']['median_speedup_old_over_new']
        print(f'{name:26} {old["median_ms"]:7.2f} {new["median_ms"]:8.2f}'
              f' {ratio:8.2f}x  {new["reason"]}')
    print(f'Detailed results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
