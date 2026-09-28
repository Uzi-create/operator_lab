"""Same-process, alternating before/after timings and exact photo regression."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import projects.metal.inspect_surface_v2 as current
from projects.metal.inspect_surface import Parameters, polygon_mask
from operators.image_io import read_image

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    from projects.benchmarks.legacy_loader import execute_legacy
    execute_legacy(spec, module)
    return module


def measure(cases, repeats):
    for fn in cases.values():
        for _ in range(3):
            fn()
    samples = {name: [] for name in cases}
    items = list(cases.items())
    for i in range(repeats):
        for name, fn in (items if i % 2 == 0 else items[::-1]):
            start = time.perf_counter_ns()
            result = fn()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            samples[name].append(elapsed)
            del result
    return {name: {'median_ms': float(np.median(values)),
                   'p95_ms': float(np.percentile(values, 95)),
                   'samples_ms': values} for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=ROOT/'projects/output/ridge_optimization/before_20260925_031251')
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output/ridge_optimization/benchmark.json')
    args = parser.parse_args()
    old = load('ridge_baseline', args.before/'inspect_surface_v2.py')
    old_v1 = load('surface_baseline', args.before/'inspect_surface.py')
    old.inspect_v1 = old_v1.inspect
    library = ctypes.CDLL(str((args.before/'operators.dll').resolve()))
    ptr = ctypes.POINTER(ctypes.c_float)
    library.metal_ridge.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_int]
    library.metal_ridge.restype = ctypes.c_int
    old._NATIVE = library
    result = {'platform': platform.platform(), 'python': sys.version,
              'numpy': np.__version__, 'opencv': cv2.__version__,
              'opencv_threads': cv2.getNumThreads(),
              'build': 'Both DLLs built with --native; same host, alternating timings',
              'hashes': {}, 'ridge': [], 'photos': []}
    for path in [args.before/'operators.dll', ROOT/'operators/operators.dll',
                 args.before/'metal_ridge.hpp', ROOT/'operators/metal_ridge.hpp']:
        result['hashes'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    rng = np.random.default_rng(20260925)
    for h, w in [(480, 640), (960, 720), (1080, 1920)]:
        gray = rng.random((h, w), dtype=np.float32)
        out = np.empty_like(gray)
        expected = old.ridge_response(gray, 'native')
        np.testing.assert_array_equal(expected, current.ridge_response(gray, 'native'))
        np.testing.assert_array_equal(expected, current.ridge_response(gray, 'native', out=out))
        row = {'shape_hw': [h, w], 'exact_equal': True, 'timings': measure({
            'before_api': lambda: old.ridge_response(gray, 'native'),
            'after_api': lambda: current.ridge_response(gray, 'native'),
            'after_reuse': lambda: current.ridge_response(gray, 'native', out=out),
        }, 31)}
        result['ridge'].append(row)
        print(json.dumps(row), flush=True)
    for manifest in ['sample_rois.json', 'new_sample_rois.json']:
        for entry in json.loads((ROOT/'projects/metal'/manifest).read_text(encoding='utf-8'))['images']:
            image = read_image(ROOT/'projects/metal/samples'/entry['file'])
            roi = polygon_mask(image.shape, entry['polygon'])
            reference = polygon_mask(image.shape, entry['reference_polygon']) if 'reference_polygon' in entry else None
            params = Parameters(**entry.get('parameters', {}))
            before = lambda: old.inspect(image, roi, params, reference)
            after = lambda: current.inspect(image, roi, params, reference)
            br, bm = before()
            ar, am = after()
            assert br == ar, entry['file'] + ': report differs'
            assert bm.keys() == am.keys()
            for key in bm:
                np.testing.assert_array_equal(bm[key], am[key], err_msg=entry['file'] + ':' + key)
            row = {'file': entry['file'], 'reports_and_maps_exact_equal': True,
                   'timings': measure({'before_v2': before, 'after_v2': after}, 9)}
            result['photos'].append(row)
            print(entry['file'], {k: v['median_ms'] for k, v in row['timings'].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved', args.output)


if __name__ == '__main__':
    main()
