"""Generate a reproducible scene, PNG comparison and honest timings."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from array import array
import argparse
import json
import math
from pathlib import Path
import platform
import random
import statistics
import struct
import time
import zlib
from operators import run

ROOT = Path(__file__).resolve().parents[1]

def png(path, pixels, width, height):
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("PNG dimensions must be positive integers")
    if len(pixels) != width * height:
        raise ValueError("PNG pixel count does not match dimensions")
    if not all(math.isfinite(v) for v in pixels):
        raise ValueError("PNG pixels must be finite")
    def chunk(kind, data):
        return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data)&0xffffffff)
    raw = b''.join(b'\0'+bytes(round(max(0, min(1, v))*255)
                    for v in pixels[y*width:(y+1)*width]) for y in range(height))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+
        chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 0, 0, 0, 0))+
        chunk(b'IDAT', zlib.compress(raw))+chunk(b'IEND', b''))

def scene(w, h):
    rng = random.Random(2026)
    clean = array('f')
    noisy = array('f')
    for y in range(h):
        for x in range(w):
            xx, yy = x/w, y/h
            value = 0.18 + 0.22*xx
            if 0.12 < xx < 0.43 and 0.18 < yy < 0.78:
                value = 0.8
            if (xx-0.71)**2+(yy-0.48)**2 < 0.15**2:
                value = 0.55 + 0.15*math.sin(xx*220)
            if 0.1 < xx < 0.9 and 0.84 < yy < 0.87:
                value = 0.9
            clean.append(value)
            noisy.append(max(0, min(1, value+rng.gauss(0, 0.07))))
    return clean, noisy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--radius', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT/"output"/"")
    args = parser.parse_args()
    if min(args.width, args.height, args.repeats) <= 0 or args.radius < 0:
        parser.error('dimensions/repeats must be positive; radius must be nonnegative')
    w, h = args.width, args.height
    clean, noisy = scene(w, h)
    outputs = {'clean': clean, 'noisy': noisy}
    timings = {}
    for mode in ['threshold', 'mean_naive', 'mean', 'adaptive']:
        parameter = 0.5 if mode == 'threshold' else 0.12
        run(noisy, w, h, mode, args.radius, parameter)  # warm-up, not timed
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = run(noisy, w, h, mode, args.radius, parameter)
            samples.append((time.perf_counter()-start)*1000)
        outputs[mode] = result
        timings[mode] = {'median_ms': statistics.median(samples), 'samples_ms': samples}
    error = max(abs(a-b) for a,b in zip(outputs['mean'], outputs['mean_naive']))
    if error > 2e-6:
        raise AssertionError('fast mean disagrees with naive mean')
    def psnr(pixels):
        mse = math.fsum((a-b)**2 for a,b in zip(pixels, clean))/len(clean)
        return -10*math.log10(mse) if mse else float('inf')
    report = {'platform': platform.platform(), 'machine': platform.machine(),
              'python': platform.python_version(), 'width': w, 'height': h,
              'radius': args.radius, 'repeats': args.repeats,
              'timing_scope': 'Python API including output allocation, ctypes, validation and native scratch allocation; excludes image generation and file IO',
              'timings': timings, 'mean_max_abs_error': error,
              'mean_speedup_vs_naive_cpp': timings['mean_naive']['median_ms']/timings['mean']['median_ms'],
              'psnr_db': {k: psnr(outputs[k]) for k in ['noisy', 'mean', 'adaptive']}}
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    for name, pixels in outputs.items():
        png(out/(name+'.png'), pixels, w, h)
    names = ['clean', 'noisy', 'mean', 'adaptive']
    combined = array('f')
    for y in range(h):
        for name in names:
            combined.extend(outputs[name][y*w:(y+1)*w])
    png(out/'comparison.png', combined, w*4, h)
    (out/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n', encoding="utf-8")
    cards = ''.join('<figure><img src="'+name+'.png"><figcaption>'+label+'</figcaption></figure>'
                    for name,label in [('clean','原始干净图'),('noisy','添加噪声'),
                                       ('mean','均值滤波'),('adaptive','自适应平滑'),('threshold','阈值化')])
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>算子实验室</title>'
        '<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:32px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}'
        'figure{margin:0;padding:16px;background:#1f2937;border-radius:12px}img{width:100%}'
        'figcaption{padding-top:12px}pre{white-space:pre-wrap}</style>'
        '<h1>算子实验室</h1><p>同一张合成图，观察降噪与细节保留。</p><main>'+cards+
        '</main><h2>本机实测</h2><pre>'+json.dumps(report, indent=2)+'</pre></html>', encoding="utf-8")
    print(json.dumps(report, indent=2))
    print('Preview:', out/'index.html')

if __name__ == '__main__':
    main()
