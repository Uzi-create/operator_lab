"""Reproducible native vs NumPy ridge API timings, alternating order."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from projects.metal.inspect_surface_v2 import ridge_response

if __name__=='__main__':
    image=np.random.default_rng(2).random((960,720),dtype=np.float32)
    samples={'native':[],'numpy':[]}
    expected=ridge_response(image,'numpy')
    np.testing.assert_allclose(ridge_response(image,'native'),expected,atol=1e-7)
    for iteration in range(9):
        for backend in (['native','numpy'] if iteration%2 else ['numpy','native']):
            start=time.perf_counter();ridge_response(image,backend)
            samples[backend].append((time.perf_counter()-start)*1000)
    report={'shape':[960,720],'platform':platform.platform(),
            'scope':'ridge API including validation and output allocation; fixed random input, one warm-up, alternating order',
            'median_ms':{k:statistics.median(v) for k,v in samples.items()},'samples_ms':samples}
    out=Path(__file__).resolve().parents[1]/'output'/'metal_v2';out.mkdir(parents=True,exist_ok=True)
    (out/'ridge_benchmark.json').write_text(json.dumps(report,indent=2)+'\n', encoding="utf-8")
    print(json.dumps(report,indent=2))
