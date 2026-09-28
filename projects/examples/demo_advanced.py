"""Run advanced operators and create a self-contained local comparison page."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from array import array
from pathlib import Path
import argparse
import html
import json
import platform
import random
import statistics
import time
from projects.examples.demo import ROOT, png, scene
from operators import run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=360)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument("--output", type=Path, default=ROOT/"output"/"advanced")
    args=parser.parse_args()
    w,h=args.width,args.height
    if min(w,h,args.repeats)<=0: parser.error('dimensions and repeats must be positive')
    clean,noisy=scene(w,h)
    low=array('f',(0.35+v*0.16 for v in clean))
    mask=array('f',(float(v>0.5) for v in clean))
    damaged=array('f',mask)
    rng=random.Random(17)
    for i in range(w*h):
        if rng.random()<0.025: damaged[i]=1-damaged[i]
    inputs={'noisy':noisy,'low_contrast':low,'damaged_mask':damaged}
    cases=[('guided','noisy',7,0.12),('clahe','low_contrast',64,3.0),
           ('erode','damaged_mask',2,0.08),('dilate','damaged_mask',2,0.08),
           ('open','damaged_mask',2,0.08),('close','damaged_mask',2,0.08)]
    outputs=dict(inputs)
    measurements={}
    for mode,source,radius,parameter in cases:
        src=inputs[source]
        run(src,w,h,mode,radius,parameter)
        samples=[]
        for _ in range(args.repeats):
            start=time.perf_counter()
            result=run(src,w,h,mode,radius,parameter)
            samples.append((time.perf_counter()-start)*1000)
        outputs[mode]=result
        measurements[mode]={'radius_or_tile_size':radius,'parameter':parameter,
                            'median_ms':statistics.median(samples),'samples_ms':samples}
    # Practical composition: opening removes isolated bright dots, closing fills dark holes.
    outputs['open_close']=run(outputs['open'],w,h,'close',2)
    report={'platform':platform.platform(),'python':platform.python_version(),
            'width':w,'height':h,'repeats':args.repeats,
            'scope':'End-to-end Python API, output/scratch allocation and validation included; one warm-up; no image IO',
            'timings':measurements}
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    for name,pixels in outputs.items(): png(out/(name+'.png'),pixels,w,h)
    pairs=[('noisy','guided','带噪输入 → 自引导滤波'),
           ('low_contrast','clahe','低对比度输入 → CLAHE'),
           ('damaged_mask','open_close','破损掩膜 → 开运算后闭运算')]
    combined=array('f')
    sections=[]
    for left,right,title in pairs:
        for y in range(h):
            combined.extend(outputs[left][y*w:(y+1)*w])
            combined.extend(outputs[right][y*w:(y+1)*w])
        sections.append('<section><h2>'+title+'</h2><div><img src="'+left+'.png"><img src="'+right+'.png"></div></section>')
    png(out/'comparison.png',combined,w*2,h*3)
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n', encoding="utf-8")
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>复杂算子实验室</title>'
        '<style>body{background:#101827;color:#e8eef7;font:16px system-ui;margin:32px auto;padding:0 20px;max-width:1300px}'
        'section{background:#1e293b;padding:20px;border-radius:12px;margin:24px 0}'
        'section div{display:flex;gap:12px}img{width:calc(50% - 6px);object-fit:contain}'
        'pre{white-space:pre-wrap}h2{font-size:19px}</style><h1>复杂算子实验室</h1>'
        '<p>左侧输入，右侧输出。C++17 核心，Python 调用，固定随机种子。</p>'+''.join(sections)+
        '<p>形态学能移除孤立噪点，也会改变细小结构；CLAHE 可能放大噪声，按任务选择参数。</p>'
        '<h2>本机测量</h2><pre>'+html.escape(json.dumps(report,indent=2))+'</pre></html>', encoding="utf-8")
    print(json.dumps(report,indent=2))
    print('Preview:',out/'index.html')

if __name__=='__main__': main()
