"""Edges -> distance field; binary mask -> signed field -> feathered alpha."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from array import array
from pathlib import Path
import argparse
import html
import json
import math
import platform
import statistics
import sys
import time
from projects.examples.demo import ROOT, png, scene
from operators import canny, distance_transform, signed_distance, feather


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=360)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument("--output", type=Path, default=ROOT/"output"/"geometry")
    args=parser.parse_args()
    w,h=args.width,args.height
    if min(w,h,args.repeats)<=0: parser.error('dimensions and repeats must be positive')
    clean,noisy=scene(w,h)
    mask=array('f',(float((x/w-.5)**2+(y/h-.5)**2<.28**2 or
                         (.12<x/w<.35 and .2<y/h<.8)) for y in range(h) for x in range(w)))
    outputs={'input':noisy,'mask':mask}
    timings={}
    def measure(name,fn):
        fn()
        samples=[]
        for _ in range(args.repeats):
            start=time.perf_counter(); result=fn()
            samples.append((time.perf_counter()-start)*1000)
        timings[name]={'median_ms':statistics.median(samples),'samples_ms':samples}
        outputs[name]=result
        return result
    edges=measure('canny',lambda:canny(noisy,w,h,.08,.20))
    inverse=array('f',(1-v for v in edges))
    measure('edge_distance',lambda:distance_transform(inverse,w,h))
    measure('distance',lambda:distance_transform(mask,w,h))
    measure('signed_distance',lambda:signed_distance(mask,w,h))
    measure('feather',lambda:feather(mask,w,h,12))
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    # Keep pixel-unit data: PNGs below are visualizations, not raw numerical results.
    for name in ['edge_distance','distance','signed_distance']:
        (out/(name+'.f32')).write_bytes(outputs[name].tobytes())
    displays=dict(outputs)
    displays['edge_distance']=array('f',(max(0,1-v/24) for v in outputs['edge_distance']))
    displays['distance']=array('f',(min(1,v/80) for v in outputs['distance']))
    displays['signed_distance']=array('f',(max(0,min(1,.5+v/80)) for v in outputs['signed_distance']))
    for name,pixels in displays.items(): png(out/(name+'.png'),pixels,w,h)
    pairs=[('input','canny','带噪输入 → Canny 边缘'),
           ('canny','edge_distance','边缘 → 距边缘的距离场（越近越亮，24 px 截断）'),
           ('mask','distance','二值掩膜 → 内部距离（80 px 显示为白）'),
           ('signed_distance','feather','有符号距离（灰色为零） → 12 px 半宽羽化')]
    combined=array('f'); sections=[]
    for left,right,label in pairs:
        for y in range(h):
            combined.extend(displays[left][y*w:(y+1)*w]);combined.extend(displays[right][y*w:(y+1)*w])
        sections.append('<section><h2>'+label+'</h2><div><img src="'+left+'.png"><img src="'+right+'.png"></div></section>')
    png(out/'comparison.png',combined,w*2,h*len(pairs))
    report={'platform':platform.platform(),'python':platform.python_version(),'width':w,'height':h,
            'repeats':args.repeats,'parameters':{'canny_low':.08,'canny_high':.20,'feather_half_width_px':12},
            'scope':'Python API including allocation and validation; one warm-up, no generation/IO',
            'raw_distance_files':{'type':'float32','byte_order':sys.byteorder,'layout':'row-major','units':'pixels'},
            'edge_pixels':int(sum(edges)),'timings':timings}
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n', encoding="utf-8")
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>边缘与距离场算子</title>'
        '<style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 20px;background:#101827;color:#e8eef7}'
        'section{padding:20px;background:#1e293b;border-radius:12px;margin:24px 0}section div{display:flex;gap:12px}'
        'img{width:calc(50% - 6px)}h2{font-size:18px}pre{white-space:pre-wrap}</style>'
        '<h1>边缘与距离场算子</h1><p>固定随机种子，左侧输入/中间结果，右侧处理结果。</p>'+''.join(sections)+
        '<p>距离 PNG 经过映射，仅供显示；同目录 .f32 文件保留像素单位的原始距离。</p>'
        '<h2>实测数据</h2><pre>'+html.escape(json.dumps(report,indent=2))+'</pre></html>', encoding="utf-8")
    print(json.dumps(report,indent=2));print('Preview:',out/'index.html')

if __name__=='__main__': main()
