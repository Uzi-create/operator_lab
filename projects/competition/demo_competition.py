"""Competition-specific synthetic failure cases; no camera or robot required."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from array import array
import argparse
from pathlib import Path
from dataclasses import asdict
import html
import json
import math
import platform
import random
import statistics
import time
from operators.competition import grasp_depth,TargetEvidenceGate
from projects.examples.demo import ROOT,png


def legacy_depth_band(depths):
    """Existing valid-fraction + 20th-percentile band rules, NOT the full detector.

    Pure Python oracle for behavior comparison, not a speed benchmark vs NumPy.
    """
    valid=sorted(v for v in depths if math.isfinite(v) and .08<=v<=1.5)
    if len(valid)<max(12,math.ceil(len(depths)*.30)):return {'accepted':False,'depth':None}
    t=(len(valid)-1)*.2; i=int(t)
    center=valid[i]+(valid[min(i+1,len(valid)-1)]-valid[i])*(t-i)
    cluster=[v for v in valid if abs(v-center)<=.015]
    if len(cluster)<max(12,math.ceil(len(valid)*.65)):
        return {'accepted':False,'depth':None}
    median=statistics.median(cluster)
    mad=statistics.median(abs(v-median) for v in cluster)
    return {'accepted':mad<=.008,'depth':median if mad<=.008 else None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"output"/"competition")
    args=parser.parse_args()
    w,h=320,240
    mask=bytearray(w*h)
    indices=[y*w+x for y in range(90,150) for x in range(130,190)]
    for i in indices:mask[i]=1
    rng=random.Random(2026)
    outputs=[]; cases=[]
    # fx=360, width=60px, z=.3m => nominal metric width .05m.
    for name,counts,params in [
        ('clean_surface',(0,100,0),{}),
        ('near_outliers_25_percent',(25,70,5),{}),
        ('two_competing_surfaces',(50,50,0),{'min_cluster_fraction':.45}),
        ('depth_holes_80_percent',(0,20,80),{})]:
        depth=array('f',[.9])*(w*h)
        values=[]
        for j in range(len(indices)):
            percentile=j*100/len(indices)
            z=.18 if percentile<counts[0] else (.3 if percentile<sum(counts[:2]) else .9)
            if name=='depth_holes_80_percent' and percentile>=20:z=0
            if z: z+=rng.uniform(-.002,.002)
            values.append(z)
        rng.shuffle(values)
        for i,v in zip(indices,values):depth[i]=v
        def compute():
            return grasp_depth(depth,mask,w,h,fx=360,fy=360,cx=160,cy=120,**params)
        compute(); samples=[]
        for _ in range(7):
            start=time.perf_counter();result=compute();samples.append((time.perf_counter()-start)*1000)
        info=asdict(result);info.pop('inliers')
        info.update(name=name,legacy_depth_band=legacy_depth_band([depth[i] for i in indices]),
                    median_ms=statistics.median(samples),samples_ms=samples)
        cases.append(info);outputs.append((name,depth,result))
    gate=TargetEvidenceGate()
    good=outputs[1][2]
    timeline=[]
    for stamp,now,key,observation in [(0,0,'A',good),(.04,.04,'A',good),(.08,.08,'A',good),
                                     (.08,.09,'A',good),(.12,.12,'A',None),(.16,.16,'B',good),
                                     (.60,.60,'B',good),(.64,.64,'B',good),(.68,.68,'B',good)]:
        decision=gate.update(observation,stamp=stamp,now=now,target_key=key)
        timeline.append(dict(stamp=stamp,now=now,target=key,**asdict(decision)))
    report={'platform':platform.platform(),'resolution':[w,h],'mask_pixels':len(indices),
            'benchmark_scope':'7-call median after one warm-up; includes Python buffer copies and C++ allocations; no IO',
            'validation':'synthetic only; not a full-detector or real-robot evaluation',
            'cases':cases,'gate_timeline':timeline}
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    cards=[]
    for name,depth,result in outputs:
        png(out/(name+'_depth.png'),array('f',(min(1,v/1.0) for v in depth)),w,h)
        png(out/(name+'_support.png'),array('f',list(result.inliers)),w,h)
        cards.append('<section><h2>'+name+'</h2><p>'+result.reason+'</p><div><img src="'+name+'_depth.png"><img src="'+name+'_support.png"></div></section>')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n', encoding="utf-8")
    rows=''.join('<tr><td>'+c['name']+'</td><td>'+str(c['legacy_depth_band']['accepted'])+'</td><td>'+c['reason']+'</td><td>'+format(c['median_ms'],'.3f')+' ms</td></tr>' for c in cases)
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>物流抓取算子实验</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#101827;color:#e8eef7}'
        'section{background:#1e293b;padding:20px;margin:20px 0;border-radius:12px}section div{display:flex;gap:12px}'
        'img{width:calc(50% - 6px)}td,th{padding:12px;text-align:left;border-bottom:1px solid #475569}pre{white-space:pre-wrap}</style>'
        '<h1>物流抓取：主深度簇与时间确认</h1><p>合成场景验证；对照仅复现旧代码的深度带选择规则。</p>'
        '<table><tr><th>场景</th><th>旧深度带接受</th><th>新结果</th><th>耗时</th></tr>'+rows+'</table>'+
        '<p>每组左侧：深度图；右侧：接受的表面像素。黑色支持图表示拒绝。</p>'+''.join(cards)+
        '<h2>确认门时间序列</h2><pre>'+html.escape(json.dumps(timeline,indent=2))+'</pre></html>', encoding="utf-8")
    print(json.dumps(report,indent=2));print('Preview:',out/'index.html')

if __name__=='__main__':main()
