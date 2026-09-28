"""Paired v1/v2 real-photo benchmark; no accuracy claims without labels."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
import statistics
import time
import cv2
from operators.image_io import read_image, write_image
import numpy as np
from projects.metal.inspect_surface import inspect as v1,polygon_mask,annotate,Parameters
from projects.metal.inspect_surface_v2 import inspect as v2


def main():
    parser=argparse.ArgumentParser()
    root=Path(__file__).resolve().parent
    parser.add_argument('--images',type=Path,default=root/'samples')
    parser.add_argument('--manifest',type=Path,default=root/'sample_rois.json')
    parser.add_argument('--output',type=Path,default=root.parent/'output'/'metal_v2')
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    args.output.mkdir(parents=True,exist_ok=True);summaries=[];cards=[];thumbs=[]
    for number,item in enumerate(json.loads(args.manifest.read_text(encoding="utf-8"))['images'],1):
        image=read_image(str(args.images/item['file']))
        if image is None:raise ValueError('cannot read '+item['file'])
        if list(image.shape[1::-1])!=item['size']:raise ValueError('ROI size mismatch')
        roi=polygon_mask(image.shape,item['polygon']);params=Parameters(**item.get('parameters',{}))
        reference=polygon_mask(image.shape,item['reference_polygon']) if 'reference_polygon' in item else None
        functions={'v1':v1,'v2':v2};timings={'v1':[],'v2':[]};results={}
        for fn in functions.values():fn(image,roi,params,reference)
        for repetition in range(args.repeats):
            for version in (['v1','v2'] if repetition%2==0 else ['v2','v1']):
                start=time.perf_counter();report,maps=functions[version](image,roi,params,reference)
                timings[version].append((time.perf_counter()-start)*1000);results[version]=(report,maps)
        for version,(report,maps) in results.items():
            overlay=annotate(image,report,roi)
            for i,candidate in enumerate(report.get('border_review_candidates',[]),1):
                x,y,w,h=candidate['bbox_xywh'];cv2.rectangle(overlay,(x,y),(x+w,y+h),(255,0,255),2)
                cv2.putText(overlay,f'B{i}',(x,max(18,y-4)),0,.55,(255,0,255),2)
            write_image(str(args.output/f'{number}_{version}.png'),overlay)
            report['timing_samples_ms']=timings[version]
            (args.output/f'{number}_{version}.json').write_text(json.dumps(report,indent=2)+'\n', encoding="utf-8")
            if version=='v2':
                for key in ['rejected_v1_pixels','border_review_mask','ridge_response']:
                    pixels=maps[key]
                    if key=='ridge_response':pixels=(np.clip(pixels/.2,0,1)*255).astype(np.uint8)
                    write_image(str(args.output/f'{number}_{key}.png'),pixels)
        summary={'file':item['file'],'v1_count':len(results['v1'][0]['scratch_candidates']),
                 'v2_count':len(results['v2'][0]['scratch_candidates']),
                 'border_review_count':len(results['v2'][0]['border_review_candidates']),
                 'v1_median_ms':statistics.median(timings['v1']),'v2_median_ms':statistics.median(timings['v2'])}
        summaries.append(summary)
        pair=[]
        for version in ['v1','v2']:
            im=read_image(str(args.output/f'{number}_{version}.png'));im=cv2.resize(im,(300,400))
            cv2.putText(im,f'{number} {version}',(8,22),0,.65,(0,0,255),2);pair.append(im)
        thumbs.append(cv2.hconcat(pair))
        cards.append(f'<section><h2>样本 {number}：v1 {summary["v1_count"]} → v2 {summary["v2_count"]} 个内部候选；边缘待复核 {summary["border_review_count"]}</h2>'
                     f'<p>v1 {summary["v1_median_ms"]:.1f} ms / v2 {summary["v2_median_ms"]:.1f} ms</p>'
                     f'<div><img src="{number}_v1.png"><img src="{number}_v2.png"></div>'
                     f'<a href="{number}_rejected_v1_pixels.png">被移除的旧版像素</a> · <a href="{number}_v2.json">v2 坐标与参数</a></section>')
    if thumbs:write_image(str(args.output/'comparison.png'),cv2.vconcat(thumbs))
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n', encoding="utf-8")
    (args.output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>金属算子 v1/v2</title><style>body{background:#101827;color:#eee;font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px}'
        'section{background:#1e293b;padding:20px;margin:20px 0}section div{display:flex;gap:12px}img{width:calc(50% - 6px)}a{color:#7dd3fc}</style>'
        '<h1>金属候选检测 v1 / v2</h1><p>每组左 v1，右 v2；紫框 B 为靠边待复核区域，红框 S 为内部候选。候选减少不等于准确率提高。</p>'+''.join(cards)+'</html>', encoding="utf-8")
    print(json.dumps(summaries,indent=2));print(args.output/'index.html')

if __name__=='__main__':main()
