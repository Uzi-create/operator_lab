"""Additional diagnostic operators on the four supplied photographs."""

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
from projects.metal.inspect_surface import polygon_mask,Parameters
from projects.metal.inspect_surface_v2 import inspect
from operators.defect_ops import local_defect_contrast,group_defect_fragments


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--images',type=Path,default=root/'samples')
    parser.add_argument('--manifest',type=Path,default=root/'sample_rois.json')
    parser.add_argument('--output',type=Path,default=root.parent/'output'/'metal_diagnostics')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cards=[];summary=[];panels=[]
    for i,item in enumerate(json.loads(args.manifest.read_text(encoding="utf-8"))['images'],1):
        image=read_image(str(args.images/item['file']))
        if image is None or list(image.shape[1::-1])!=item['size']:raise ValueError('missing image or incorrect ROI dimensions')
        roi=polygon_mask(image.shape,item['polygon'])
        params=Parameters(**item.get('parameters',{}))
        reference=polygon_mask(image.shape,item['reference_polygon']) if 'reference_polygon' in item else None
        report,maps=inspect(image,roi,params,reference)
        w,h=report['working_size'];work=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        def compute():
            contrast=local_defect_contrast(gray,maps['inspection_mask'])
            groups=group_defect_fragments(maps['scratch_mask'],max_gap=12)
            return contrast,groups
        compute();samples=[]
        for _ in range(5):
            start=time.perf_counter();contrast,groups=compute();samples.append((time.perf_counter()-start)*1000)
        overlay=work.copy()
        for number,group in enumerate(groups,1):
            x,y,bw,bh=group['bbox_xywh']
            color=(255,255,0) if group['fragment_count']>1 else (30,30,230)
            cv2.rectangle(overlay,(x,y),(x+bw-1,y+bh-1),color,2)
            cv2.putText(overlay,f'G{number}/{group["fragment_count"]}',(x,max(15,y-3)),0,.45,color,1)
        heat=cv2.applyColorMap((np.clip(contrast['score']/6,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        heat[~contrast['valid']]=0
        write_image(str(args.output/f'{i}_groups.png'),overlay)
        write_image(str(args.output/f'{i}_contrast.png'),heat)
        write_image(str(args.output/f'{i}_contrast_valid.png'),contrast['valid'].astype(np.uint8)*255)
        np.savez_compressed(args.output/f'{i}_contrast.npz',**contrast)
        result={'source':item['file'],'coordinate_size':[w,h],'coordinates':'working image pixels',
                'groups':groups,'group_count':len(groups),'input_fragment_count':sum(g['fragment_count'] for g in groups),
                'contrast_score_display_clip':6,'added_operator_median_ms':statistics.median(samples),
                'timing_samples_ms':samples,'timing_scope':'Only local contrast plus grouping; excludes v2 and IO',
                'is_ground_truth':False,'oxidation_confirmed':None}
        (args.output/f'{i}.json').write_text(json.dumps(result,indent=2)+'\n', encoding="utf-8")
        summary.append({k:result[k] for k in ['source','input_fragment_count','group_count','added_operator_median_ms']})
        pair=cv2.hconcat([cv2.resize(overlay,(300,400)),cv2.resize(heat,(300,400))])
        cv2.putText(pair,f'Sample {i}: groups / local contrast',(8,22),0,.5,(255,255,255),1);panels.append(pair)
        cards.append(f'<section><h2>样本 {i}</h2><p>{result["input_fragment_count"]} 个片段 → {len(groups)} 个复核组；新增计算 {result["added_operator_median_ms"]:.1f} ms</p>'
                     f'<div><img src="{i}_groups.png"><img src="{i}_contrast.png"></div><a href="{i}.json">分组坐标</a> · '
                     f'<a href="{i}_contrast_valid.png">有效对比度范围</a> · <a href="{i}_contrast.npz">原始数值</a></section>')
    if panels:write_image(str(args.output/'overview.png'),cv2.vconcat(panels))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n', encoding="utf-8")
    (args.output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>金属缺陷诊断算子</title><style>body{font:16px system-ui;background:#101827;color:#eee;max-width:1200px;margin:30px auto;padding:0 20px}'
        'section{background:#1e293b;padding:20px;margin:20px 0}section div{display:flex;gap:12px}img{width:calc(50% - 6px)}a{color:#7dd3fc}</style>'
        '<h1>局部对比度与片段分组</h1><p>左：复核组，青框表示多个片段组成的组；右：局部标准化对比度，黑色为不可评估区域。</p>'
        '<p>热图是异常响应，不是划痕概率或氧化证据；分组不填充像素，也不等于实际缺陷数量。</p>'+''.join(cards)+'</html>', encoding="utf-8")
    print(json.dumps(summary,indent=2));print(args.output/'index.html')

if __name__=='__main__':main()
