"""Run explicit image/ROI manifest; output candidate evidence and measured times."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import html
import json
from pathlib import Path
import platform
import statistics
import time
import cv2
from operators.image_io import read_image, write_image
from projects.metal.inspect_surface import inspect,polygon_mask,annotate,Parameters


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--images',type=Path,default=Path(__file__).with_name('samples'))
    parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('sample_rois.json'))
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'output'/'metal')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    entries=json.loads(args.manifest.read_text(encoding="utf-8"))['images'];summaries=[];cards=[]
    for number,entry in enumerate(entries,1):
        path=args.images/entry['file'];image=read_image(str(path))
        if image is None:raise ValueError('cannot read '+str(path))
        if list(image.shape[1::-1])!=entry['size']:raise ValueError('ROI coordinates do not match image size')
        roi=polygon_mask(image.shape,entry['polygon'])
        reference=polygon_mask(image.shape,entry['reference_polygon']) if 'reference_polygon' in entry else None
        params=Parameters(**entry.get('parameters',{}))
        inspect(image,roi,params,reference)
        samples=[]
        for _ in range(args.repeats):
            start=time.perf_counter();report,maps=inspect(image,roi,params,reference)
            samples.append((time.perf_counter()-start)*1000)
        report.update(source=str(path),median_ms=statistics.median(samples),samples_ms=samples,
                      timing_scope='Full inspect API incl resizing, no decoding/annotation/file IO',
                      opencv=cv2.__version__,numpy=__import__('numpy').__version__,platform=platform.platform())
        stem=f'sample_{number}'
        for label,pixels in [('overlay',annotate(image,report,roi)),('roi',roi)]+list(maps.items()):
            if label=='scratch_response':pixels=(pixels.clip(0,.25)/.25*255).astype('uint8')
            if label=='color_response':pixels=(pixels.clip(0,30)/30*255).astype('uint8')
            if not write_image(str(args.output/f'{stem}_{label}.png'),pixels):raise IOError('image output failed')
        (args.output/f'{stem}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")
        summary={'file':entry['file'],'scratch_candidates':len(report['scratch_candidates']),
                 'discoloration_candidates':len(report['discoloration_candidates']),
                 'median_ms':report['median_ms'],'oxidation_confirmed':None}
        summaries.append(summary)
        cards.append(f'<section><h2>样本 {number}</h2><p>划痕候选 {summary["scratch_candidates"]}；色斑候选 {summary["discoloration_candidates"]}；{report["median_ms"]:.1f} ms</p>'
                     f'<a href="{stem}_overlay.png"><img src="{stem}_overlay.png"></a><p><a href="{stem}.json">坐标与参数 JSON</a> · '
                     f'<a href="{stem}_inspection_mask.png">实际检测范围</a> · <a href="{stem}_scratch_response.png">划痕响应</a></p></section>')
    thumbnails=[]
    for number in range(1,len(entries)+1):
        picture=read_image(str(args.output/f'sample_{number}_overlay.png'))
        thumb=cv2.resize(picture,(384,512))
        cv2.putText(thumb,f'Sample {number}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        thumbnails.append(thumb)
    if thumbnails:
        if len(thumbnails)%2: thumbnails.append(__import__('numpy').zeros_like(thumbnails[0]))
        sheet=cv2.vconcat([cv2.hconcat(thumbnails[i:i+2]) for i in range(0,len(thumbnails),2)])
        write_image(str(args.output/'contact_sheet.png'),sheet)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n', encoding="utf-8")
    (args.output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>金属表面检测实验</title><style>body{font:16px system-ui;background:#101827;color:#e2e8f0;max-width:1300px;margin:32px auto;padding:0 20px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}section{background:#1e293b;padding:18px;border-radius:12px}img{width:100%}a{color:#7dd3fc}</style>'
        '<h1>金属划痕 / 色斑候选</h1><p>红框 S：划痕候选；橙框 C：色斑候选；蓝线：人工定义的金属检测区域。不是缺陷真值标注。</p>'
        '<p>照片无法确认氧化成因；此页不输出合格/不合格。点击图片放大，查看误报与漏检。</p><main>'+''.join(cards)+'</main></html>', encoding="utf-8")
    print(json.dumps(summaries,indent=2));print('Preview:',args.output/'index.html')

if __name__=='__main__':main()
