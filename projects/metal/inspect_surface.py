"""Metal surface candidate inspection. NumPy + OpenCV; no learned model."""
from dataclasses import dataclass, asdict
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class Parameters:
    max_side: int = 960
    border_margin: int = 16
    scratch_floor: float = 0.065
    scratch_mad_factor: float = 5.0
    min_scratch_area: int = 8
    min_scratch_length: float = 8.0
    min_elongation: float = 2.0
    color_delta_floor: float = 12.0
    color_mad_factor: float = 4.0
    min_color_area: int = 70


def _robust_threshold(values, floor, factor):
    median=float(np.median(values))
    mad=float(np.median(np.abs(values-median)))
    return max(floor,median+factor*1.4826*mad)


def polygon_mask(shape, polygon):
    points=np.asarray(polygon,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
        raise ValueError('polygon needs at least three finite x,y points')
    h,w=shape[:2]
    if np.any(points<0) or np.any(points[:,0]>=w) or np.any(points[:,1]>=h):
        raise ValueError('polygon coordinates must be inside the image')
    mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[np.column_stack((np.rint(points[:,0]).clip(0,w-1),np.rint(points[:,1]).clip(0,h-1))).astype(np.int32)],255)
    return mask


def _components(binary, response, sx, sy, kind, params):
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    accepted=[];kept=np.zeros_like(binary)
    for label in range(1,count):
        x,y,w,h,area=(int(v) for v in stats[label])
        min_area=params.min_scratch_area if kind=='scratch_candidate' else params.min_color_area
        if area<min_area:continue
        local=(labels[y:y+h,x:x+w]==label)
        yy,xx=np.nonzero(local)
        points=np.column_stack((xx+x,yy+y)).astype(np.float32)
        (_, _),(rw,rh),angle=cv2.minAreaRect(points)
        length=max(rw,rh)+1;thickness=min(rw,rh)+1; elongation=length/thickness
        strength=float(response[y:y+h,x:x+w][local].max())
        if kind=='scratch_candidate':
            if length<params.min_scratch_length:continue
            if elongation<params.min_elongation and not (area>=24 and strength>=.14):continue
            if thickness>28:continue
        kept[y:y+h,x:x+w][local]=255
        original_w=round(binary.shape[1]/sx);original_h=round(binary.shape[0]/sy)
        left=max(0,math.floor(x/sx));top=max(0,math.floor(y/sy))
        right=min(original_w,math.ceil((x+w)/sx));bottom=min(original_h,math.ceil((y+h)/sy))
        accepted.append({'kind':kind,'bbox_xywh':[left,top,right-left,bottom-top],
                         'area_px_original_estimate':round(area/(sx*sy)),
                         'length_px_original_estimate':round(length/math.sqrt(sx*sy),2),
                         'elongation':round(elongation,3),'peak_response':round(strength,4),
                         'chemical_cause':'unconfirmed' if kind=='discoloration_candidate' else None})
    return sorted(accepted,key=lambda r:r['peak_response'],reverse=True),kept


def _inspect_core(image, roi_mask, params=Parameters(), reference_mask=None):
    """BGR uint8 input, explicit metal ROI. Boxes use original pixel coordinates.

    reference_mask may select a known-good metal patch in the SAME image.
    Without it, color baseline is estimated from the ROI and is exploratory.
    Returned response maps/masks use working resolution (report includes size).
    """
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('image must be uint8 HxWx3 BGR')
    if min(image.shape[:2])<3:raise ValueError('image is too small')
    for mask in [roi_mask]+([] if reference_mask is None else [reference_mask]):
        if not isinstance(mask,np.ndarray) or mask.shape!=image.shape[:2] or not np.isfinite(mask).all():
            raise ValueError('masks must be finite HxW arrays')
    for name in ('max_side','border_margin','min_scratch_area','min_color_area'):
        value=getattr(params,name)
        if type(value) is not int or value<1:raise ValueError(name+' must be a positive integer')
    for name,value in asdict(params).items():
        if not math.isfinite(value) or value<=0:raise ValueError(name+' must be finite and positive')
    if params.max_side<64 or params.border_margin<13:
        raise ValueError('max_side must be >=64; border_margin must be >=13')
    original_h,original_w=image.shape[:2]
    scale=min(1.,params.max_side/max(original_h,original_w))
    w,h=max(1,round(original_w*scale)),max(1,round(original_h*scale))
    sx,sy=w/original_w,h/original_h
    work=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
    roi=cv2.resize((roi_mask!=0).astype(np.uint8)*255,(w,h),interpolation=cv2.INTER_NEAREST)
    size=2*params.border_margin+1
    interior=cv2.erode(roi,np.ones((size,size),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    valid=interior & (gray>.10) & (gray<.97)
    if np.count_nonzero(valid)<100:raise ValueError('not enough inspectable metal pixels after border/exposure exclusion')
    smooth=cv2.GaussianBlur(gray,(0,0),.7)
    response=np.zeros_like(gray)
    for diameter in (7,13):
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter,diameter))
        white=cv2.morphologyEx(smooth,cv2.MORPH_TOPHAT,kernel)
        black=cv2.morphologyEx(smooth,cv2.MORPH_BLACKHAT,kernel)
        response=np.maximum(response,np.maximum(white,black))
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)/8
    jxx=cv2.GaussianBlur(gx*gx,(0,0),2);jyy=cv2.GaussianBlur(gy*gy,(0,0),2)
    jxy=cv2.GaussianBlur(gx*gy,(0,0),2)
    coherence=np.sqrt((jxx-jyy)**2+4*jxy*jxy)/(jxx+jyy+1e-8)
    threshold=_robust_threshold(response[valid],params.scratch_floor,params.scratch_mad_factor)
    scratch=((response>threshold) & valid & ((coherence>.30)|(response>2*threshold))).astype(np.uint8)*255
    scratch=cv2.morphologyEx(scratch,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    scratch[~valid]=0
    scratches,scratch_mask=_components(scratch,response,sx,sy,'scratch_candidate',params)
    # OpenCV float Lab: L in [0,100], a/b approximately [-127,127].
    lab=cv2.cvtColor(work.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    if reference_mask is None:
        baseline=valid
        baseline_source='robust_ROI_baseline_not_verified_good'
    else:
        baseline=(cv2.resize((reference_mask!=0).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0)&valid
        baseline_source='user_supplied_good_patch'
    if np.count_nonzero(baseline)<25:raise ValueError('reference region needs >=25 valid working pixels')
    reference=np.median(lab[:,:,1:][baseline],axis=0)
    ab=cv2.GaussianBlur(lab[:,:,1:],(0,0),2)
    delta=np.linalg.norm(ab-reference,axis=2)
    color_threshold=_robust_threshold(delta[baseline],params.color_delta_floor,params.color_mad_factor)
    color=((delta>color_threshold)&valid).astype(np.uint8)*255
    color=cv2.morphologyEx(color,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    color=cv2.morphologyEx(color,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8));color[~valid]=0
    spots,color_mask=_components(color,delta,sx,sy,'discoloration_candidate',params)
    report={'status':'review_candidates','parameters':asdict(params),'original_size':[original_w,original_h],
            'working_size':[w,h],'roi_source':'explicit_user_or_demo_polygon',
            'inspectable_fraction_of_roi':float(valid.sum()/max(1,(roi>0).sum())),
            'scratch_threshold':threshold,'color_ab_distance_threshold':color_threshold,
            'color_baseline_ab':reference.tolist(),'color_baseline_source':baseline_source,
            'scratch_candidates':scratches,'discoloration_candidates':spots,
            'oxidation_confirmed':None,'units':'pixels; no physical calibration',
            'limitations':['Candidates only; no OK/NG decision or chemical oxidation identification.',
                          'Excluded borders, shadows and glare may contain missed defects.',
                          'ROI baseline can absorb widespread discoloration; use a verified good reference.',
                          'Four views of similar damage do not establish independent accuracy.']}
    return report,{'scratch_response':response,'color_response':delta,'scratch_mask':scratch_mask,
                   'color_mask':color_mask,'inspection_mask':valid.astype(np.uint8)*255}, {'gray':gray,'smooth':smooth,'roi':roi}


def inspect(image, roi_mask, params=Parameters(), reference_mask=None):
    """Inspect uint8 BGR image inside ROI; return report and working-resolution maps.

    Public v1 result schema is unchanged. Internal preprocessing can be reused by v2.
    """
    report,maps,_=_inspect_core(image,roi_mask,params,reference_mask)
    return report,maps


def annotate(image,report,roi_mask):
    result=image.copy()
    contours,_=cv2.findContours((roi_mask!=0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result,contours,-1,(255,180,0),2)
    for key,color,prefix in [('scratch_candidates',(30,30,240),'S'),('discoloration_candidates',(0,170,255),'C')]:
        for number,item in enumerate(report[key],1):
            x,y,w,h=item['bbox_xywh'];cv2.rectangle(result,(x,y),(x+w,y+h),color,2)
            cv2.putText(result,f'{prefix}{number}',(x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    return result
