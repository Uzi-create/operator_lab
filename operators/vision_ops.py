"""CPU machine vision: normalized gray images, explicit masks, pixel coordinates.

OpenCV is the optimized backend for matching/connected components/interpolation.
These APIs are independent implementations, not HALCON-compatible replacements.
"""
import math
import cv2
import numpy as np
from operators.fast_stats import compute as local_stats
from operators.robust_geometry import (points_array,ransac,line_model,line_distances,
                             circle_model,circle_distances)


def _gray(image):
    if not isinstance(image,np.ndarray) or image.ndim!=2 or image.dtype not in (np.float32,np.float64):
        raise ValueError('Expected float32/float64 HxW grayscale')
    if not image.size or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
        raise ValueError('Image must be nonempty and finite in [0,1]')
    return np.ascontiguousarray(image,dtype=np.float32)


def _mask(mask):
    if not isinstance(mask,np.ndarray) or mask.ndim!=2 or not mask.size or mask.dtype.kind not in 'buif' or not np.isfinite(mask).all():
        raise ValueError('Mask must be a nonempty finite numeric HxW array')
    return np.ascontiguousarray(mask!=0,dtype=np.uint8)


def dynamic_threshold(gray,mask=None,*,radius=15,offset=.04,polarity='both',backend='auto'):
    """Compare to masked local mean; strict threshold, invalid centers excluded."""
    if not math.isfinite(offset) or offset<0 or polarity not in ('bright','dark','both'):
        raise ValueError('Invalid threshold or polarity')
    stats=local_stats(gray,mask,outer=radius,backend=backend)
    delta=stats['residual']
    selected=delta>offset if polarity=='bright' else (-delta>offset if polarity=='dark' else np.abs(delta)>offset)
    return selected & stats['valid']


def illumination_correct(gray,mask=None,*,radius=31,target=.5,backend='auto'):
    """Additive local background correction, not flat-field radiometric calibration."""
    if not math.isfinite(target) or not 0<=target<=1:raise ValueError('target must be [0,1]')
    stats=local_stats(gray,mask,outer=radius,backend=backend)
    result=np.clip(target+stats['residual'],0,1)
    result[~stats['valid']]=0
    return {'image':result,'background':stats['background'],'valid':stats['valid']}


def hysteresis_threshold(response,low,high,mask=None):
    """Keep 8-connected weak components containing at least one strong pixel."""
    if not isinstance(response,np.ndarray) or response.ndim!=2 or not response.size or response.dtype.kind not in 'uif' or not np.isfinite(response).all():
        raise ValueError('Response must be a finite real HxW array')
    if not math.isfinite(low) or not math.isfinite(high) or low>high:raise ValueError('Require finite low <= high')
    domain=np.ones(response.shape,bool) if mask is None else _mask(mask).astype(bool)
    if domain.shape!=response.shape:raise ValueError('Mask shape mismatch')
    weak=np.ascontiguousarray((response>=low)&domain,dtype=np.uint8)
    count,labels=cv2.connectedComponents(weak,connectivity=8)
    strong=(response>=high)&domain
    keep=np.bincount(labels[strong],minlength=count)>0;keep[0]=False
    return keep[labels]


def fill_holes(mask,max_area=None):
    """Fill enclosed 4-connected background holes; boundary-connected voids remain."""
    binary=_mask(mask)
    if max_area is not None and (type(max_area) is not int or max_area<1):raise ValueError('max_area must be positive')
    n,labels,stats,_=cv2.connectedComponentsWithStats(1-binary,connectivity=4)
    fill=np.ones(n,bool);fill[0]=False
    fill[np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))]=False
    if max_area is not None:fill &= stats[:,cv2.CC_STAT_AREA]<=max_area
    return (binary!=0)|fill[labels]


def select_regions(mask,min_area=1,max_area=None):
    """Area filtering of 8-connected regions, retaining the original pixels."""
    binary=_mask(mask)
    if type(min_area) is not int or min_area<1 or (max_area is not None and (type(max_area) is not int or max_area<min_area)):
        raise ValueError('Invalid area range')
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
    keep=stats[:,cv2.CC_STAT_AREA]>=min_area
    if max_area is not None:keep &= stats[:,cv2.CC_STAT_AREA]<=max_area
    keep[0]=False
    return keep[labels]


def region_features(mask,min_area=1,max_regions=10000):
    """Area, bounding box, centroid, PCA orientation and occupancy; no physical units."""
    binary=_mask(mask)
    if type(min_area) is not int or min_area<1 or type(max_regions) is not int or max_regions<1:raise ValueError('Invalid limits')
    n,labels,stats,centers=cv2.connectedComponentsWithStats(binary,connectivity=8)
    selected=np.flatnonzero(stats[1:,cv2.CC_STAT_AREA]>=min_area)+1
    if len(selected)>max_regions:raise ValueError('Too many regions; filter noise first')
    results=[]
    for label in selected:
        x,y,w,h,area=map(int,stats[label]);roi=(labels[y:y+h,x:x+w]==label).astype(np.uint8)
        moments=cv2.moments(roi,binaryImage=True)
        covariance=np.array([[moments['mu20'],moments['mu11']],[moments['mu11'],moments['mu02']]])/area
        eigen,axes=np.linalg.eigh(covariance);axis=axes[:,-1]
        orientation=None if eigen[-1]-eigen[0]<1e-12 else float(math.degrees(math.atan2(axis[1],axis[0]))%180)
        results.append({'label':int(label),'area':area,'bbox_xywh':[x,y,w,h],
                        'centroid_xy':centers[label].tolist(),'orientation_deg':orientation,
                        'elongation':float(math.sqrt((max(0,eigen[-1])+.25)/(max(0,eigen[0])+.25))),
                        'occupancy':area/(w*h)})
    return results


def match_template(image,template,*,min_score=.8,max_matches=10,min_distance=None,search_box=None):
    """Translation-only normalized correlation + center-distance NMS.

    No rotation/scale invariance. Score is correlation, not probability.
    search_box=(x,y,w,h); returned boxes/centers use original image coordinates.
    """
    image=_gray(image);template=_gray(template)
    if float(template.std())<1e-6:raise ValueError('Constant template has undefined normalized correlation')
    if not math.isfinite(min_score) or not -1<=min_score<=1 or type(max_matches) is not int or not 1<=max_matches<=10000:
        raise ValueError('Invalid score or match limit')
    x0=y0=0
    if search_box is not None:
        if len(search_box)!=4 or any(type(v) is not int for v in search_box):raise ValueError('Search box must be integer xywh')
        x0,y0,w,h=search_box
        if min(x0,y0)<0 or min(w,h)<1 or x0+w>image.shape[1] or y0+h>image.shape[0]:raise ValueError('Search box outside image')
        image=image[y0:y0+h,x0:x0+w]
    th,tw=template.shape
    if th>image.shape[0] or tw>image.shape[1]:raise ValueError('Template larger than search image')
    radius=min(tw,th)/2 if min_distance is None else min_distance
    if not math.isfinite(radius) or radius<1:raise ValueError('min_distance must be >=1 pixel')
    scores=cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
    original_scores=scores.copy()
    results=[]
    for _ in range(max_matches):
        _,peak,_,(x,y)=cv2.minMaxLoc(scores)
        if peak<min_score:break
        dx=dy=0.
        def refine(left,center,right):
            denominator=left-2*center+right
            return float(np.clip(.5*(left-right)/denominator,-.5,.5)) if denominator< -1e-9 else 0.
        if 0<x<scores.shape[1]-1:dx=refine(original_scores[y,x-1],original_scores[y,x],original_scores[y,x+1])
        if 0<y<scores.shape[0]-1:dy=refine(original_scores[y-1,x],original_scores[y,x],original_scores[y+1,x])
        results.append({'score':float(peak),'bbox_xywh':[x+x0,y+y0,tw,th],
                        'center_xy':[x+x0+(tw-1)/2+dx,y+y0+(th-1)/2+dy]})
        # Suppress only nearby centers, rather than destroying the whole score map.
        r=int(math.ceil(radius));left=max(0,x-r);right=min(scores.shape[1],x+r+1)
        top=max(0,y-r);bottom=min(scores.shape[0],y+r+1)
        yy,xx=np.ogrid[top:bottom,left:right]
        scores[top:bottom,left:right][(xx-x)**2+(yy-y)**2<=radius**2]=-2
    return results


def measure_edges(image,start,end,*,width=9,sigma=1.,threshold=.03,polarity='both',min_distance=3.):
    """Rectangular caliper: average transverse samples, smooth, detect gradient peaks.

    start/end are pixel-center (x,y), width is transverse sample count. Returns
    parabolic subpixel peaks and profile; precision is not guaranteed on real optics.
    Threshold units are normalized intensity per pixel along start -> end.
    """
    image=_gray(image);ends=points_array([start,end],2,2)
    if type(width) is not int or not 1<=width<=4096 or not math.isfinite(sigma) or not .1<=sigma<=100 or not math.isfinite(threshold) or threshold<=0 or not math.isfinite(min_distance) or min_distance<1:
        raise ValueError('Invalid caliper parameters')
    if polarity not in ('bright','dark','both'):raise ValueError('Invalid polarity')
    start,end=ends;direction=end-start;length=float(np.linalg.norm(direction))
    if length<4 or length>32765:raise ValueError('Caliper length must be 4..32765 pixels')
    direction/=length;normal=np.array([-direction[1],direction[0]])
    n=int(math.ceil(length))+1
    if n*width>16000000:raise ValueError('Caliper sampling budget exceeded')
    distances=np.linspace(0,length,n);cross=np.arange(width)-(width-1)/2
    coordinates=start+distances[None,:,None]*direction+cross[:,None,None]*normal
    if coordinates[:,:,0].min()<0 or coordinates[:,:,1].min()<0 or coordinates[:,:,0].max()>image.shape[1]-1 or coordinates[:,:,1].max()>image.shape[0]-1:
        raise ValueError('Complete caliper rectangle must be inside image')
    sampled=cv2.remap(image,coordinates[:,:,0].astype(np.float32),coordinates[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
    profile=sampled.mean(axis=0);step=length/(n-1)
    smooth=cv2.GaussianBlur(profile[None,:],(0,0),sigma/step,borderType=cv2.BORDER_REPLICATE)[0]
    derivative=np.gradient(smooth,step)
    response=derivative if polarity=='bright' else (-derivative if polarity=='dark' else np.abs(derivative))
    candidates=np.flatnonzero((response[1:-1]>response[:-2])&(response[1:-1]>=response[2:])&(response[1:-1]>=threshold))+1
    selected=[]
    for index in candidates[np.argsort(response[candidates])[::-1]]:
        if any(abs(index-j)*step<min_distance for j in selected):continue
        selected.append(int(index))
    edges=[]
    for index in sorted(selected):
        a,b,c=map(float,response[index-1:index+2]);den=a-2*b+c
        offset=float(np.clip(.5*(a-c)/den,-.5,.5)) if den< -1e-12 else 0.
        position=(index+offset)*step
        edges.append({'xy':(start+position*direction).tolist(),'distance':position,
                      'amplitude':float(derivative[index]),'polarity':'bright' if derivative[index]>0 else 'dark'})
    return {'edges':edges,'profile':profile,'gradient':derivative,'sample_distances':distances}


def fit_line(points,*,threshold=1.,max_iterations=256,seed=0,min_inliers=2):
    """RANSAC orthogonal line fit; returns centroid, unit direction, inliers, RMS."""
    points=points_array(points,2,2)
    (center,direction),mask,rms=ransac(points,2,line_model,line_distances,threshold,max_iterations,seed,min_inliers)
    return {'point':center,'direction':direction,'inliers':mask,'rms':rms}


def fit_circle(points,*,threshold=1.,max_iterations=256,seed=0,min_inliers=3):
    """RANSAC circle with centered algebraic refinement; short arcs are ill-conditioned."""
    points=points_array(points,2,3)
    (center,radius),mask,rms=ransac(points,3,circle_model,circle_distances,threshold,max_iterations,seed,min_inliers)
    angles=np.sort(np.arctan2(*(points[mask]-center)[:,::-1].T)%(2*np.pi))
    coverage=2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max()
    return {'center':center,'radius':radius,'inliers':mask,'rms':rms,'angular_coverage_deg':float(np.degrees(coverage))}
