"""Independent masked contrast and fragment grouping operators, pixel units."""
import math
import cv2
import numpy as np


def _mask(mask,shape):
    if not isinstance(mask,np.ndarray) or mask.shape!=shape or mask.dtype.kind not in 'buif':
        raise ValueError('mask must be a real numeric array matching the image')
    if not np.isfinite(mask).all():raise ValueError('mask must be finite')
    return mask!=0


def _local_defect_contrast_reference(gray,mask,inner_radius=3,outer_radius=12,
                          noise_floor=.02,min_support=8,min_support_fraction=.5):
    """Compare each valid center to a square ring, excluding its inner square.

    Clipped image boundaries; no zeros/background outside mask in statistics.
    gray: finite float32/64 [0,1]. Signed residual and absolute standardized score
    are zero where support is insufficient; always consult valid/count maps.
    Support fraction uses the available geometric ring after image clipping.
    """
    if not isinstance(gray,np.ndarray) or gray.ndim!=2 or gray.dtype not in (np.float32,np.float64):
        raise ValueError('gray must be float32/float64 HxW')
    if not gray.size or not np.isfinite(gray).all() or gray.min()<0 or gray.max()>1:
        raise ValueError('gray must be nonempty, finite and in [0,1]')
    valid=_mask(mask,gray.shape)
    if any(type(v) is not int for v in (inner_radius,outer_radius,min_support)) or not 0<=inner_radius<outer_radius:
        raise ValueError('radii must be integers, 0 <= inner < outer')
    if outer_radius>4096 or min_support<1:raise ValueError('outer radius <=4096 and min_support >=1 required')
    if not math.isfinite(noise_floor) or not 1e-6<=noise_floor<=1 or not math.isfinite(min_support_fraction) or not 0<min_support_fraction<=1:
        raise ValueError('noise floor must be [1e-6,1]; support fraction must be (0,1]')
    def ring(values):
        def box(radius):
            return cv2.boxFilter(values,cv2.CV_64F,(2*radius+1,2*radius+1),normalize=False,borderType=cv2.BORDER_CONSTANT)
        return box(outer_radius)-box(inner_radius)
    weights=valid.astype(np.float64);values=gray.astype(np.float64)
    count=np.rint(ring(weights)).clip(0)
    capacity=np.rint(ring(np.ones(gray.shape,np.float64))).clip(0)
    total=ring(values*weights);squares=ring(values*values*weights)
    denominator=np.maximum(count,1)
    mean=total/denominator;variance=np.maximum(0,squares/denominator-mean*mean)
    supported=valid&(count>=min_support)&(count>=capacity*min_support_fraction)&(capacity>0)
    residual=np.where(supported,values-mean,0)
    score=np.abs(residual)/np.sqrt(variance+noise_floor*noise_floor)
    return {'residual':residual.astype(np.float32),'score':score.astype(np.float32),
            'background':np.where(supported,mean,0).astype(np.float32),
            'valid':supported,'support_count':count.astype(np.int32),
            'geometric_count':capacity.astype(np.int32)}


def local_defect_contrast(gray,mask,inner_radius=3,outer_radius=12,
                          noise_floor=.02,min_support=8,min_support_fraction=.5,
                          *,backend='auto'):
    """Same ring-score contract; auto uses fused C++ stats when built.

    backend='opencv' retains a portable fallback; native is strict.
    See fast_stats.compute for support semantics and thread-local allocation reuse.
    """
    if type(inner_radius) is not int or inner_radius < 0:
        raise ValueError('inner radius must be a nonnegative integer')
    if __package__:
        from .fast_stats import compute
    else:
        from operators.fast_stats import compute
    result=compute(gray,mask,inner=inner_radius,outer=outer_radius,
                   noise_floor=noise_floor,min_support=min_support,
                   min_support_fraction=min_support_fraction,backend=backend)
    result.pop('variance')
    return result


def group_defect_fragments(mask,max_gap=10.,max_angle_deg=25.,max_lateral=2.,
                           min_area=3,min_elongation=2.,max_components=512):
    """Group aligned end-to-end components, preserving all original pixels.

    Returns diagnostic groups, NOT a new filled mask or a count of true defects.
    Uses PCA axis and nearest extreme endpoints. No merging of parallel adjacent
    lines or perpendicular crossings unless they were already connected in input.
    Transitive groups may span multiple gaps. O(K^2), bounded by max_components.
    """
    if not isinstance(mask,np.ndarray) or mask.ndim!=2 or not mask.size:
        raise ValueError('mask must be a nonempty HxW array')
    binary=_mask(mask,mask.shape).astype(np.uint8)
    for name,value in [('max_gap',max_gap),('max_angle_deg',max_angle_deg),('max_lateral',max_lateral),('min_elongation',min_elongation)]:
        if not math.isfinite(value) or value<=0:raise ValueError(name+' must be finite and positive')
    if max_angle_deg>90 or min_elongation<1:raise ValueError('angle <=90 and elongation >=1 required')
    if type(min_area) is not int or min_area<1 or type(max_components) is not int or max_components<1:
        raise ValueError('area/component limits must be positive integers')
    count,labels,stats,centers=cv2.connectedComponentsWithStats(binary,8)
    qualified=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=min_area]
    if len(qualified)>max_components:raise ValueError('too many components; filter noise or raise max_components explicitly')
    components=[]
    for label in qualified:
        x,y,w,h,area=map(int,stats[label]);ys,xs=np.nonzero(labels[y:y+h,x:x+w]==label)
        points=np.column_stack((xs+x,ys+y)).astype(np.float64)
        centered=points-points.mean(axis=0)
        covariance=centered.T@centered/max(1,len(points))
        eigenvalues,eigenvectors=np.linalg.eigh(covariance)
        axis=eigenvectors[:,-1];projection=centered@axis
        ratio=math.sqrt((eigenvalues[-1]+.25)/(max(0,eigenvalues[0])+.25))
        components.append({'label':label,'bbox':[x,y,w,h],'area':area,'axis':axis,
                           'ends':points[[np.argmin(projection),np.argmax(projection)]],
                           'elongation':ratio})
    parents=list(range(len(components)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    cosine=math.cos(math.radians(max_angle_deg))
    links=[]
    for i,a in enumerate(components):
        if a['elongation']<min_elongation:continue
        for j in range(i+1,len(components)):
            b=components[j]
            if b['elongation']<min_elongation or abs(float(a['axis']@b['axis']))<cosine:continue
            ax,ay,aw,ah=a['bbox'];bx,by,bw,bh=b['bbox']
            if max(bx-(ax+aw-1),ax-(bx+bw-1),by-(ay+ah-1),ay-(by+bh-1))>max_gap:continue
            delta=b['ends'][None,:,:]-a['ends'][:,None,:]
            distances=np.linalg.norm(delta,axis=2);p,q=np.unravel_index(np.argmin(distances),distances.shape)
            displacement=delta[p,q];distance=float(distances[p,q])
            if distance>max_gap or distance==0:continue
            # Connector must align with BOTH components, not just their axes with each other.
            if any(abs(float(displacement@c['axis']))/distance<cosine for c in (a,b)):continue
            if any(abs(float(displacement[0]*c['axis'][1]-displacement[1]*c['axis'][0]))>max_lateral for c in (a,b)):continue
            parents[find(j)]=find(i);links.append([a['label'],b['label']])
    groups={}
    for i,c in enumerate(components):groups.setdefault(find(i),[]).append(c)
    result=[]
    for members in groups.values():
        x=min(c['bbox'][0] for c in members);y=min(c['bbox'][1] for c in members)
        right=max(c['bbox'][0]+c['bbox'][2] for c in members);bottom=max(c['bbox'][1]+c['bbox'][3] for c in members)
        member_labels=[c['label'] for c in members]
        result.append({'component_labels':member_labels,'bbox_xywh':[x,y,right-x,bottom-y],
                       'area_px':sum(c['area'] for c in members),
                       'fragment_count':len(members),'links':[p for p in links if p[0] in member_labels],
                       'status':'review_group'})
    return result
