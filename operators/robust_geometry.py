"""Deterministic, bounded RANSAC and least-squares refinements (no ROS dependency)."""
import math
import numpy as np


def points_array(points, dimensions, minimum):
    a=np.asarray(points,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=dimensions or len(a)<minimum or not np.isfinite(a).all():
        raise ValueError(f'Need >= {minimum} finite {dimensions}D points')
    return np.ascontiguousarray(a)


def ransac(points, sample_size, model, distances, threshold, iterations, seed, min_inliers):
    if not math.isfinite(threshold) or threshold<=0 or type(iterations) is not int or not 1<=iterations<=10000:
        raise ValueError('Positive finite threshold and 1..10000 iterations required')
    if type(min_inliers) is not int or not sample_size<=min_inliers<=len(points):
        raise ValueError('Invalid minimum inlier count')
    rng=np.random.default_rng(seed);best=None;best_count=0;best_error=math.inf;limit=iterations
    for iteration in range(iterations):
        if iteration>=limit:break
        candidate=model(points[rng.choice(len(points),sample_size,replace=False)])
        if candidate is None:continue
        d=distances(points,candidate);mask=d<=threshold;count=int(mask.sum())
        error=float(d[mask].sum()) if count else math.inf
        if count>best_count or (count==best_count and error<best_error):
            best,best_count,best_error=mask,count,error
            probability=(count/len(points))**sample_size
            if probability>=1:limit=iteration+1
            elif probability>0:limit=min(limit,max(iteration+1,math.ceil(math.log(.001)/math.log1p(-probability))))
    if best is None or best_count<min_inliers:
        raise ValueError('No nondegenerate model with sufficient support')
    fitted=None
    for _ in range(3):
        fitted=model(points[best])
        if fitted is None:raise ValueError('Degenerate inlier geometry')
        updated=distances(points,fitted)<=threshold
        if int(updated.sum())<min_inliers:raise ValueError('Refinement lost required support')
        if np.array_equal(best,updated):break
        best=updated
    # Mask and reported residuals must always describe the returned model.
    d=distances(points,fitted);best=d<=threshold
    return fitted,best,float(np.sqrt(np.mean(d[best]**2)))


def line_model(points):
    center=points.mean(axis=0);a=points-center
    _,s,v=np.linalg.svd(a,full_matrices=False)
    if not len(s) or s[0]<1e-12:return None
    direction=v[0]
    if direction[np.argmax(np.abs(direction))]<0:direction=-direction
    return center,direction


def line_distances(points,model):
    center,direction=model;d=points-center
    return np.abs(d[:,0]*direction[1]-d[:,1]*direction[0])


def circle_model(points):
    origin=points.mean(axis=0);a=points-origin
    matrix=np.column_stack((2*a,np.ones(len(a))))
    solution,_,rank,_=np.linalg.lstsq(matrix,np.sum(a*a,axis=1),rcond=None)
    if rank<3:return None
    radius2=solution[2]+np.dot(solution[:2],solution[:2])
    if radius2<=0 or not np.isfinite(radius2):return None
    return origin+solution[:2],math.sqrt(radius2)


def circle_distances(points,model):
    center,radius=model
    return np.abs(np.linalg.norm(points-center,axis=1)-radius)


def plane_model(points):
    center=points.mean(axis=0);a=points-center
    _,s,v=np.linalg.svd(a,full_matrices=False)
    if len(s)<2 or s[1]<=max(1e-12,s[0]*1e-10):return None
    normal=v[-1]
    if normal[np.argmax(np.abs(normal))]<0:normal=-normal
    return normal,-float(normal@center)


def plane_distances(points,model):
    normal,offset=model
    return np.abs(points@normal+offset)
