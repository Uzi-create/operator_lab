"""Offline metric RGB-D geometry. No ROS, motion commands, implicit registration or GPU.

Optical frame: +X right, +Y down, +Z forward. Pixels are (u=x,v=y).
Inputs must already be undistorted; shared resolution does not imply registration.
"""
from dataclasses import dataclass
import math
import numpy as np
from operators.robust_geometry import points_array,ransac,plane_model,plane_distances


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if type(self.width) is not int or type(self.height) is not int or min(self.width,self.height)<1:
            raise ValueError('Camera dimensions must be positive integers')
        if not all(math.isfinite(v) for v in (self.fx,self.fy,self.cx,self.cy)) or min(self.fx,self.fy)<=0:
            raise ValueError('Finite intrinsics and positive focal lengths required')


def _depth(depth,camera,depth_scale,min_depth,max_depth):
    if not isinstance(camera,Intrinsics):raise TypeError('camera must be Intrinsics')
    if not isinstance(depth,np.ndarray) or depth.shape!=(camera.height,camera.width) or depth.dtype.kind not in 'uif':
        raise ValueError('Depth must be a real numeric array matching camera resolution')
    if not all(math.isfinite(v) for v in (depth_scale,min_depth,max_depth)) or depth_scale<=0 or not 0<=min_depth<max_depth:
        raise ValueError('Invalid depth scale or metric range')
    with np.errstate(over='ignore',invalid='ignore'):
        z=depth.astype(np.float64)*depth_scale
    valid=np.isfinite(z)&(z>min_depth)&(z<=max_depth)
    return z,valid


def depth_to_points(depth,camera,*,depth_scale=1.,min_depth=.01,max_depth=10.,mask=None,stride=1):
    """Backproject valid depths; set depth_scale=.001 for uint16 millimeters.

    Invalid/zero/nonfinite depths are excluded. stride samples the original grid,
    not resized intrinsics. points are Nx3 meters, pixels are original Nx2 (u,v).
    """
    z,valid=_depth(depth,camera,depth_scale,min_depth,max_depth)
    if type(stride) is not int or stride<1:raise ValueError('stride must be a positive integer')
    if mask is not None:
        if not isinstance(mask,np.ndarray) or mask.shape!=z.shape or mask.dtype.kind not in 'buif' or not np.isfinite(mask).all():
            raise ValueError('Invalid depth mask')
        valid &= mask!=0
    z=z[::stride,::stride];valid=valid[::stride,::stride]
    v,u=np.nonzero(valid);v=v*stride;u=u*stride
    zz=z[valid]
    points=np.column_stack(((u-camera.cx)*zz/camera.fx,(v-camera.cy)*zz/camera.fy,zz))
    return {'points':points,'pixels':np.column_stack((u,v)).astype(np.int32),'valid_sampled':valid}


def project_points(points,camera):
    """Pinhole projection, no occlusion test. Invalid/behind-camera pixels are NaN."""
    points=points_array(points,3,0)
    if not isinstance(camera,Intrinsics):raise TypeError('camera must be Intrinsics')
    positive=points[:,2]>0
    pixels=np.full((len(points),2),np.nan)
    pixels[positive,0]=points[positive,0]/points[positive,2]*camera.fx+camera.cx
    pixels[positive,1]=points[positive,1]/points[positive,2]*camera.fy+camera.cy
    # Snap only floating-point roundoff at exact image-center boundaries.
    for axis,upper in [(0,camera.width-1),(1,camera.height-1)]:
        pixels[np.abs(pixels[:,axis])<1e-9,axis]=0
        pixels[np.abs(pixels[:,axis]-upper)<1e-9,axis]=upper
    valid=positive & np.isfinite(pixels).all(axis=1) & (pixels[:,0]>=0)&(pixels[:,0]<=camera.width-1)&(pixels[:,1]>=0)&(pixels[:,1]<=camera.height-1)
    return {'pixels':pixels,'inside_image':valid,'in_front':positive}


def transform_points(points,transform):
    """Apply rigid 4x4 T_destination_from_source; row-array implementation of R*p+t."""
    points=points_array(points,3,0);transform=np.asarray(transform,dtype=np.float64)
    if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1],atol=1e-9,rtol=0):
        raise ValueError('Expected finite homogeneous rigid transform')
    rotation=transform[:3,:3]
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6,rtol=0) or not math.isclose(np.linalg.det(rotation),1.,abs_tol=1e-6):
        raise ValueError('Rotation must be orthonormal, determinant +1')
    return points@rotation.T+transform[:3,3]


def voxel_downsample(points,voxel_size,*,min_points=1):
    """Voxel centroids with counts and input->output mapping; rejected voxels map to -1."""
    points=points_array(points,3,0)
    if not math.isfinite(voxel_size) or voxel_size<=0 or type(min_points) is not int or min_points<1:
        raise ValueError('Positive voxel size/min_points required')
    scaled=points/voxel_size
    if not np.isfinite(scaled).all() or (np.abs(scaled)>=9e18).any():raise ValueError('Voxel index exceeds int64 range')
    keys=np.floor(scaled).astype(np.int64)
    unique,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    centroids=np.empty((len(unique),3),np.float64)
    for axis in range(3):centroids[:,axis]=np.bincount(inverse,weights=points[:,axis],minlength=len(unique))/counts
    keep=counts>=min_points;mapping=np.full(len(unique),-1,dtype=np.int64);mapping[keep]=np.arange(keep.sum())
    return {'points':centroids[keep],'counts':counts[keep],'voxel_indices':unique[keep],
            'source_to_voxel':mapping[inverse]}


def fit_plane(points,*,threshold=.005,max_iterations=256,seed=0,min_inliers=3):
    """RANSAC + orthogonal plane refinement. normal*p+offset=0; largest normal component positive.

    Threshold and RMS share point units (normally meters). This is a geometric
    dominant plane, not automatically the floor or a collision-free support plane.
    """
    points=points_array(points,3,3)
    (normal,offset),mask,rms=ransac(points,3,plane_model,plane_distances,threshold,max_iterations,seed,min_inliers)
    return {'normal':normal,'offset':offset,'inliers':mask,'rms':rms}


def plane_heights(points,normal,offset):
    """Signed perpendicular distances; caller controls normal direction."""
    points=points_array(points,3,0);normal=np.asarray(normal,dtype=np.float64)
    if normal.shape!=(3,) or not np.isfinite(normal).all() or not math.isfinite(offset) or np.linalg.norm(normal)<1e-12:
        raise ValueError('Invalid plane')
    return (points@normal+offset)/np.linalg.norm(normal)


def estimate_rigid_transform(source,target):
    """Kabsch alignment of known correspondences; does not discover matches or reject outliers."""
    source=points_array(source,3,3);target=points_array(target,3,3)
    if source.shape!=target.shape:raise ValueError('Correspondences must have equal shape')
    a=source-source.mean(axis=0);b=target-target.mean(axis=0)
    if np.linalg.matrix_rank(a)<2 or np.linalg.matrix_rank(b)<2:raise ValueError('Collinear/coincident correspondences are ambiguous')
    u,_,vt=np.linalg.svd(a.T@b);correction=np.eye(3)
    correction[-1,-1]=np.sign(np.linalg.det(vt.T@u.T))
    rotation=vt.T@correction@u.T
    transform=np.eye(4);transform[:3,:3]=rotation
    transform[:3,3]=target.mean(axis=0)-rotation@source.mean(axis=0)
    errors=np.linalg.norm(transform_points(source,transform)-target,axis=1)
    return {'transform':transform,'rms':float(np.sqrt(np.mean(errors**2))),'residuals':errors}


def depth_normals(depth,camera,*,depth_scale=1.,min_depth=.01,max_depth=10.,max_depth_jump=.02):
    """Central-difference organized normals facing the camera (nonpositive Z).

    Invalid depth, outer border and neighbor depth discontinuities are excluded.
    No smoothing or model-based normal estimation is implied.
    """
    z,valid=_depth(depth,camera,depth_scale,min_depth,max_depth)
    if not math.isfinite(max_depth_jump) or max_depth_jump<=0:raise ValueError('Positive depth discontinuity threshold required')
    z=np.where(valid,z,0)
    v,u=np.indices(z.shape)
    p=np.stack(((u-camera.cx)*z/camera.fx,(v-camera.cy)*z/camera.fy,z),axis=-1)
    normals=np.zeros(p.shape,np.float32);supported=np.zeros(z.shape,bool)
    if min(z.shape)<3:return {'normals':normals,'valid':supported}
    dx=p[1:-1,2:]-p[1:-1,:-2];dy=p[2:,1:-1]-p[:-2,1:-1]
    cross=np.cross(dx,dy);length=np.linalg.norm(cross,axis=2)
    good=valid[1:-1,1:-1].copy()
    for neighbor,ok in [(z[1:-1,2:],valid[1:-1,2:]),(z[1:-1,:-2],valid[1:-1,:-2]),
                        (z[2:,1:-1],valid[2:,1:-1]),(z[:-2,1:-1],valid[:-2,1:-1])]:
        good &= ok & (np.abs(neighbor-z[1:-1,1:-1])<=max_depth_jump)
    good &= length>1e-12
    cross/=np.maximum(length[:,:,None],1e-12)
    cross[cross[:,:,2]>0]*=-1;cross[~good]=0
    normals[1:-1,1:-1]=cross;supported[1:-1,1:-1]=good
    return {'normals':normals,'valid':supported}
