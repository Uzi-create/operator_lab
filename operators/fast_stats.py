"""Masked box/ring statistics, native fused integral tables or OpenCV fallback.

One workspace per Python thread reuses C++ integral allocations; returned arrays
are independent. Inputs are never mutated. All image coordinates are pixels.
"""
import ctypes
import math
from pathlib import Path
import sys
import threading
import cv2
import numpy as np

_THREAD = threading.local()


class _Workspace:
    def __init__(self):
        path = Path(__file__).resolve().parent / {
            'win32': 'operators.dll', 'darwin': 'liboperators.dylib'}.get(sys.platform, 'liboperators.so')
        self.library = ctypes.CDLL(str(path))
        d = ctypes.POINTER(ctypes.c_double)
        f = ctypes.POINTER(ctypes.c_float)
        b = ctypes.POINTER(ctypes.c_uint8)
        i = ctypes.POINTER(ctypes.c_int32)
        self.library.stats_create.restype = ctypes.c_void_p
        self.library.stats_destroy.argtypes = [ctypes.c_void_p]
        self.library.stats_destroy.restype = None
        self.library.masked_stats.argtypes = [ctypes.c_void_p,d,b,ctypes.c_int,ctypes.c_int,
            ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_double,f,f,f,f,b,i,i]
        self.library.masked_stats.restype = ctypes.c_int
        self.handle = self.library.stats_create()
        if not self.handle:
            raise MemoryError('Cannot allocate native statistics workspace')

    def __del__(self):
        if getattr(self, 'handle', None):
            self.library.stats_destroy(self.handle)
            self.handle = None


def compute(gray, mask=None, *, inner=-1, outer=7, noise_floor=.02,
            min_support=1, min_support_fraction=.5, backend='auto'):
    """Return mean/background, variance, signed residual, score and support maps.

    inner=-1 includes the complete box. Otherwise exclude the inner square.
    Statistics ignore masked pixels; fraction uses the image-clipped geometry.
    Invalid centers have zero floating outputs, but retain support counts.
    """
    if not isinstance(gray,np.ndarray) or gray.ndim!=2 or gray.dtype not in (np.float32,np.float64):
        raise ValueError('gray must be float32/float64 HxW')
    if not gray.size or gray.size>2147483647 or not np.isfinite(gray).all() or gray.min()<0 or gray.max()>1:
        raise ValueError('gray must contain finite [0,1] values and fit int32 indexing')
    if mask is None:
        valid = np.ones(gray.shape,np.uint8)
    else:
        if not isinstance(mask,np.ndarray) or mask.shape!=gray.shape or mask.dtype.kind not in 'buif' or not np.isfinite(mask).all():
            raise ValueError('mask must be a finite real HxW array')
        valid = np.ascontiguousarray(mask!=0,dtype=np.uint8)
    if any(type(v) is not int for v in (inner,outer,min_support)) or not -1<=inner<outer<=4096 or min_support<1 or min_support>2147483647:
        raise ValueError('Require -1 <= inner < outer <=4096 and positive int32 support')
    if not math.isfinite(noise_floor) or not 1e-6<=noise_floor<=1 or not math.isfinite(min_support_fraction) or not 0<min_support_fraction<=1:
        raise ValueError('Invalid noise floor or support fraction')
    if backend not in ('auto','native','opencv'):
        raise ValueError('backend must be auto, native or opencv')
    workspace = None
    if backend!='opencv':
        try:
            workspace=getattr(_THREAD,'workspace',None)
            if workspace is None:
                workspace=_Workspace();_THREAD.workspace=workspace
        except (OSError,AttributeError) as error:
            if backend=='native':
                raise RuntimeError('Native statistics unavailable; run python build.py') from error
    values=np.ascontiguousarray(gray,dtype=np.float64)
    if workspace is not None:
        result={key:np.empty(gray.shape,np.float32) for key in ('background','variance','residual','score')}
        result.update(valid=np.empty(gray.shape,np.uint8),support_count=np.empty(gray.shape,np.int32),
                      geometric_count=np.empty(gray.shape,np.int32))
        def ptr(a,kind):return a.ctypes.data_as(ctypes.POINTER(kind))
        code=workspace.library.masked_stats(workspace.handle,ptr(values,ctypes.c_double),ptr(valid,ctypes.c_uint8),
            gray.shape[1],gray.shape[0],inner,outer,noise_floor,min_support,min_support_fraction,
            *(ptr(result[k],ctypes.c_float) for k in ('background','variance','residual','score')),
            ptr(result['valid'],ctypes.c_uint8),ptr(result['support_count'],ctypes.c_int32),ptr(result['geometric_count'],ctypes.c_int32))
        if code:raise RuntimeError(f'Native statistics failed ({code})')
        result['valid']=result['valid'].view(np.bool_)
        return result
    def ring(a):
        big=cv2.boxFilter(a,cv2.CV_64F,(2*outer+1,)*2,normalize=False,borderType=cv2.BORDER_CONSTANT)
        return big if inner<0 else big-cv2.boxFilter(a,cv2.CV_64F,(2*inner+1,)*2,normalize=False,borderType=cv2.BORDER_CONSTANT)
    weights=valid.astype(np.float64)
    count=np.rint(ring(weights)).clip(0);capacity=np.rint(ring(np.ones(gray.shape,np.float64))).clip(0)
    mean=ring(values*weights)/np.maximum(count,1)
    variance=np.maximum(0,ring(values*values*weights)/np.maximum(count,1)-mean*mean)
    supported=(valid!=0)&(count>=min_support)&(count>=capacity*min_support_fraction)&(capacity>0)
    residual=np.where(supported,values-mean,0)
    return {'background':np.where(supported,mean,0).astype(np.float32),
            'variance':np.where(supported,variance,0).astype(np.float32),
            'residual':residual.astype(np.float32),
            'score':(np.abs(residual)/np.sqrt(variance+noise_floor**2)).astype(np.float32),
            'valid':supported,'support_count':count.astype(np.int32),'geometric_count':capacity.astype(np.int32)}
