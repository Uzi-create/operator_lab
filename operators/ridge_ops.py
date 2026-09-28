"""Reusable ridge response, independent of any inspection application."""
import ctypes
from pathlib import Path
import sys
import numpy as np

def ridge_response_numpy(gray, *, out=None):
    """Symmetric bright/dark ridge contrast at 4 normals and 3 half-widths.

    Each side must differ from the center in the SAME sign. A monotone ramp or
    ideal step is rejected, unlike a generic edge magnitude. Units: [0,1].
    Samples beyond the image replicate its edge; inspection excludes borders.
    """
    if not isinstance(gray,np.ndarray) or gray.ndim!=2 or gray.dtype!=np.float32:
        raise ValueError('gray must be a float32 HxW array')
    if min(gray.shape)<1 or not np.isfinite(gray).all() or gray.min()<0 or gray.max()>1:
        raise ValueError('gray values must be finite and in [0,1]')
    _validate_output(gray,out)
    h,w=gray.shape;pad=np.pad(gray,3,mode='edge')
    if out is None:out=np.zeros_like(gray)
    else:out.fill(0)
    for dx,dy in [(1,0),(0,1),(1,1),(1,-1)]:
        for radius in (1,2,3):
            xx,yy=dx*radius,dy*radius
            first=pad[3+yy:3+yy+h,3+xx:3+xx+w]
            second=pad[3-yy:3-yy+h,3-xx:3-xx+w]
            a=gray-first;b=gray-second
            contrast=np.where(a*b>0,np.minimum(np.abs(a),np.abs(b)),0)
            np.maximum(out,contrast,out=out)
    return out


def _validate_output(gray,out):
    if out is None:return
    if not isinstance(out,np.ndarray) or out.shape!=gray.shape or out.dtype!=np.float32:
        raise ValueError('out must be float32 with the input shape')
    if not out.flags.c_contiguous or not out.flags.aligned or not out.flags.writeable:
        raise ValueError('out must be writable, aligned and C-contiguous')
    if np.shares_memory(gray,out):
        raise ValueError('input and output must not overlap')


_NATIVE = None

def ridge_response(gray, backend='auto', *, out=None):
    """Native or NumPy ridge with optional reusable, non-overlapping output.

    Native values are checked once in C++, before any output write. out must be
    aligned writable C-contiguous float32, same shape, and disjoint from input.
    Input views are copied only if needed for alignment/contiguity.
    """
    global _NATIVE
    if backend not in ('auto','native','numpy'):raise ValueError('unknown ridge backend')
    if not isinstance(gray,np.ndarray) or gray.ndim!=2 or gray.dtype!=np.float32:
        raise ValueError('gray must be float32 HxW')
    if min(gray.shape)<1 or max(gray.shape)>2147483647:
        raise ValueError('dimensions must be positive int32')
    _validate_output(gray,out)
    if backend=='numpy':return ridge_response_numpy(gray,out=out)
    path=Path(__file__).resolve().parent/({'darwin':'liboperators.dylib','win32':'operators.dll'}.get(sys.platform,'liboperators.so'))
    if _NATIVE is None and path.exists():
        try:
            library=ctypes.CDLL(str(path))
        except OSError as error:
            if backend=='native':raise RuntimeError('Native library cannot load; rebuild it for this machine') from error
            return ridge_response_numpy(gray,out=out)
        if hasattr(library,'metal_ridge'):
            ptr=ctypes.POINTER(ctypes.c_float)
            library.metal_ridge.argtypes=[ptr,ptr,ctypes.c_int,ctypes.c_int]
            library.metal_ridge.restype=ctypes.c_int
            _NATIVE=library
    if _NATIVE is None:
        if backend=='native':raise RuntimeError('Run python3 operator_lab/build.py first')
        return ridge_response_numpy(gray,out=out)
    source=np.require(gray,dtype=np.float32,requirements=['C','A'])
    if out is None:out=np.empty_like(source)
    ptr=ctypes.POINTER(ctypes.c_float)
    code=_NATIVE.metal_ridge(source.ctypes.data_as(ptr),out.ctypes.data_as(ptr),source.shape[1],source.shape[0])
    if code:raise ValueError('native ridge rejected input')
    return out
