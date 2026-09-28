"""Small dependency-free API; buffers are array('f'), not Python lists."""
from array import array
import ctypes
import math
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_LIB = None
_MODES = {"threshold": 0, "mean": 1, "adaptive": 2, "mean_naive": 3,
          "guided": 4, "clahe": 5, "erode": 6, "dilate": 7, "open": 8, "close": 9}

def run(pixels, width, height, mode="mean", radius=3, parameter=0.08):
    """Return a new float32 array. Input: finite grayscale values in [0,1].

    Clipped borders. threshold uses strict > parameter; adaptive parameter is
    a positive intensity scale, with larger values causing stronger smoothing.
    guided is self-guided, epsilon=parameter**2. Morphology uses square windows.
    For clahe, radius means tile side in pixels (>=2), parameter is clip limit
    (>=1); prefer clahe() below for explicit parameter names.
    No copy of the input buffer is made. Do not mutate it during this call.
    """
    _validate_buffer(pixels, width, height, radius)
    if mode not in _MODES:
        raise ValueError("unknown mode: " + str(mode))
    parameter = ctypes.c_float(float(parameter)).value
    if not math.isfinite(parameter) or (mode in ("adaptive", "guided") and parameter <= 0):
        raise ValueError("invalid parameter")
    if mode == "clahe" and (radius < 2 or parameter < 1):
        raise ValueError("CLAHE needs tile size >= 2 and clip limit >= 1")
    library = _library()
    result = array("f", [0]) * len(pixels)
    source = (ctypes.c_float * len(pixels)).from_buffer(pixels)
    target = (ctypes.c_float * len(result)).from_buffer(result)
    code = library.run_operator(source, target, width, height, radius, _MODES[mode], parameter)
    if code == 1:
        raise ValueError("invalid native input: pixels must be finite and in [0,1]")
    if code:
        raise RuntimeError("native allocation or computation failed")
    return result


def guided(pixels, width, height, radius=7, scale=0.12):
    """Self-guided filter; regularization epsilon = scale squared."""
    return run(pixels, width, height, "guided", radius, scale)


def clahe(pixels, width, height, tile_size=64, clip_limit=2.0):
    """256-bin local contrast enhancement, bilinear tile-center interpolation."""
    return run(pixels, width, height, "clahe", tile_size, clip_limit)


def _validate_buffer(pixels, width, height, radius=0):
    for name, value in (("width", width), ("height", height), ("radius", radius)):
        if type(value) is not int or not 0 <= value <= 2147483647:
            raise ValueError(name + " must be a nonnegative int32")
    if width == 0 or height == 0:
        raise ValueError("dimensions must be positive")
    if not isinstance(pixels, array) or pixels.typecode != "f" or pixels.itemsize != 4:
        raise TypeError("pixels must be array('f') with 4-byte floats")
    if len(pixels) != width * height:
        raise ValueError("buffer length does not match dimensions")


def _library():
    global _LIB
    if _LIB is None:
        path = _ROOT / ({"darwin": "liboperators.dylib", "win32": "operators.dll"}.get(sys.platform, "liboperators.so"))
        if not path.exists():
            raise RuntimeError("Native library missing. Run python3 build.py first.")
        _LIB = ctypes.CDLL(str(path))
        ptr = ctypes.POINTER(ctypes.c_float)
        _LIB.run_operator.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_float]
        _LIB.run_operator.restype = ctypes.c_int
        _LIB.run_geometry.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_float, ctypes.c_float]
        _LIB.run_geometry.restype = ctypes.c_int
    return _LIB


def _geometry(pixels, width, height, mode, first=0.0, second=0.0):
    _validate_buffer(pixels, width, height)
    first, second = (ctypes.c_float(float(v)).value for v in (first, second))
    if not all(math.isfinite(v) for v in (first, second)):
        raise ValueError("parameters must be finite float32 values")
    if mode == 0 and not 0 < first <= second:
        raise ValueError("Canny needs 0 < low <= high")
    if mode == 3 and first <= 0:
        raise ValueError("feather width must be positive")
    result = array('f', [0]) * len(pixels)
    source = (ctypes.c_float * len(pixels)).from_buffer(pixels)
    target = (ctypes.c_float * len(result)).from_buffer(result)
    code = _library().run_geometry(source, target, width, height, mode, first, second)
    if code == 1:
        raise ValueError("invalid input: pixels must be finite and in [0,1]")
    if code:
        raise RuntimeError("native allocation or computation failed")
    return result


def canny(pixels, width, height, low=0.08, high=0.20):
    """5-tap binomial blur, Sobel/4, four-direction NMS, 8-connected hysteresis.

    Return a binary float32 edge map. The outer pixel border is always zero.
    Thresholds use L2 gradient magnitude after Sobel/4, not uint8 units.
    """
    return _geometry(pixels, width, height, 0, low, high)


def distance_transform(pixels, width, height):
    """Euclidean distance in pixels to nearest background center (input <=0.5).

    No implicit background outside the image. All-foreground returns +inf.
    Output is NOT normalized to [0,1].
    """
    return _geometry(pixels, width, height, 1)


def signed_distance(pixels, width, height):
    """Positive inside (>0.5), negative outside, to nearest opposite-class center.

    All-foreground returns +inf; all-background returns -inf. Pixel units.
    """
    return _geometry(pixels, width, height, 2)


def feather(pixels, width, height, feather_width=8.0):
    """Smoothstep of signed distance; feather_width is transition half-width.

    Full/empty masks stay full/empty. The input is binarized at >0.5.
    """
    return _geometry(pixels, width, height, 3, feather_width)
