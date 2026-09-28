"""Offline competition perception operators. No ROS publishers or motion commands."""
from array import array
from dataclasses import dataclass
import ctypes
import math
from typing import Optional, Tuple
from operators.core import _library, _validate_buffer


@dataclass(frozen=True)
class DepthEstimate:
    accepted: bool
    reason: str
    area: int
    valid_count: int
    cluster_count: int
    valid_fraction: float
    cluster_fraction: float
    runner_ratio: float
    depth_m: Optional[float]
    mad_m: Optional[float]
    pixel: Optional[Tuple[float, float]]
    point_m: Optional[Tuple[float, float, float]]
    size_m: Optional[Tuple[float, float]]
    inliers: bytes


def grasp_depth(depth, mask, width, height, *, fx, fy, cx, cy,
                min_depth=0.08, max_depth=1.5, band_width=0.03,
                min_samples=12, min_valid_fraction=0.30,
                min_cluster_fraction=0.65, max_mad=0.008, ambiguity_ratio=0.8):
    """One registered color component -> dominant surface and optical-frame point.

    depth: array('f'), METERS. 0/NaN/inf/out-of-range depths are missing samples.
    mask: bytes/bytearray, one byte per pixel (nonzero means selected).
    Same resolution alone does not imply RGB-depth registration.
    band_width is FULL depth span; never sample outside the supplied component.
    """
    _validate_buffer(depth,width,height)
    if not isinstance(mask,(bytes,bytearray)) or len(mask)!=width*height:
        raise ValueError('mask must be bytes/bytearray with width*height entries')
    if type(min_samples) is not int or not 1<=min_samples<=2147483647:
        raise ValueError('min_samples must be a positive int32')
    config=(ctypes.c_double*12)(min_depth,max_depth,band_width,min_samples,
        min_valid_fraction,min_cluster_fraction,max_mad,ambiguity_ratio,fx,fy,cx,cy)
    if not all(math.isfinite(v) for v in config): raise ValueError('parameters must be finite')
    fn=_library().estimate_grasp_depth
    fn.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_ubyte),
                 ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double),
                 ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_ubyte)]
    fn.restype=ctypes.c_int
    src=(ctypes.c_float*len(depth)).from_buffer(depth)
    selection=(ctypes.c_ubyte*len(mask)).from_buffer_copy(mask)
    output=(ctypes.c_double*14)(); support=(ctypes.c_ubyte*len(mask))()
    status=fn(src,selection,width,height,config,output,support)
    if status==1: raise ValueError('invalid depth estimator configuration')
    if status==2: raise RuntimeError('native allocation failed')
    reasons={0:'accepted',3:'insufficient_support',4:'ambiguous_surfaces',5:'depth_spread'}
    accepted=status==0
    return DepthEstimate(accepted,reasons[status],int(output[0]),int(output[1]),int(output[2]),
        output[3],output[4],output[5],output[6] if accepted else None,
        output[7] if accepted else None,(output[8],output[9]) if accepted else None,
        (output[10],output[11],output[6]) if accepted else None,
        (output[12],output[13]) if accepted else None,bytes(support))


@dataclass(frozen=True)
class GateResult:
    state: str
    ready: bool
    point_m: Optional[Tuple[float, float, float]]
    fresh: bool
    confirmations: int


class TargetEvidenceGate:
    """Time-based confirmation of ONE associated target, not a multi-object tracker.

    All stamp/now values must use the same monotonic seconds clock. Repeated or
    decreasing stamps are rejected. reset() is required after a clock/mission epoch change.
    Coasting never authorizes an action, and does not count as a confirmation.
    """
    def __init__(self, confirm_frames=3, min_duration=0.06, max_age=0.15,
                 max_gap=0.15, hold_timeout=0.4, max_jump=0.025):
        if type(confirm_frames) is not int or confirm_frames<1:
            raise ValueError('confirm_frames must be a positive integer')
        values=(min_duration,max_age,max_gap,hold_timeout,max_jump)
        if not all(math.isfinite(v) for v in values) or min_duration<0 or min(values[1:])<=0:
            raise ValueError('invalid gate configuration')
        if hold_timeout<max_gap: raise ValueError('hold_timeout must be >= max_gap')
        self.confirm_frames=confirm_frames;self.min_duration=min_duration
        self.max_age=max_age;self.max_gap=max_gap;self.hold_timeout=hold_timeout;self.max_jump=max_jump
        self.reset()

    def reset(self):
        self.last_stamp=None;self.last_now=None;self.last_match=None
        self.point=None;self.count=0;self.started=None;self.key=None

    def _miss(self,now,reason):
        self.count=0;self.started=None
        if self.last_match is not None and now-self.last_match<=self.hold_timeout:
            return GateResult(reason,False,self.point,False,0)
        self.point=None;self.key=None;self.last_match=None
        return GateResult('lost',False,None,False,0)

    def update(self,estimate,*,stamp,now,target_key):
        if not isinstance(target_key,str) or not target_key:
            raise ValueError('target_key must be a stable nonempty association key')
        if not math.isfinite(stamp) or not math.isfinite(now): raise ValueError('timestamps must be finite')
        if self.last_now is not None and now<self.last_now:
            self.reset()
            raise ValueError('clock moved backwards; state reset')
        self.last_now=now
        if stamp>now: return self._miss(now,'future_frame')
        if self.last_stamp is not None and stamp<=self.last_stamp:
            return self._miss(now,'out_of_order')
        self.last_stamp=stamp
        if now-stamp>self.max_age: return self._miss(now,'stale_frame')
        if estimate is None or not estimate.accepted: return self._miss(now,'no_observation')
        point=estimate.point_m
        if point is None or len(point)!=3 or not all(math.isfinite(v) for v in point):
            raise ValueError('accepted estimate needs a finite 3D point')
        # Expire using observation time, not callback count.
        if self.last_match is not None and now-self.last_match>self.hold_timeout:
            self.point=None;self.key=None;self.count=0;self.started=None;self.last_match=None
        if self.point is not None:
            if target_key!=self.key:
                return self._miss(now,'different_target')
            if math.dist(point,self.point)>self.max_jump:
                return self._miss(now,'position_jump')
        if self.last_match is None or stamp-self.last_match>self.max_gap or self.count==0:
            self.count=0;self.started=stamp
        self.point=tuple(point);self.key=target_key;self.last_match=stamp;self.count+=1
        ready=self.count>=self.confirm_frames and stamp-self.started>=self.min_duration
        return GateResult('confirmed' if ready else 'confirming',ready,self.point,True,self.count)
