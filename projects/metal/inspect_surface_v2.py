"""Experimental v2: double-sided ridge verification and separate border review.

v1 remains unchanged and callable. Neither version is an OK/NG classifier.
"""
from dataclasses import dataclass,asdict
import copy
import cv2
import numpy as np
from operators import ridge_ops as _ridge
from operators.ridge_ops import ridge_response, ridge_response_numpy
from projects.metal.inspect_surface import _inspect_core, Parameters, _components


@dataclass(frozen=True)
class Refinement:
    ridge_floor: float = 0.06
    min_supported_fraction: float = 0.60
    border_ridge_floor: float = 0.065
    border_margin: int = 7



def inspect(image,roi_mask,params=Parameters(),reference_mask=None,refinement=Refinement()):
    for value in [refinement.ridge_floor,refinement.border_ridge_floor,refinement.min_supported_fraction]:
        if not np.isfinite(value) or not 0<value<=1:raise ValueError('ridge floors must be in (0,1]')
    if type(refinement.border_margin) is not int or not 7<=refinement.border_margin<params.border_margin:
        raise ValueError('border margin must be >=7 and below the main inspection margin')
    report,maps,prepared=_inspect_core(image,roi_mask,params,reference_mask)
    report=copy.deepcopy(report);w,h=report['working_size'];ow,oh=report['original_size']
    gray=prepared['gray'];smooth=prepared['smooth']
    ridge=ridge_response(smooth)
    # Preserve entire v1 components; do not split one defect into new small boxes.
    count,labels,_,_=cv2.connectedComponentsWithStats(maps['scratch_mask'],8)
    evidence=maps['scratch_response']
    totals=np.bincount(labels.ravel(),weights=evidence.ravel(),minlength=count)
    support=np.bincount(labels.ravel(),weights=(evidence*(ridge>=refinement.ridge_floor)).ravel(),minlength=count)
    fractions=support/np.maximum(totals,1e-12)
    accepted=fractions>=refinement.min_supported_fraction;accepted[0]=False
    verified=(accepted[labels]).astype(np.uint8)*255
    candidates,kept=_components(verified,maps['scratch_response'],w/ow,h/oh,'scratch_candidate',params)
    roi=prepared['roi']
    m=refinement.border_margin
    extended=cv2.erode(roi,np.ones((2*m+1,2*m+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    border=extended & (maps['inspection_mask']==0) & (gray>.10)&(gray<.97)
    # Uses only the short-footprint ridge here, not v1's larger morphology.
    border_binary=(border&(ridge>=refinement.border_ridge_floor)).astype(np.uint8)*255
    border_binary=cv2.morphologyEx(border_binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    border_binary[~border]=0
    near,near_mask=_components(border_binary,ridge,w/ow,h/oh,'scratch_candidate',params)
    for item in near:
        item['kind']='border_review_candidate'
        item['review_reason']='Near metal boundary; contour reflections remain a possible cause.'
    report['version']='experimental_v2'
    report['refinement']=asdict(refinement)
    report['ridge_backend']='native' if _ridge._NATIVE is not None else 'numpy'
    report['v1_scratch_candidate_count']=len(report['scratch_candidates'])
    report['scratch_candidates']=candidates
    report['border_review_candidates']=near
    report['limitations'].append('Ridge verification can reject real broad/one-sided damage; fewer boxes do not prove better accuracy.')
    maps=dict(maps)
    maps['ridge_response']=ridge
    maps['rejected_v1_pixels']=((maps['scratch_mask']>0)&(kept==0)).astype(np.uint8)*255
    maps['scratch_mask']=kept;maps['border_review_mask']=near_mask
    return report,maps
