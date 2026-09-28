"""CPU image registration and caliper measurements, in original image pixels.

FFT/remapping/filtering run inside OpenCV. Intensity inputs are float32/64 [0,1].
No camera calibration or conversion from pixels to physical units is implied.
"""
from functools import lru_cache
import math

import cv2
import numpy as np

from operators.vision_ops import _gray, fit_circle, measure_edges


@lru_cache(maxsize=8)
def _hann(shape):
    result = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=1)
def _odd_phase_bias():
    # Some OpenCV releases subtract an integer peak centroid from grid_size/2
    # on odd DFT grids, so even identity registration returns +0.5 pixels.
    # Calibrate the loaded implementation, rather than assuming a version or
    # applying another correction after a future upstream fix. This does not
    # change the input's periodic boundary convention by padding to even sizes.
    impulse = np.zeros((5, 5), np.float32)
    impulse[2, 2] = 1
    shift, _ = cv2.phaseCorrelate(impulse, impulse)
    return tuple(float(v) for v in shift)


def estimate_translation(reference, moving, *, window=True, min_response=.15,
                         max_shift=None):
    """Phase correlation: displacement (dx,dy) from reference to moving image.

    alignment_matrix maps moving coordinates back to reference (negative shift).
    Only translation is modeled. Motion is periodic/ambiguous beyond half the
    image extent; repeated textures are ambiguous even at a high response.
    response is OpenCV peak response, not probability (it may exceed one).
    Low texture/confidence returns success=False, rather than a fabricated zero.
    Inputs are never modified, including when a Hann window is used.
    """
    reference = _gray(reference)
    moving = _gray(moving)
    if reference.shape != moving.shape or min(reference.shape) < 8:
        raise ValueError('Equal image shapes, at least 8x8, required')
    if type(window) is not bool or not math.isfinite(min_response) or not 0 <= min_response <= 1:
        raise ValueError('Invalid window or response threshold')
    limits = None
    if max_shift is not None:
        limits = np.asarray(max_shift, dtype=np.float64)
        if limits.shape != (2,) or not np.isfinite(limits).all() or np.any(limits < 0):
            raise ValueError('max_shift must be finite nonnegative (dx,dy) limits')
    # Subtraction allocates owned buffers, avoiding phaseCorrelate's in-place
    # window behavior on user arrays and removing the dominant DC component.
    a = reference - reference.mean()
    b = moving - moving.mean()
    result = {'success': False, 'reason': 'insufficient_texture',
              'shift_xy': None, 'response': 0., 'alignment_matrix': None}
    if float(np.linalg.norm(a)) / math.sqrt(a.size) < 1e-6 or float(np.linalg.norm(b)) / math.sqrt(b.size) < 1e-6:
        return result
    if window:
        weight = _hann(reference.shape)
        a *= weight
        b *= weight
    shift, response = cv2.phaseCorrelate(a, b)
    shift = np.asarray(shift, dtype=np.float64)
    odd_x = cv2.getOptimalDFTSize(reference.shape[1]) % 2
    odd_y = cv2.getOptimalDFTSize(reference.shape[0]) % 2
    if odd_x or odd_y:
        bias = _odd_phase_bias()
        shift -= (bias[0] * odd_x, bias[1] * odd_y)
    result.update(shift_xy=shift, response=float(response), reason='low_response')
    if not np.isfinite(shift).all() or not math.isfinite(response) or response < min_response:
        return result
    if limits is not None and np.any(np.abs(shift) > limits):
        result['reason'] = 'shift_limit'
        return result
    result.update(success=True, reason='ok', alignment_matrix=np.array(
        [[1., 0., -shift[0]], [0., 1., -shift[1]]]))
    return result


def measure_stripes(image, start, end, *, width=9, sigma=1., threshold=.03,
                    polarity='bright', min_width=2., max_width=None,
                    min_distance=3.):
    """Pair adjacent opposite-polarity caliper edges into complete stripes/gaps.

    bright means low->high->low along start->end; dark means the reverse.
    width is the transverse averaging sample count; width_px in each result is
    the measured separation along the caliper, not a perpendicular correction.
    Stripes cut off by the caliper boundary are not returned. No edge is skipped
    to invent a pair across an intervening detected edge.
    """
    if polarity not in ('bright', 'dark') or not math.isfinite(min_width) or min_width <= 0:
        raise ValueError('Invalid stripe polarity or minimum width')
    if max_width is not None and (not math.isfinite(max_width) or max_width < min_width):
        raise ValueError('max_width must be finite and >= min_width')
    measured = measure_edges(image, start, end, width=width, sigma=sigma,
                             threshold=threshold, polarity='both', min_distance=min_distance)
    pairs = []
    for first, second in zip(measured['edges'], measured['edges'][1:]):
        separation = second['distance'] - first['distance']
        if first['polarity'] != polarity or second['polarity'] == polarity:
            continue
        if separation < min_width or (max_width is not None and separation > max_width):
            continue
        pairs.append({'start_xy': first['xy'], 'end_xy': second['xy'],
                      'center_xy': ((np.asarray(first['xy']) + second['xy']) / 2).tolist(),
                      'width_px': float(separation),
                      'edge_amplitudes': [first['amplitude'], second['amplitude']]})
    return dict(measured, stripes=pairs)


def measure_circle(image, center, radius, *, radial_range=6., num_rays=128,
                   sigma=1., threshold=.03, polarity='both', fit_threshold=1.,
                   min_inliers=12, min_inlier_ratio=.6, min_coverage_deg=180., seed=0):
    """Radial calipers around an approximate center/radius, then robust circle fit.

    Each ray keeps its strongest interior gradient peak and refines it using a
    parabola. bright/dark describe intensity change from inside to outside.
    Search radii span radius +/- radial_range with <=1 pixel steps. Entire rays
    outside the image are discarded. Outputs preserve pixel-center (x,y).
    This needs an initial circle estimate; it is not a whole-image circle finder.
    Angular coverage is the complement of the largest gap between inlier angles;
    it does not prove continuous support. min_inlier_ratio is measured among
    detected edge points, not all attempted rays. Short arcs/ellipses need another
    model. Check success before using the fit; rejected fits retain diagnostics.
    """
    image = _gray(image)
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('center must be a finite (x,y)')
    if not all(math.isfinite(v) for v in (radius, radial_range, sigma, threshold, fit_threshold, min_inlier_ratio, min_coverage_deg)):
        raise ValueError('Circle parameters must be finite')
    if not 3 <= radial_range <= 4096 or radius <= radial_range or not .1 <= sigma <= radial_range / 2 or threshold <= 0 or fit_threshold <= 0:
        raise ValueError('Invalid radial range, radius, sigma, or thresholds')
    if type(num_rays) is not int or not 16 <= num_rays <= 4096:
        raise ValueError('num_rays must be 16..4096')
    if (type(min_inliers) is not int or not 3 <= min_inliers <= num_rays or
            not 0 < min_inlier_ratio <= 1 or not 0 <= min_coverage_deg <= 360 or
            type(seed) is not int or seed < 0):
        raise ValueError('Invalid support requirements')
    if polarity not in ('bright', 'dark', 'both') or max(image.shape) >= 32767:
        raise ValueError('Invalid polarity or image exceeds OpenCV remap limit')
    n = int(math.ceil(2 * radial_range)) + 1
    if n * num_rays > 16000000:
        raise ValueError('Radial sampling budget exceeded')
    radii = np.linspace(radius - radial_range, radius + radial_range, n)
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    xy = center + directions[:, None, :] * radii[None, :, None]
    if not np.isfinite(xy).all() or (np.abs(xy) > np.finfo(np.float32).max).any():
        raise ValueError('Sampling coordinates exceed float32 range')
    supported = ((xy[:, :, 0] >= 0) & (xy[:, :, 0] <= image.shape[1] - 1) &
                 (xy[:, :, 1] >= 0) & (xy[:, :, 1] <= image.shape[0] - 1)).all(axis=1)
    profiles = cv2.remap(image, xy[:, :, 0].astype(np.float32),
                         xy[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    step = radii[1] - radii[0]
    smooth = cv2.GaussianBlur(profiles, (0, 1), sigma / step, borderType=cv2.BORDER_REPLICATE)
    gradient = np.gradient(smooth, step, axis=1)
    response = gradient if polarity == 'bright' else (-gradient if polarity == 'dark' else np.abs(gradient))
    middle = response[:, 1:-1]
    peaks = (middle > response[:, :-2]) & (middle >= response[:, 2:]) & (middle >= threshold)
    peak_values = np.where(peaks, middle, -np.inf)
    indices = np.argmax(peak_values, axis=1) + 1
    good = supported & np.isfinite(peak_values.max(axis=1))
    rows = np.flatnonzero(good)
    indices = indices[good]
    left, peak, right = (response[rows, indices + d].astype(np.float64) for d in (-1, 0, 1))
    offset = np.clip(.5 * (left - right) / (left - 2 * peak + right), -.5, .5)
    selected_radii = radii[indices] + offset * step
    points = center + selected_radii[:, None] * directions[good]
    result = {'success': False, 'reason': 'insufficient_edges', 'center': None,
              'radius': None, 'rms': None, 'angular_coverage_deg': 0., 'inlier_ratio': 0.,
              'edge_points': points, 'ray_indices': rows,
              'amplitudes': gradient[rows, indices], 'inliers': np.zeros(len(points), bool),
              'supported_rays': supported}
    if len(points) < min_inliers:
        return result
    try:
        fitted = fit_circle(points, threshold=fit_threshold, min_inliers=min_inliers, seed=seed)
    except ValueError as error:
        result['reason'] = 'fit_failed'
        result['detail'] = str(error)
        return result
    result.update(fitted)
    result['inlier_ratio'] = float(np.count_nonzero(fitted['inliers']) / len(points))
    if result['inlier_ratio'] < min_inlier_ratio:
        result['reason'] = 'insufficient_inlier_ratio'
        return result
    if fitted['angular_coverage_deg'] < min_coverage_deg:
        result['reason'] = 'insufficient_coverage'
        return result
    result.update(success=True, reason='ok')
    return result
