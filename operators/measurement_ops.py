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


def _linear_overlap_correlation(reference, moving, dx, dy):
    """Pearson correlation on the actual, non-wrapping overlap at a shift."""
    h, w = reference.shape
    x, y = int(round(dx)), int(round(dy))
    width, height = w - abs(x), h - abs(y)
    if width < 12 or height < 12:
        return 0., 0.
    ax, bx = max(0, -x), max(0, x)
    ay, by = max(0, -y), max(0, y)
    a = reference[ay:ay + height, ax:ax + width]
    b = moving[by:by + height, bx:bx + width]
    am, bm = a - a.mean(), b - b.mean()
    norm = float(np.linalg.norm(am) * np.linalg.norm(bm))
    corr = float(np.sum(am * bm) / norm) if norm > 1e-12 else 0.
    return corr, float(width * height) / (w * h)


def _strong_self_similarity(image):
    """Detect a distinct, almost identical copy of the central scene patch."""
    h, w = image.shape
    if min(h, w) < 48:
        return 0.
    y0, x0 = h // 4, w // 4
    patch = image[y0:3 * h // 4, x0:3 * w // 4]
    if float(patch.std()) < 1e-4:
        return 0.
    scores = cv2.matchTemplate(image, patch, cv2.TM_CCOEFF_NORMED)
    # A few adjacent subpixel positions naturally correlate with the original.
    radius = max(3, int(round(min(h, w) * .025)))
    scores[max(0, y0 - radius):y0 + radius + 1,
           max(0, x0 - radius):x0 + radius + 1] = -1.
    return float(scores.max())


def _local_motion_clusters(reference, moving, global_shift):
    """Find spatially supported motions that disagree with the global peak."""
    h, w = reference.shape
    if min(h, w) < 96:
        return 0, 0, 0, False, 0., 0.
    sx, sy = global_shift
    x0, x1 = max(0, math.ceil(-sx)) + 2, min(w, math.floor(w - sx)) - 2
    y0, y1 = max(0, math.ceil(-sy)) + 2, min(h, math.floor(h - sy)) - 2
    if x1 - x0 < 96 or y1 - y0 < 96:
        return 0, 0, 0, False, 0., 0.
    # First align the proposed global motion. A tile now estimates only its
    # residual, avoiding local FFT aliasing on otherwise valid large shifts.
    aligned = cv2.warpAffine(moving, np.float32([[1, 0, -sx], [0, 1, -sy]]),
                             (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    rows, cols = 3, 4
    tile_h, tile_w = (y1 - y0) // rows, (x1 - x0) // cols
    window = _hann((tile_h, tile_w))
    motions = []
    coarse_motions, coarse_positions = [], []
    for row in range(rows):
        for col in range(cols):
            y, x = y0 + row * tile_h, x0 + col * tile_w
            a = reference[y:y + tile_h, x:x + tile_w]
            b = aligned[y:y + tile_h, x:x + tile_w]
            if min(float(a.std()), float(b.std())) < 1e-4:
                continue
            shift, response = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), window)
            if response >= .25 and np.isfinite(shift).all():
                motions.append(shift)
                coarse_motions.append(shift)
                coarse_positions.append((2 * col / (cols - 1) - 1,
                                         2 * row / (rows - 1) - 1))
    # Wide, shallow border strips remain useful when a large central object
    # contaminates every coarse tile. Top/bottom resolve horizontal residuals;
    # left/right resolve vertical residuals. Disjoint strips give independent
    # evidence instead of counting the same foreground multiple times.
    strip_h = max(24, (y1 - y0) // 6)
    strip_w = max(24, (x1 - x0) // 6)
    borders = [
        (slice(y0, y0 + strip_h), slice(x0, x1)),
        (slice(y1 - strip_h, y1), slice(x0, x1)),
        (slice(y0 + strip_h, y1 - strip_h), slice(x0, x0 + strip_w)),
        (slice(y0 + strip_h, y1 - strip_h), slice(x1 - strip_w, x1)),
    ]
    for ys, xs in borders:
        a, b = reference[ys, xs], aligned[ys, xs]
        if min(a.shape) < 24 or min(float(a.std()), float(b.std())) < 1e-4:
            continue
        shift, response = cv2.phaseCorrelate(
            a - a.mean(), b - b.mean(), _hann(a.shape))
        if response >= .35 and np.isfinite(shift).all():
            motions.append(shift)
    if motions:
        motions = np.asarray(motions)
    else:
        motions = np.empty((0, 2), dtype=np.float64)
    # A second coherent cluster is evidence of two scene motions. Independent
    # outlier tiles or motion at an occlusion seam do not suffice.
    tolerance = max(1.5, .025 * min(tile_h, tile_w))
    same = np.linalg.norm(motions, axis=1) <= tolerance
    best_competitor = 0
    for candidate in motions[~same]:
        count = int((np.linalg.norm(motions - candidate, axis=1) <= tolerance).sum())
        best_competitor = max(best_competitor, count)
    affine_deformation, affine_fit_ratio = 0., 0.
    if len(coarse_motions) >= 8:
        values = np.asarray(coarse_motions, dtype=np.float64)
        positions = np.asarray(coarse_positions, dtype=np.float64)
        design = np.column_stack((np.ones(len(values)), positions))
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        predictions = design @ coefficients
        scatter = float(np.sum((values - values.mean(axis=0)) ** 2))
        if scatter > 1e-8:
            affine_fit_ratio = max(0., 1. - float(np.sum((values - predictions) ** 2)) / scatter)
        corners = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        affine_deformation = float(np.linalg.norm(corners @ coefficients[1:], axis=1).max())
    # Coarse regions can all contain the same high-contrast foreground. Probe
    # two opposite, non-overlapping outer bands that may still expose static
    # background. A one-sided disagreement is never sufficient: thin bands can
    # contain noise, occlusions, or a scene edge with no reliable translation.
    thin_conflict = False
    if best_competitor < 2:
        def probe(ys, xs):
            a, b = reference[ys, xs], aligned[ys, xs]
            if min(a.shape) < 8 or min(float(a.std()), float(b.std())) < 1e-4:
                return None
            shift, response = cv2.phaseCorrelate(
                a - a.mean(), b - b.mean(), _hann(a.shape))
            if not np.isfinite(shift).all() or response < .6:
                return None
            return np.asarray(shift)

        for thickness in (8, 12):
            if y1 - y0 >= 2 * thickness + 16:
                top = probe(slice(y0, y0 + thickness), slice(x0, x1))
                bottom = probe(slice(y1 - thickness, y1), slice(x0, x1))
                if (top is not None and bottom is not None and
                    min(abs(top[0]), abs(bottom[0])) > 2. and
                        abs(top[0] - bottom[0]) <= 1.5):
                    thin_conflict = True
                    break
            if x1 - x0 >= 2 * thickness + 16:
                left = probe(slice(y0, y1), slice(x0, x0 + thickness))
                right = probe(slice(y0, y1), slice(x1 - thickness, x1))
                if (left is not None and right is not None and
                    min(abs(left[1]), abs(right[1])) > 2. and
                        abs(left[1] - right[1]) <= 1.5):
                    thin_conflict = True
                    break
    return (int(same.sum()), best_competitor, len(motions), thin_conflict,
            affine_deformation, affine_fit_ratio)


def estimate_scene_motion(reference, moving, *, max_analysis_width=480,
                          min_response=.15, max_shift=None):
    """Conservative image-plane shift for consecutive camera frames.

    Returns image displacement from reference to moving in original pixels.
    This is not a 3-D camera or robot trajectory. The model assumes linear
    (non-wrapping) translation; rotation, parallax and independent objects can
    make one global displacement undefined. Ambiguous estimates fail closed.
    ``max_analysis_width`` bounds the longest analysis dimension for speed.
    A caller with a known frame rate can set ``max_shift=(dx, dy)`` limits.
    OpenCV response is a peak measure, not a calibrated probability.
    """
    reference, moving = _gray(reference), _gray(moving)
    if reference.shape != moving.shape or min(reference.shape) < 16:
        raise ValueError('Equal image shapes, at least 16x16, required')
    if type(max_analysis_width) is not int or not 32 <= max_analysis_width <= 4096:
        raise ValueError('max_analysis_width must be an integer in 32..4096')
    if not math.isfinite(min_response) or not 0 <= min_response <= 1:
        raise ValueError('min_response must be finite and in [0,1]')
    limits = None
    if max_shift is not None:
        limits = np.asarray(max_shift, dtype=np.float64)
        if limits.shape != (2,) or not np.isfinite(limits).all() or np.any(limits < 0):
            raise ValueError('max_shift must be finite nonnegative (dx,dy) limits')
    h0, w0 = reference.shape
    scale = min(1., max_analysis_width / max(h0, w0))
    if scale < 1:
        size = (max(16, round(w0 * scale)), max(16, round(h0 * scale)))
        reference = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        moving = cv2.resize(moving, size, interpolation=cv2.INTER_AREA)
    h, w = reference.shape
    result = {'success': False, 'reason': 'insufficient_texture',
              'shift_xy': None, 'response': 0., 'alignment_matrix': None,
              'analysis_shape_hw': (h, w)}
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(moving, dtype=np.float32)
    if min(float(a.std()), float(b.std())) < 1e-4:
        return result
    aa = np.zeros((h * 2, w * 2), dtype=np.float32)
    bb = np.zeros_like(aa)
    aa[:h, :w] = a - a.mean()
    bb[:h, :w] = b - b.mean()
    shift, response = cv2.phaseCorrelate(aa, bb)
    shift = np.asarray(shift, dtype=np.float64)
    shift *= (w0 / w, h0 / h)
    result.update(shift_xy=shift, response=float(response), reason='low_response')
    if not np.isfinite(shift).all() or not math.isfinite(response) or response < min_response:
        return result
    if limits is not None and np.any(np.abs(shift) > limits):
        result['reason'] = 'shift_limit'
        return result
    sx, sy = shift * (w / w0, h / h0)
    overlap = max(0., 1. - abs(sx) / w) * max(0., 1. - abs(sy) / h)
    result['overlap_fraction'] = float(overlap)
    if overlap < .2:
        result['reason'] = 'insufficient_overlap'
        return result
    similarity = _strong_self_similarity(a)
    result['self_similarity'] = similarity
    if similarity > .965:
        result['reason'] = 'ambiguous_texture'
        return result
    # A circularly shifted image can perfectly support two opposite linear
    # interpretations. Test all other wrapped aliases with useful overlap.
    primary, _ = _linear_overlap_correlation(a, b, sx, sy)
    result['overlap_correlation'] = primary
    # A chance FFT peak on unrelated frames can exceed min_response,
    # especially on tiny images with too few independent samples. True
    # translations must also preserve visible structure in the overlap.
    if primary < .25:
        result['reason'] = 'low_overlap_correlation'
        return result
    for offset_x in (-w, 0, w):
        for offset_y in (-h, 0, h):
            if offset_x == 0 and offset_y == 0:
                continue
            alternative, support = _linear_overlap_correlation(
                a, b, sx + offset_x, sy + offset_y)
            if support >= .2 and alternative >= max(.75, primary - .12):
                result['reason'] = 'ambiguous_wrap'
                return result
    same, other, tile_count, thin_conflict, affine_deformation, affine_fit_ratio = (
        _local_motion_clusters(a, b, (sx, sy)))
    result['local_consensus'] = {'global_tiles': same,
                                 'competing_tiles': other, 'valid_tiles': tile_count,
                                 'opposite_border_conflict': thin_conflict,
                                 'affine_deformation': affine_deformation,
                                 'affine_fit_ratio': affine_fit_ratio}
    if (other >= 2 or thin_conflict or
            (affine_deformation > .45 and affine_fit_ratio > .65)):
        result['reason'] = 'inconsistent_motion'
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
