"""Batched subpixel calipers and rectangle edge metrology in image pixels.

The approximate geometry is supplied by the caller. These operators refine
nearby edges; they do not classify parts or supply camera calibration.
"""
import math

import cv2
import numpy as np

from operators.vision_ops import _gray, fit_line
from operators.robust_geometry import points_array


def _profiles(image, start, end, num_calipers, search_half_length, width, sigma):
    ends = points_array([start, end], 2, 2)
    start, end = ends
    delta = end - start
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length < 4:
        raise ValueError('Reference segment must be finite and at least 4 pixels long')
    if type(num_calipers) is not int or not 4 <= num_calipers <= 2048:
        raise ValueError('num_calipers must be 4..2048')
    if type(width) is not int or not 1 <= width <= 256 or num_calipers * width >= 32767:
        raise ValueError('Invalid caliper averaging width or remap row count')
    if (not math.isfinite(search_half_length) or not 3 <= search_half_length <= 4096 or
            not math.isfinite(sigma) or not .1 <= sigma <= min(100, search_half_length/2)):
        raise ValueError('Invalid search distance or smoothing sigma')
    if max(image.shape) >= 32767:
        raise ValueError('Image exceeds OpenCV remap dimension limit')
    n = int(math.ceil(2 * search_half_length)) + 1
    if n * num_calipers * width > 16000000:
        raise ValueError('Caliper sampling budget exceeded')
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]])
    stations = np.linspace(0, length, num_calipers)
    centers = start + stations[:, None] * tangent
    distances = np.linspace(-search_half_length, search_half_length, n)
    cross = np.arange(width) - (width - 1)/2
    xy = (centers[:, None, None, :] + cross[None, :, None, None] * tangent +
          distances[None, None, :, None] * normal)
    if not np.isfinite(xy).all() or (np.abs(xy) > np.finfo(np.float32).max).any():
        raise ValueError('Sampling coordinates exceed float32 range')
    supported = ((xy[..., 0] >= 0) & (xy[..., 0] <= image.shape[1]-1) &
                 (xy[..., 1] >= 0) & (xy[..., 1] <= image.shape[0]-1)).all(axis=(1, 2))
    maps = xy.reshape(num_calipers * width, n, 2).astype(np.float32)
    sampled = cv2.remap(image, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT).reshape(num_calipers, width, n)
    profiles = sampled.mean(axis=1)
    step = distances[1] - distances[0]
    smooth = cv2.GaussianBlur(profiles, (0, 1), sigma/step, borderType=cv2.BORDER_REPLICATE)
    gradient = np.gradient(smooth, step, axis=1)
    return centers, stations, normal, tangent, distances, gradient, supported


def _measure_line(image, start, end, *, num_calipers, search_half_length,
                  width, sigma, threshold, polarity, selection, fit_threshold,
                  min_inliers, min_inlier_ratio, min_coverage, max_angle_deviation, seed):
    if (not all(math.isfinite(v) for v in [threshold, fit_threshold, min_inlier_ratio,
                                          min_coverage, max_angle_deviation]) or
            threshold <= 0 or fit_threshold <= 0 or not 0 < min_inlier_ratio <= 1 or
            not 0 < min_coverage <= 1 or not 0 < max_angle_deviation < 90):
        raise ValueError('Invalid edge, fit or coverage thresholds')
    if type(min_inliers) is not int or not 3 <= min_inliers <= num_calipers:
        raise ValueError('min_inliers must be 3..num_calipers')
    if polarity not in ('bright', 'dark', 'both') or selection not in ('nearest', 'strongest'):
        raise ValueError('Invalid polarity or selection rule')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    centers, stations, normal, tangent, distances, gradient, supported = _profiles(
        image, start, end, num_calipers, search_half_length, width, sigma)
    response = gradient if polarity == 'bright' else (-gradient if polarity == 'dark' else np.abs(gradient))
    middle = response[:, 1:-1]
    peaks = (middle > response[:, :-2]) & (middle >= response[:, 2:]) & (middle >= threshold)
    if selection == 'strongest':
        scores = np.where(peaks, middle, -np.inf)
    else:
        scores = np.where(peaks, -np.abs(distances[None, 1:-1]), -np.inf)
    good = supported & np.isfinite(scores.max(axis=1))
    rows = np.flatnonzero(good)
    index = np.argmax(scores[good], axis=1) + 1
    a, b, c = (response[rows, index+d].astype(np.float64) for d in (-1, 0, 1))
    fraction = np.clip(.5 * (a-c)/(a-2*b+c), -.5, .5)
    offsets = distances[index] + fraction * (distances[1]-distances[0])
    edge_points = centers[good] + offsets[:, None] * normal
    result = {'success': False, 'reason': 'insufficient_edges', 'point': None,
              'direction': None, 'normal': None, 'line_offset': None, 'segment_xy': None,
              'rms': None, 'coverage': 0., 'inlier_ratio': 0., 'angle_deviation_deg': None,
              'edge_points': edge_points, 'caliper_indices': rows, 'offsets': offsets,
              'amplitudes': gradient[rows, index], 'inliers': np.zeros(len(rows), bool),
              'supported_calipers': supported}
    if len(rows) < min_inliers:
        return result
    try:
        fitted = fit_line(edge_points, threshold=fit_threshold, min_inliers=min_inliers, seed=seed)
    except ValueError as error:
        result.update(reason='fit_failed', detail=str(error))
        return result
    direction = fitted['direction'].copy()
    if direction @ tangent < 0:
        direction *= -1
    fitted_normal = np.array([-direction[1], direction[0]])
    supported_stations = stations[rows[fitted['inliers']]]
    coverage = float(np.ptp(supported_stations) / stations[-1])
    ratio = float(fitted['inliers'].mean())
    angle = math.degrees(math.acos(float(np.clip(direction @ tangent, -1, 1))))
    projected = (edge_points[fitted['inliers']] - fitted['point']) @ direction
    segment = fitted['point'] + np.array([projected.min(), projected.max()])[:, None] * direction
    result.update(fitted, direction=direction, normal=fitted_normal,
                  line_offset=float(fitted_normal @ fitted['point']), segment_xy=segment,
                  coverage=coverage, inlier_ratio=ratio, angle_deviation_deg=angle)
    if ratio < min_inlier_ratio:
        result['reason'] = 'insufficient_inlier_ratio'
    elif coverage < min_coverage:
        result['reason'] = 'insufficient_coverage'
    elif angle > max_angle_deviation:
        result['reason'] = 'direction_mismatch'
    else:
        result.update(success=True, reason='ok')
    return result


def measure_line(image, start, end, *, num_calipers=32, search_half_length=10.,
                 width=5, sigma=1., threshold=.03, polarity='both', selection='nearest',
                 fit_threshold=.5, min_inliers=8, min_inlier_ratio=.6,
                 min_coverage=.5, max_angle_deviation=20., seed=0):
    """Refine an approximate line with simultaneous transverse calipers + RANSAC.

    Calipers run along the left normal (-dy,dx) of start->end. bright/dark refer
    to that scan direction. width averages along the reference line. nearest
    selects the peak closest to each predicted center; strongest selects maximum
    gradient. Ties choose the lower sample index. Entire out-of-frame calipers
    are skipped. inliers index edge_points; coverage measures reference segment
    span, not continuous support. Returned endpoints span measured inliers.
    success=False can retain diagnostic fit fields; do not treat them as valid.
    """
    return _measure_line(_gray(image), start, end, num_calipers=num_calipers,
        search_half_length=search_half_length, width=width, sigma=sigma,
        threshold=threshold, polarity=polarity, selection=selection,
        fit_threshold=fit_threshold, min_inliers=min_inliers, min_inlier_ratio=min_inlier_ratio,
        min_coverage=min_coverage, max_angle_deviation=max_angle_deviation, seed=seed)


def measure_rectangle(image, center, size, angle_deg=0., *, num_calipers=24,
                      search_half_length=6., width=5, sigma=1., threshold=.03,
                      fit_threshold=.5, min_inlier_ratio=.6, max_angle_error=5., seed=0):
    """Refine four approximate rectangle edges without forcing exact right angles.

    center=(x,y), size=(width,height), angle follows +X toward +Y in image space.
    The middle 80% of each side is sampled to avoid corners. Independent lines
    yield TL/TR/BR/BL corners in the prior's local frame. Opposite side lengths
    are averaged into width_px/height_px; orthogonality and opposing-angle checks
    reject geometry incompatible with a rectangle. Perspective quadrilaterals
    should be rectified or measured using separate lines instead.
    """
    image = _gray(image)
    center, size = np.asarray(center, np.float64), np.asarray(size, np.float64)
    if center.shape != (2,) or size.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(size).all() or (size < 10).any():
        raise ValueError('Finite center and size >=10 pixels required')
    if not math.isfinite(angle_deg) or not math.isfinite(max_angle_error) or not 0 < max_angle_error <= 30:
        raise ValueError('Invalid rectangle orientation tolerance')
    if type(num_calipers) is not int or not 8 <= num_calipers <= 2048:
        raise ValueError('num_calipers must be 8..2048')
    if not math.isfinite(search_half_length) or search_half_length >= float(size.min())/2:
        raise ValueError('Search range must not reach the opposite rectangle edge')
    angle = math.radians(angle_deg)
    u = np.array([math.cos(angle), math.sin(angle)])
    v = np.array([-u[1], u[0]])
    corners = center + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) @ np.stack((u*size[0]/2, v*size[1]/2))
    lines = []
    for i in range(4):
        a, b = corners[i], corners[(i+1)%4]
        lines.append(_measure_line(image, a*.9+b*.1, a*.1+b*.9, num_calipers=num_calipers,
            search_half_length=search_half_length, width=width, sigma=sigma, threshold=threshold,
            polarity='both', selection='nearest', fit_threshold=fit_threshold,
            min_inliers=max(4, num_calipers//3), min_inlier_ratio=min_inlier_ratio,
            min_coverage=.5, max_angle_deviation=max_angle_error, seed=seed))
    result = {'success': False, 'reason': 'edge_failed', 'lines': lines, 'corners_xy': None,
              'center_xy': None, 'width_px': None, 'height_px': None, 'angle_deg': None,
              'orthogonality_error_deg': None, 'opposite_angle_error_deg': None}
    if not all(line['success'] for line in lines):
        return result
    directions = np.array([line['direction'] for line in lines])
    perpendicular = max(abs(90-math.degrees(math.acos(float(np.clip(directions[i] @ directions[(i+1)%4], -1, 1))))) for i in range(4))
    parallel = max(math.degrees(math.acos(float(np.clip(abs(directions[i] @ directions[i+2]), 0, 1)))) for i in range(2))
    result.update(orthogonality_error_deg=perpendicular, opposite_angle_error_deg=parallel)
    if max(perpendicular, parallel) > max_angle_error:
        result['reason'] = 'not_rectangular'
        return result
    refined = []
    for i in range(4):
        first, second = lines[(i-1)%4], lines[i]
        matrix = np.array([first['normal'], second['normal']])
        if abs(np.linalg.det(matrix)) < 1e-6:
            result['reason'] = 'parallel_adjacent_edges'
            return result
        refined.append(np.linalg.solve(matrix, [first['line_offset'], second['line_offset']]))
    refined = np.asarray(refined)
    if not np.isfinite(refined).all() or not cv2.isContourConvex(refined.astype(np.float32)):
        result['reason'] = 'invalid_corners'
        return result
    if np.linalg.norm(refined-corners, axis=1).max() > 2*search_half_length:
        result['reason'] = 'corner_extrapolation'
        return result
    lengths = np.linalg.norm(np.roll(refined, -1, axis=0)-refined, axis=1)
    first_edge = refined[1]-refined[0]
    result.update(success=True, reason='ok', corners_xy=refined, center_xy=refined.mean(axis=0),
                  width_px=float((lengths[0]+lengths[2])/2), height_px=float((lengths[1]+lengths[3])/2),
                  angle_deg=math.degrees(math.atan2(first_edge[1], first_edge[0])))
    return result
