"""Cached rotation/scale chamfer matching for low-texture 2D shapes.

Images are finite float32/64 grayscale in [0,1]. Coordinates are pixel centers
(x,y). Positive angles follow OpenCV: counterclockwise in the displayed image.
Every accepted candidate is checked against ALL template edge points at the
original scene resolution. Distance/support scores are not probabilities.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

__all__ = ['ShapeTemplate', 'create_shape_template', 'match_shape']


def _immutable(array):
    return np.frombuffer(array.tobytes(), array.dtype).reshape(array.shape)


def _image_u8(image):
    if (not isinstance(image, np.ndarray) or image.ndim != 2 or not image.size or
            image.dtype not in (np.float32, np.float64) or max(image.shape) > 32767):
        raise ValueError('Expected nonempty float32/64 grayscale, dimensions <=32767')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Image values must be finite in [0,1]')
    return np.ascontiguousarray(np.rint(image * 255), dtype=np.uint8)


def _edges(image, low, high, sigma):
    if sigma:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.Canny(image, low, high, L2gradient=True)


def _rotation(angle, scale):
    radians = math.radians(angle)
    c, s = math.cos(radians) * scale, math.sin(radians) * scale
    return np.array([[c, s], [-s, c]], np.float64)


def _kernel(offsets):
    # Bilinear splatting preserves all points, including coincident pixels.
    # Correlating this kernel equals mean bilinear distance sampling (up to
    # floating-point error); binary rasterization would discard edge weights.
    minimum = np.floor(offsets.min(axis=0)).astype(np.int64)
    maximum = np.ceil(offsets.max(axis=0)).astype(np.int64)
    dimensions = maximum - minimum + 1
    if np.any(dimensions > 4096):
        raise ValueError('A transformed edge kernel exceeds 4096 pixels')
    kernel = np.zeros(tuple(dimensions[::-1]), np.float32)
    position = offsets - minimum
    base = np.floor(position).astype(np.int64)
    fraction = position - base
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        weight = ((fraction[:, 0] if dx else 1-fraction[:, 0]) *
                  (fraction[:, 1] if dy else 1-fraction[:, 1]))
        use = weight > 0
        np.add.at(kernel, (base[use, 1]+dy, base[use, 0]+dx),
                  (weight[use]/len(offsets)).astype(np.float32))
    return _immutable(kernel), tuple(int(v) for v in minimum)


@dataclass(frozen=True)
class _Pose:
    angle_degrees: float
    scale: float
    offsets_xy: np.ndarray
    kernel: np.ndarray
    kernel_origin_xy: tuple


@dataclass(frozen=True)
class ShapeTemplate:
    """Build with create_shape_template; cached arrays have immutable storage."""

    shape: tuple
    center_xy: tuple
    points_xy: np.ndarray
    poses: tuple
    canny_low: float
    canny_high: float
    blur_sigma: float


def create_shape_template(image, *, angles=(-30., -15., 0., 15., 30.),
                          scales=(.9, 1., 1.1), canny_low=40., canny_high=100.,
                          blur_sigma=.7, mask=None, min_edges=12):
    """Cache all requested angle/scale hypotheses; no edge subsampling.

    angles are degrees, scales are positive isotropic factors. Angles/scales
    are discrete: this matcher does NOT interpolate them or estimate a
    perspective warp. mask, when supplied, is a bool array selecting template
    edges AFTER extraction; its boundary does not create artificial edges.
    Empty/underconstrained templates raise ValueError. Limits reject overly
    large models instead of silently discarding points/poses for speed.
    """
    if (not math.isfinite(canny_low) or not math.isfinite(canny_high) or
            not 0 <= canny_low < canny_high <= 1020 or
            not math.isfinite(blur_sigma) or not 0 <= blur_sigma <= 10 or
            type(min_edges) is not int or not 4 <= min_edges <= 50000):
        raise ValueError('Invalid edge extraction parameters')
    image_u8 = _image_u8(image)
    angle_values = np.asarray(angles, dtype=np.float64)
    scale_values = np.asarray(scales, dtype=np.float64)
    if (angle_values.ndim != 1 or scale_values.ndim != 1 or
            not angle_values.size or not scale_values.size or
            not np.isfinite(angle_values).all() or not np.isfinite(scale_values).all() or
            np.any(np.abs(angle_values) > 36000) or
            np.any(scale_values < .05) or np.any(scale_values > 20)):
        raise ValueError('Use finite angle and scale sequences; scales in [0.05,20]')
    angle_values = np.unique((angle_values+180) % 360-180)
    scale_values = np.unique(scale_values)
    if len(angle_values) * len(scale_values) > 720:
        raise ValueError('At most 720 angle/scale combinations are supported')
    edges = _edges(image_u8, canny_low, canny_high, blur_sigma)
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != image.shape:
            raise ValueError('mask must be a bool array with the image shape')
        edges[~mask] = 0
    y, x = np.nonzero(edges)
    if not min_edges <= len(x) <= 50000:
        raise ValueError('Template must contain min_edges..50000 edge pixels')
    points = np.column_stack((x, y)).astype(np.float64)
    singular = np.linalg.svd(points-points.mean(axis=0), compute_uv=False)
    if singular[-1] < 1e-3 * singular[0]:
        raise ValueError('Template edge geometry is collinear')
    center = ((image.shape[1]-1)/2, (image.shape[0]-1)/2)
    offsets = points - center
    poses = []
    total_pixels = 0
    for scale in scale_values:
        for angle in angle_values:
            transformed = offsets @ _rotation(float(angle), float(scale)).T
            kernel, origin = _kernel(transformed)
            total_pixels += kernel.size
            if total_pixels > 16_000_000:
                raise ValueError('Cached kernels exceed 16 million pixels; narrow pose grid')
            poses.append(_Pose(float(angle), float(scale), _immutable(transformed), kernel, origin))
    return ShapeTemplate(image.shape, center, _immutable(points), tuple(poses),
                         float(canny_low), float(canny_high), float(blur_sigma))


def _sample(field, points):
    x, y = points[..., 0], points[..., 1]
    if (x.min() < 0 or y.min() < 0 or x.max() > field.shape[1]-1 or
            y.max() > field.shape[0]-1):
        return None
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    jx, jy = np.minimum(ix+1, field.shape[1]-1), np.minimum(iy+1, field.shape[0]-1)
    fx, fy = x-ix, y-iy
    return ((1-fy)*((1-fx)*field[iy, ix]+fx*field[iy, jx]) +
            fy*((1-fx)*field[jy, ix]+fx*field[jy, jx]))


def _in_roi(center, roi):
    return roi[0] <= center[0] < roi[0]+roi[2] and roi[1] <= center[1] < roi[1]+roi[3]


def _verify(center, pose, distance, clip_distance, tolerance, roi):
    if not _in_roi(center, roi):
        return None
    values = _sample(distance, pose.offsets_xy+center)
    if values is None:
        return None
    supported = values <= tolerance
    return {'center_xy': np.asarray(center, np.float64),
            'mean_distance_px': float(values.mean()),
            'clipped_mean_distance_px': float(np.minimum(values, clip_distance).mean()),
            'support_fraction': float(supported.mean()),
            'inlier_rms_px': (float(np.sqrt(np.mean(values[supported]**2)))
                             if supported.any() else None)}


def _verify_many(centers, pose, distance, clip_distance, tolerance, roi):
    """Batch up to eight refinements, preserving scalar sampling and ordering.

    _verify remains the independent scalar reference. This batches only array
    work: all edge points and all candidate positions are still evaluated.
    """
    lower = pose.offsets_xy.min(axis=0)
    upper = pose.offsets_xy.max(axis=0)
    valid = ((centers[:, 0] >= roi[0]) & (centers[:, 0] < roi[0]+roi[2]) &
             (centers[:, 1] >= roi[1]) & (centers[:, 1] < roi[1]+roi[3]) &
             np.all(centers+lower >= 0, axis=1) &
             np.all(centers+upper <= np.array(distance.shape[::-1])-1, axis=1))
    indices = np.flatnonzero(valid)
    results = [None] * len(centers)
    if not len(indices):
        return results
    values = _sample(distance, pose.offsets_xy[None, :, :]+centers[indices, None, :])
    means = values.mean(axis=1)
    clipped_means = np.minimum(values, clip_distance).mean(axis=1)
    supported = values <= tolerance
    fractions = supported.mean(axis=1)
    for row, index in enumerate(indices):
        inliers = values[row, supported[row]]
        results[index] = {
            'center_xy': centers[index],
            'mean_distance_px': float(means[row]),
            'clipped_mean_distance_px': float(clipped_means[row]),
            'support_fraction': float(fractions[row]),
            'inlier_rms_px': float(np.sqrt(np.mean(inliers**2))) if len(inliers) else None,
        }
    return results


def _key(result):
    return (result['clipped_mean_distance_px'], -result['support_fraction'],
            result['mean_distance_px'])


def _refinement_improvement_bound(distance):
    # Bilinear interpolation has coordinate derivatives bounded by the largest
    # adjacent-pixel difference in each direction. The .5/.25/.125 search can
    # move at most .875 px along either axis. Clipping and averaging cannot
    # increase this bound. Read the actual field (including float32 rounding),
    # rather than assuming the ideal distance field has a unit gradient.
    dx = float(np.abs(np.diff(distance, axis=1)).max()) if distance.shape[1] > 1 else 0.
    dy = float(np.abs(np.diff(distance, axis=0)).max()) if distance.shape[0] > 1 else 0.
    return .875*(dx+dy)+1e-6  # conservative floating-point guard


def _candidates(distance, pose, roi, step, limit, clip_distance):
    if step == 1:
        kernel, origin = pose.kernel, pose.kernel_origin_xy
    else:
        kernel, origin = _kernel(pose.offsets_xy/step)
    field = distance[::step, ::step]
    if field.shape[0] < kernel.shape[0] or field.shape[1] < kernel.shape[1]:
        return [], False
    # A result location (u,v) is a kernel top-left, so center = (u,v)-origin.
    # Clipping the distance field before interpolation is only a proposal
    # surrogate. _verify clips AFTER interpolation and checks every edge.
    score = cv2.matchTemplate(np.ascontiguousarray(field), kernel, cv2.TM_CCORR)
    left = max(0, math.ceil(roi[0]/step + origin[0]))
    top = max(0, math.ceil(roi[1]/step + origin[1]))
    right = min(score.shape[1], math.ceil((roi[0]+roi[2])/step + origin[0]))
    bottom = min(score.shape[0], math.ceil((roi[1]+roi[3])/step + origin[1]))
    if left >= right or top >= bottom:
        return [], False
    score = score[top:bottom, left:right]
    local_minimum = cv2.erode(score, np.ones((3, 3), np.uint8))
    y, x = np.nonzero((score <= local_minimum) & (score < clip_distance-1e-4))
    if not len(x):
        return [], False
    values = score[y, x]
    # Stable ordering also makes equal-score plateaus repeatable across calls.
    order = np.lexsort((x, y, values))
    truncated = len(order) > limit
    order = order[:limit]
    centers = np.column_stack(((x[order]+left-origin[0])*step,
                               (y[order]+top-origin[1])*step))
    return centers.astype(np.float64), truncated


def _corners(matrix, shape):
    h, w = shape
    corners = np.array([[0., 0.], [w-1., 0.], [w-1., h-1.], [0., h-1.]])
    return corners @ matrix[:2, :2].T + matrix[:2, 2]


def _iou(a, b):
    area_a = abs(cv2.contourArea(a.astype(np.float32)))
    area_b = abs(cv2.contourArea(b.astype(np.float32)))
    intersect, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    return float(intersect/max(area_a+area_b-intersect, 1e-12))


def match_shape(image, model, *, roi=None, max_matches=1, distance_tolerance=2.,
                min_support=.7, max_mean_distance=1.5, clip_distance=6.,
                candidates_per_pose=24, coarse_step=1, refine_translation=True,
                nms_iou=.35):
    """Find low-texture shape instances on a cached discrete angle/scale grid.

    roi=(x,y,width,height) limits centers, using original scene coordinates;
    scene edges outside the ROI remain available. All transformed template
    edge points must remain inside the scene, even with occlusion enabled.
    max_mean_distance bounds the mean distance AFTER clipping at clip_distance.
    min_support bounds the fraction within distance_tolerance (scene pixels).

    Default coarse_step=1 computes the complete integer-translation score map
    for EACH cached pose. Local minima are shortlisted (candidates_per_pose),
    then independently verified against ALL full-resolution edge points.
    Optional coarse_step=2..8 accelerates proposal generation and searches a
    full-resolution translation neighborhood before final verification. It
    can lose recall: this is explicit, never an automatically enabled shortcut.
    Subpixel translation refinement uses .5/.25/.125 pixel coordinate searches;
    returned angles/scales remain on the supplied grid.

    No candidate is accepted from a coarse/raster score. candidate_budget_hit
    reports a truncated shortlist; increasing it can recover missed targets.
    The shortlist score interpolates a clipped distance field; final statistics
    instead clip interpolated distances. These differ near the clipping radius.
    Even full-grid search is not a global continuous pose optimum. Symmetric
    shapes, dense clutter, strong perspective and broad occlusion are ambiguous.
    """
    if not isinstance(model, ShapeTemplate) or not model.poses:
        raise ValueError('model must be built by create_shape_template')
    numeric = (distance_tolerance, min_support, max_mean_distance, clip_distance, nms_iou)
    if (not all(math.isfinite(v) for v in numeric) or
            not 0 < distance_tolerance <= clip_distance <= 1000 or
            not 0 < min_support <= 1 or not 0 <= max_mean_distance < clip_distance or
            not 0 <= nms_iou < 1 or type(max_matches) is not int or not 1 <= max_matches <= 100 or
            type(candidates_per_pose) is not int or not 1 <= candidates_per_pose <= 10000 or
            type(coarse_step) is not int or not 1 <= coarse_step <= 8 or
            type(refine_translation) is not bool):
        raise ValueError('Invalid shape matching parameters')
    image_u8 = _image_u8(image)
    height, width = image.shape
    if roi is None:
        roi = (0, 0, width, height)
    else:
        values = np.asarray(roi)
        if (values.shape != (4,) or not np.issubdtype(values.dtype, np.integer) or
                values[0] < 0 or values[1] < 0 or values[2] <= 0 or values[3] <= 0 or
                int(values[0])+int(values[2]) > width or int(values[1])+int(values[3]) > height):
            raise ValueError('roi must be integer (x,y,width,height) inside the scene')
        roi = tuple(int(v) for v in values)
    edge_image = _edges(image_u8, model.canny_low, model.canny_high, model.blur_sigma)
    count = int(np.count_nonzero(edge_image))
    result = {'success': False, 'reason': 'no_scene_edges', 'matches': [],
              'scene_edge_count': count, 'template_edge_count': len(model.points_xy),
              'pose_count': len(model.poses), 'candidates_verified': 0,
              'candidate_budget_hit': False, 'coarse_step': coarse_step}
    if not count:
        return result
    distance = cv2.distanceTransform((edge_image == 0).astype(np.uint8),
                                    cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    clipped = np.minimum(distance, clip_distance)
    improvement_bound = _refinement_improvement_bound(distance) if refine_translation else 0.
    accepted = []
    refinement_offsets = np.array(((-1, -1), (0, -1), (1, -1), (-1, 0),
                                   (1, 0), (-1, 1), (0, 1), (1, 1)), np.float64)
    for pose in model.poses:
        centers, hit = _candidates(clipped, pose, roi, coarse_step,
                                   candidates_per_pose, clip_distance)
        result['candidate_budget_hit'] |= hit
        verified = []
        initial = None
        if coarse_step == 1 and len(centers):
            initial = []
            initial_chunk = max(1, min(64, 131072 // len(model.points_xy)))
            for start in range(0, len(centers), initial_chunk):
                initial.extend(_verify_many(centers[start:start+initial_chunk], pose,
                                            distance, clip_distance, distance_tolerance, roi))
        for index, center in enumerate(centers):
            best = (initial[index] if initial is not None else
                    _verify(center, pose, distance, clip_distance, distance_tolerance, roi))
            result['candidates_verified'] += 1
            if coarse_step > 1:
                # Full integer neighborhood: no reduced edge sample at this stage.
                for dy in range(1-coarse_step, coarse_step):
                    for dx in range(1-coarse_step, coarse_step):
                        trial = _verify(center+(dx, dy), pose, distance, clip_distance,
                                        distance_tolerance, roi)
                        if trial is not None and (best is None or _key(trial) < _key(best)):
                            best = trial
            # Even the mathematically best permitted refinement cannot pass.
            # This prunes work, not accepted matches; no candidate/edge budget
            # or approximate score is used to decide this rejection.
            if best is not None and best['clipped_mean_distance_px']-improvement_bound > max_mean_distance:
                best = None
            verified.append(best)
        if refine_translation:
            active = [i for i, best in enumerate(verified) if best is not None]
            # Bound the temporary (#candidates x 8 x #edges) distance matrix.
            # Chunking changes memory allocation only; every neighbor and the
            # original per-candidate tie order are still checked.
            chunk_size = max(1, min(64, 131072 // (8*len(model.points_xy))))
            for step in (.5, .25, .125):
                for start in range(0, len(active), chunk_size):
                    group = active[start:start+chunk_size]
                    origins = np.array([verified[i]['center_xy'] for i in group])
                    trials_xy = origins[:, None, :]+refinement_offsets[None, :, :]*step
                    trials = _verify_many(trials_xy.reshape(-1, 2), pose, distance,
                                          clip_distance, distance_tolerance, roi)
                    for row, index in enumerate(group):
                        best = verified[index]
                        for trial in trials[8*row:8*(row+1)]:
                            if trial is not None and _key(trial) < _key(best):
                                best = trial
                        verified[index] = best
        for best in verified:
            if best is None:
                continue
            if (best['support_fraction'] < min_support or
                    best['clipped_mean_distance_px'] > max_mean_distance):
                continue
            matrix = np.eye(3, dtype=np.float64)
            matrix[:2, :2] = _rotation(pose.angle_degrees, pose.scale)
            matrix[:2, 2] = best['center_xy']-matrix[:2, :2]@np.array(model.center_xy)
            best.update(angle_degrees=pose.angle_degrees, scale=pose.scale,
                        matrix=matrix, corners_xy=_corners(matrix, model.shape),
                        edge_count=len(model.points_xy))
            accepted.append(best)
    accepted.sort(key=_key)
    selected = []
    for candidate in accepted:
        if any(_iou(candidate['corners_xy'], old['corners_xy']) > nms_iou for old in selected):
            continue
        selected.append(candidate)
        if len(selected) == max_matches:
            break
    result.update(success=bool(selected), reason='ok' if selected else 'no_match', matches=selected)
    return result
