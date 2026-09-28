"""Cached ORB planar localization using OpenCV's native CPU implementation.

Images are finite float32/float64 HxW grayscale in [0, 1], quantized to uint8
for ORB. Coordinates are pixel centers (x, y); H maps template -> scene.
This is a single planar-instance locator, not a 3D pose estimator. Texture,
moderate viewpoint change and overlapping visible content are required.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

__all__ = ['OrbTemplate', 'create_orb_template', 'locate_planar_template']


@dataclass(frozen=True)
class OrbTemplate:
    """Reusable features; build with factory, do not construct/replace fields.

    Factory arrays are backed by immutable bytes, not only a writeable flag.
    """

    shape: tuple
    points_xy: np.ndarray
    descriptors: np.ndarray
    max_features: int
    scale_factor: float
    levels: int
    fast_threshold: int


def _image_u8(image):
    if (not isinstance(image, np.ndarray) or image.ndim != 2 or
            image.dtype not in (np.float32, np.float64) or not image.size):
        raise ValueError('Expected nonempty float32/float64 HxW grayscale')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Image values must be finite in [0,1]')
    if max(image.shape) > 32767:
        raise ValueError('Image dimensions must be <=32767 pixels')
    return np.ascontiguousarray(np.rint(image * 255), dtype=np.uint8)


def _detector(model):
    # A fresh detector avoids shared mutable OpenCV state across caller threads.
    return cv2.ORB_create(nfeatures=model.max_features,
                          scaleFactor=model.scale_factor, nlevels=model.levels,
                          edgeThreshold=15, patchSize=31,
                          fastThreshold=model.fast_threshold)


def _features(detector, image):
    keys, descriptors = detector.detectAndCompute(image, None)
    points = np.array([key.pt for key in keys], dtype=np.float32).reshape(-1, 2)
    return points, descriptors


def _spread(points):
    if len(points) < 4 or len(np.unique(points, axis=0)) < 4:
        return False
    values = np.linalg.svd(points.astype(np.float64) - points.mean(axis=0),
                           compute_uv=False)
    return bool(values[0] > 1e-6 and values[1] > values[0] * 1e-3)


def create_orb_template(image, *, max_features=1500, scale_factor=1.2,
                        levels=8, fast_threshold=15):
    """Extract and cache template keypoints/descriptors once.

    Raises ValueError on invalid parameters, fewer than eight keypoints, or
    degenerate feature geometry. Read-only cached arrays own their storage;
    subsequent changes to the input image cannot alter this model.
    max_features bounds detection and descriptor-matching cost per image.
    """
    if (type(max_features) is not int or not 32 <= max_features <= 10000 or
            type(levels) is not int or not 1 <= levels <= 16 or
            not math.isfinite(scale_factor) or not 1.05 <= scale_factor <= 2 or
            type(fast_threshold) is not int or not 0 <= fast_threshold <= 255):
        raise ValueError('Invalid ORB parameters')
    image_u8 = _image_u8(image)
    empty = OrbTemplate(image.shape, None, None, max_features,
                        float(scale_factor), levels, fast_threshold)
    points, descriptors = _features(_detector(empty), image_u8)
    if descriptors is None or len(points) < 8 or not _spread(points):
        raise ValueError('Template needs at least eight nondegenerate ORB keypoints')
    # Immutable backing storage also rejects setflags(write=True); frozen
    # dataclasses alone do not protect a mutable ndarray stored in a field.
    points = np.frombuffer(points.tobytes(), np.float32).reshape(points.shape)
    descriptors = np.frombuffer(descriptors.tobytes(), np.uint8).reshape(descriptors.shape)
    return OrbTemplate(image.shape, points, descriptors, max_features,
                       float(scale_factor), levels, fast_threshold)


def _empty_result(reason, scene_keypoints=0):
    return {'success': False, 'reason': reason, 'homography': None,
            'corners_xy': None, 'template_points_xy': np.empty((0, 2), np.float32),
            'scene_points_xy': np.empty((0, 2), np.float32),
            'inliers': np.empty(0, bool), 'match_count': 0, 'inlier_count': 0,
            'inlier_ratio': 0., 'reprojection_rms': None,
            'template_coverage': 0., 'scene_keypoint_count': scene_keypoints}


def _project(points, homography):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    denominator = homogeneous[:, 2]
    if (not np.isfinite(homogeneous).all() or
            np.any(np.abs(denominator) <= 1e-10)):
        return None
    return homogeneous[:, :2] / denominator[:, None]


def _corners(homography, shape):
    height, width = shape
    corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]],
                       np.float64)
    denominator = corners @ homography[2, :2] + homography[2, 2]
    # A pole inside the rectangle makes extrapolated template bounds meaningless.
    if not (np.all(denominator > 1e-10) or np.all(denominator < -1e-10)):
        return None
    projected = _project(corners, homography)
    if projected is None or np.max(np.abs(projected)) > 1e8:
        return None
    contour = projected.astype(np.float32)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 1.:
        return None
    return projected


def locate_planar_template(image, model, *, ratio=.75, mutual=True,
                           ransac_threshold=3., min_matches=12,
                           min_inliers=10, min_inlier_ratio=.35,
                           min_template_coverage=.02, confidence=.995,
                           max_iterations=2000):
    """Locate one textured plane under rotation, scale and moderate perspective.

    ORB + Hamming two-neighbor ratio + optional reverse-nearest check + RANSAC.
    ``ratio`` is a descriptor distance test; ``inlier_ratio`` is a diagnostic,
    neither is a probability that the object is present. The returned boolean
    inliers index the returned matched point arrays. Reprojection RMS and RANSAC
    threshold use scene pixels. Corners are TL, TR, BR, BL pixel centers.

    Invalid inputs raise ValueError. Ordinary detection failures return
    success=False and a reason; homography/corners stay None. Coverage is the
    inlier convex hull area divided by template pixel-center rectangle area;
    this rejects poorly constrained fits concentrated in a tiny patch.
    Template descriptors are reused, scene features are extracted on each call.
    Different calls may safely share one unmodified model.
    """
    if not isinstance(model, OrbTemplate):
        raise ValueError('model must be built by create_orb_template')
    if (not math.isfinite(ratio) or not 0 < ratio < 1 or type(mutual) is not bool or
            not math.isfinite(ransac_threshold) or not 0 < ransac_threshold <= 100 or
            type(min_matches) is not int or not 4 <= min_matches <= 10000 or
            type(min_inliers) is not int or not 4 <= min_inliers <= 10000 or
            not math.isfinite(min_inlier_ratio) or not 0 < min_inlier_ratio <= 1 or
            not math.isfinite(min_template_coverage) or not 0 < min_template_coverage <= 1 or
            not math.isfinite(confidence) or not 0 < confidence < 1 or
            type(max_iterations) is not int or not 1 <= max_iterations <= 100000):
        raise ValueError('Invalid matching or RANSAC parameters')
    scene_points, scene_descriptors = _features(_detector(model), _image_u8(image))
    result = _empty_result('insufficient_scene_features', len(scene_points))
    if scene_descriptors is None or len(scene_points) < 4:
        return result
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates = [pair[0] for pair in matcher.knnMatch(model.descriptors, scene_descriptors, k=2)
                  if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    if mutual and candidates:
        # Reverse nearest-neighbor decisions are independent for each query.
        # Only forward survivors can be retained: querying other scene rows
        # cannot affect their answer, including descriptor-distance ties.
        scene_indices = sorted({match.trainIdx for match in candidates})
        reverse = matcher.match(scene_descriptors[scene_indices], model.descriptors)
        reverse_indices = {scene_indices[match.queryIdx]: match.trainIdx for match in reverse}
        candidates = [match for match in candidates
                      if reverse_indices.get(match.trainIdx) == match.queryIdx]
    # Even without reverse matching, permit one correspondence per scene point.
    unique = []
    used_scene, used_template = set(), set()
    # Apply the same float32 rounding in bulk, avoiding two NumPy dispatches
    # per match while keeping candidate order and tie-breaking unchanged.
    scene_keys = np.round(scene_points, 1)
    template_keys = np.round(model.points_xy, 1)
    for match in sorted(candidates, key=lambda match: match.distance):
        scene_key = tuple(scene_keys[match.trainIdx])
        template_key = tuple(template_keys[match.queryIdx])
        if scene_key in used_scene or template_key in used_template:
            continue
        used_scene.add(scene_key)
        used_template.add(template_key)
        unique.append(match)
    source = np.array([model.points_xy[m.queryIdx] for m in unique],
                       np.float32).reshape(-1, 2)
    target = np.array([scene_points[m.trainIdx] for m in unique],
                       np.float32).reshape(-1, 2)
    result.update(template_points_xy=source, scene_points_xy=target,
                  match_count=len(unique), inliers=np.zeros(len(unique), bool))
    if len(unique) < min_matches:
        result['reason'] = 'insufficient_matches'
        return result
    if not _spread(source) or not _spread(target):
        result['reason'] = 'degenerate_matches'
        return result
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC,
                                        ransac_threshold, maxIters=max_iterations,
                                        confidence=confidence)
    if homography is None or mask is None or not np.isfinite(homography).all():
        result['reason'] = 'homography_failed'
        return result
    if abs(homography[2, 2]) <= 1e-12:
        result['reason'] = 'invalid_projection'
        return result
    homography = homography / homography[2, 2]
    projection = _project(source, homography)
    if projection is None:
        result['reason'] = 'invalid_projection'
        return result
    errors = np.linalg.norm(projection - target, axis=1)
    # Recheck after OpenCV's final refinement, rather than reporting stale inliers.
    inliers = mask.ravel().astype(bool) & (errors <= ransac_threshold)
    count = int(inliers.sum())
    result.update(inliers=inliers, inlier_count=count, inlier_ratio=count/len(source))
    if count:
        result['reprojection_rms'] = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    if count < min_inliers or result['inlier_ratio'] < min_inlier_ratio:
        result['reason'] = 'insufficient_inliers'
        return result
    if not _spread(source[inliers]) or not _spread(target[inliers]):
        result['reason'] = 'degenerate_inliers'
        return result
    height, width = model.shape
    coverage = float(cv2.contourArea(cv2.convexHull(source[inliers])) / ((width-1)*(height-1)))
    result['template_coverage'] = coverage
    if coverage < min_template_coverage:
        result['reason'] = 'insufficient_template_coverage'
        return result
    corners = _corners(homography, model.shape)
    if corners is None:
        result['reason'] = 'invalid_projection'
        return result
    result.update(success=True, reason='ok', homography=homography, corners_xy=corners)
    return result
