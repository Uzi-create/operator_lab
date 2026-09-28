"""Local 6-DoF point-to-plane ICP using exact target nearest neighbors."""
import math
from numbers import Real

import cv2
import numpy as np

from operators.registration_ops import NearestNeighborIndex, _points, _pose


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _normals(values, count):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape != (count, 3) or values.dtype.kind not in 'uif':
        raise ValueError('target_normals must be a real target-count Nx3 array')
    values = np.ascontiguousarray(values, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or not np.isfinite(lengths).all() or (lengths <= 1e-12).any():
        raise ValueError('Every target normal must be finite and nonzero')
    return values/lengths[:, None]


def _state(index, target, normals, transformed, max_distance, max_plane_residual,
           trim_fraction, min_correspondences, min_overlap):
    neighbors = index.query(transformed, max_distance=max_distance)
    indices = neighbors['indices']
    residuals = np.full(len(transformed), np.inf)
    valid = neighbors['valid']
    selected = np.flatnonzero(valid)
    if len(selected):
        residuals[selected] = np.einsum('ij,ij->i',
            transformed[selected]-target[indices[selected]], normals[indices[selected]])
    selected = selected[np.abs(residuals[selected]) <= max_plane_residual]
    overlap = len(selected)/len(transformed)
    if len(selected) < min_correspondences or overlap < min_overlap:
        raise ValueError('Insufficient distance-gated plane correspondences')
    keep = math.ceil(trim_fraction*len(selected))
    if keep < min_correspondences:
        raise ValueError('Too few plane correspondences after trimming')
    if keep < len(selected):
        order = np.argsort(np.abs(residuals[selected]), kind='stable')[:keep]
        selected = selected[order]
    inliers = np.zeros(len(transformed), bool)
    inliers[selected] = True
    rms = float(np.sqrt(np.mean(residuals[selected]**2)))
    return dict(indices=indices, inliers=inliers, plane_residuals=residuals,
                point_distances=neighbors['distances'], overlap=overlap, rms=rms)


def _jacobian(transformed, normals, state):
    chosen = np.flatnonzero(state['inliers'])
    x = transformed[chosen]
    n = normals[state['indices'][chosen]]
    center = x.mean(axis=0)
    jacobian = np.column_stack((np.cross(x-center, n), n))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular[-1] <= singular[0]*1e-7:
        raise ValueError('Point-to-plane geometry does not constrain all six pose freedoms')
    return jacobian, center, chosen


def icp_point_to_plane(source, target, target_normals, *, max_distance,
                       max_plane_residual=None, initial_transform=None,
                       trim_fraction=1., min_correspondences=20,
                       min_overlap=.1, max_iterations=40,
                       translation_tolerance=1e-6, rotation_tolerance=1e-6,
                       backend='auto'):
    """Rigid point-to-plane ICP, final-pose correspondence and RMS checks.

    Target normals must align one-for-one with target/index target_points.
    Transform is T_target_from_source. Exact nearest-neighbor lookup uses the
    existing native KD tree or NumPy reference backend. This is a local solver,
    requiring a useful initial pose and multiple independent surface normals.
    A single plane cannot constrain all six freedoms and is rejected.
    """
    source = _points(source, 3)
    if backend not in ('auto', 'native', 'numpy'):
        raise ValueError('backend must be auto, native or numpy')
    max_distance = _real(max_distance, 'max_distance')
    max_plane_residual = (max_distance if max_plane_residual is None else
                          _real(max_plane_residual, 'max_plane_residual'))
    trim_fraction = _real(trim_fraction, 'trim_fraction')
    min_overlap = _real(min_overlap, 'min_overlap')
    translation_tolerance = _real(translation_tolerance, 'translation_tolerance')
    rotation_tolerance = _real(rotation_tolerance, 'rotation_tolerance')
    if (max_distance <= 0 or max_plane_residual <= 0 or max_plane_residual > max_distance or
            not 0 < trim_fraction <= 1 or not 0 < min_overlap <= 1 or
            translation_tolerance <= 0 or rotation_tolerance <= 0):
        raise ValueError('Invalid distance, trim, overlap or tolerance')
    if (type(min_correspondences) is not int or min_correspondences < 6 or
            type(max_iterations) is not int or not 1 <= max_iterations <= 10000 or
            len(source) < min_correspondences):
        raise ValueError('Invalid minimum correspondences or iteration count')
    transform = _pose(initial_transform)
    owned = not isinstance(target, NearestNeighborIndex)
    index = NearestNeighborIndex(target, backend=backend) if owned else target
    try:
        if index._closed:
            raise RuntimeError('Nearest-neighbor index is closed')
        target_points = index._points
        normals = _normals(target_normals, len(target_points))
        transformed = source@transform[:3, :3].T+transform[:3, 3]
        state = _state(index, target_points, normals, transformed, max_distance,
                       max_plane_residual, trim_fraction, min_correspondences, min_overlap)
        history = []
        converged = False
        status = 'max_iterations'
        for iteration in range(1, max_iterations+1):
            jacobian, center, chosen = _jacobian(transformed, normals, state)
            residuals = state['plane_residuals'][chosen]
            delta = np.linalg.lstsq(jacobian, -residuals, rcond=None)[0]
            if not np.isfinite(delta).all():
                raise ValueError('Point-to-plane least-squares update is nonfinite')
            old_loss = float(np.mean(residuals**2))
            accepted = False
            for alpha in (1., .5, .25, .125, .0625):
                rotation = cv2.Rodrigues(alpha*delta[:3])[0]
                translation = alpha*delta[3:]
                candidate = (transformed-center)@rotation.T+center+translation
                new_residual = np.einsum('ij,ij->i',
                    candidate[chosen]-target_points[state['indices'][chosen]],
                    normals[state['indices'][chosen]])
                if float(np.mean(new_residual**2)) <= old_loss+1e-18:
                    accepted = True
                    break
            if not accepted:
                status = 'stalled'
                break
            update = np.eye(4)
            update[:3, :3] = rotation
            update[:3, 3] = center-rotation@center+translation
            transform = update@transform
            transformed = source@transform[:3, :3].T+transform[:3, 3]
            state = _state(index, target_points, normals, transformed, max_distance,
                           max_plane_residual, trim_fraction, min_correspondences, min_overlap)
            angle = float(np.linalg.norm(alpha*delta[:3]))
            motion = float(np.linalg.norm(alpha*delta[3:]))
            history.append(dict(iteration=iteration, rms=state['rms'],
                                inlier_count=int(state['inliers'].sum()),
                                overlap=state['overlap'], rotation_step=angle,
                                translation_step=motion))
            if angle <= rotation_tolerance and motion <= translation_tolerance:
                converged = True
                status = 'converged'
                break
        _jacobian(transformed, normals, state)
        return dict(transform=transform, transformed_source=transformed,
                    target_indices=state['indices'], inliers=state['inliers'],
                    plane_residuals=state['plane_residuals'],
                    point_distances=state['point_distances'], rms=state['rms'],
                    overlap=state['overlap'], inlier_fraction=float(state['inliers'].mean()),
                    iterations=len(history), converged=converged, status=status,
                    history=history, backend=index.backend)
    finally:
        if owned:
            index.close()
