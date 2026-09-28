"""End-to-end depth components, 3D backprojection and ground/obstacle masks."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.depth_components_ops import depth_components
from operators.box_ops import gravity_aligned_box
from operators.ground_ops import segment_ground_obstacles
from operators.image_io import write_image
from operators.robot_ops import Intrinsics, depth_to_points, transform_points


def make_scene(seed=328):
    rng = np.random.default_rng(seed)
    camera = Intrinsics(320, 240, 300., 300., 159.5, 119.5)
    y, x = np.indices((camera.height, camera.width))
    first = (35 <= x) & (x < 105) & (45 <= y) & (y < 122)
    second = (184 <= x) & (x < 272) & (125 <= y) & (y < 206)
    ground_depth_m = 1.5
    depth_m = np.where(first | second, 1.1, ground_depth_m)
    depth = np.rint((depth_m+rng.normal(0, .0015, depth_m.shape))*1000).astype(np.uint16)
    valid = rng.random(depth.shape) >= .01
    depth[~valid] = 0
    true_object = (first | second) & valid
    true_ground = ~(first | second) & valid
    T_world_from_camera = np.eye(4)
    T_world_from_camera[:3, :3] = np.diag([1., -1., -1.])
    T_world_from_camera[:3, 3] = [0., 0., ground_depth_m]
    return dict(depth_mm=depth, true_object=true_object, true_ground=true_ground,
                camera=camera, T_world_from_camera=T_world_from_camera)


def evaluate(data, *, backend='native'):
    components = depth_components(data['depth_mm'], depth_scale=.001, max_depth=3.,
                                  absolute_jump=.015, min_area=100, backend=backend)
    assert components['region_count'] == 3, components['regions']
    projected = depth_to_points(data['depth_mm'], data['camera'], depth_scale=.001,
                                max_depth=3., mask=components['labels'] > 0, stride=2)
    world_points = transform_points(projected['points'], data['T_world_from_camera'])
    point_labels = components['labels'][projected['pixels'][:, 1], projected['pixels'][:, 0]]
    boxes = []
    for label in (2, 3):
        surface = world_points[point_labels == label]
        box = gravity_aligned_box(surface, up=(0, 0, 1))
        assert box['success'], box['reason']
        local = (surface-box['center'])@box['axes']
        assert (abs(local) <= box['size']/2+1e-12).all()
        assert box['size'][2] < .015  # visible upper surface, not full solid object
        boxes.append(box)
    ground = segment_ground_obstacles(world_points, up=(0., 0., 1.),
                                      expected_ground_height=0., height_tolerance=.05,
                                      max_tilt_deg=5., distance_threshold=.012,
                                      obstacle_min_height=.05, obstacle_max_height=1.,
                                      min_ground_points=100, min_ground_ratio=.3, seed=51)
    assert ground['success'], ground['reason']
    uv = projected['pixels']
    truth_object = data['true_object'][uv[:, 1], uv[:, 0]]
    truth_ground = data['true_ground'][uv[:, 1], uv[:, 0]]
    false_ground = int(np.count_nonzero(ground['ground'] & ~truth_ground))
    missed_ground = int(np.count_nonzero(truth_ground & ~ground['ground']))
    false_obstacle = int(np.count_nonzero(ground['obstacles'] & ~truth_object))
    missed_obstacle = int(np.count_nonzero(truth_object & ~ground['obstacles']))
    assert not (false_ground or missed_ground or false_obstacle or missed_obstacle)
    normal_error = float(np.degrees(np.arccos(np.clip(ground['normal']@[0, 0, 1], -1, 1))))
    assert normal_error < .1
    checks = dict(components=3, sampled_points=len(world_points),
                  ground_points=int(ground['ground'].sum()),
                  obstacle_points=int(ground['obstacles'].sum()),
                  false_ground=false_ground, missed_ground=missed_ground,
                  false_obstacle=false_obstacle, missed_obstacle=missed_obstacle,
                  plane_normal_error_deg=normal_error,
                  ground_height_error_m=float(abs(ground['ground_height_at_origin'])),
                  ground_rms_m=ground['rms'],
                  visible_surface_boxes=[{'label': label, 'center_m': box['center'].tolist(),
                                          'size_m': box['size'].tolist(),
                                          'footprint_area_m2': box['footprint_area']}
                                         for label, box in zip((2, 3), boxes)])
    return dict(components=components, projected=projected, world_points=world_points,
                ground=ground, boxes=boxes), checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/rgbd_scene')
    parser.add_argument('--backend', choices=('native', 'numpy', 'auto'), default='native')
    args = parser.parse_args()
    data = make_scene()
    result, checks = evaluate(data, backend=args.backend)
    args.output.mkdir(parents=True, exist_ok=True)
    intensity = np.rint(np.clip(data['depth_mm']/2000, 0, 1)*255).astype(np.uint8)
    raw = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    raw[data['depth_mm'] == 0] = 0
    palette = np.array([[0, 0, 0], [130, 130, 130], [60, 210, 70], [60, 180, 240]], np.uint8)
    connected = palette[result['components']['labels']]
    classified = np.zeros_like(raw)
    uv = result['projected']['pixels']
    classified[uv[:, 1], uv[:, 0]] = np.where(result['ground']['ground'][:, None], [60, 210, 70],
                                              np.where(result['ground']['obstacles'][:, None],
                                                       [60, 180, 240], [100, 100, 100])).astype(np.uint8)
    write_image(args.output/'overview.png', cv2.hconcat((raw, connected, classified)))
    report = dict(scope='Synthetic aligned RGB-D-style depth, 1.5m downward-looking calibrated camera, two elevated objects. Boxes enclose visible surfaces only, not full hidden volumes. No color semantics or field validation.',
                  checks=checks, camera=dict(width=data['camera'].width, height=data['camera'].height,
                                              fx=data['camera'].fx, fy=data['camera'].fy),
                  transform=data['T_world_from_camera'].tolist())
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Depth to ground and objects</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Organized depth to 3D ground/obstacles</h1><p>Left: depth, middle: depth-jump components, right: sampled 3D points classified by ground height.</p><img src="overview.png">',
        encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
