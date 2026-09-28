"""Wall-dominant point cloud with a constrained ground and obstacles."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.ground_ops import segment_ground_obstacles
from operators.image_io import write_image


def make_scene(seed=211, ground_count=3000, wall_count=6000, obstacle_count=1000):
    rng = np.random.default_rng(seed)
    ground = np.column_stack((rng.uniform(-2, 2, (ground_count, 2)),
                              rng.normal(0, .003, ground_count)))
    wall = np.column_stack((1.4+rng.normal(0, .003, wall_count),
                            rng.uniform(-2, 2, wall_count),
                            rng.uniform(.1, 1.5, wall_count)))
    objects = rng.uniform([-.7, -.8, .2], [.7, .8, .85], (obstacle_count, 3))
    return dict(points=np.vstack((ground, wall, objects)), ground_count=ground_count,
                wall_count=wall_count, obstacle_count=obstacle_count)


def evaluate(data):
    result = segment_ground_obstacles(data['points'], expected_ground_height=0.,
                                      height_tolerance=.08, max_tilt_deg=8.,
                                      distance_threshold=.012, min_ground_points=100,
                                      min_ground_ratio=.1, seed=31)
    assert result['success'], result['reason']
    ground_count, wall_count = data['ground_count'], data['wall_count']
    ground_recall = float(result['ground'][:ground_count].mean())
    wall_false_ground = float(result['ground'][ground_count:ground_count+wall_count].mean())
    obstacle_recall = float(result['obstacles'][ground_count:].mean())
    normal_error = float(np.degrees(np.arccos(np.clip(result['normal']@[0, 0, 1], -1, 1))))
    height_error = float(abs(result['ground_height_at_origin']))
    assert ground_recall > .99 and wall_false_ground == 0 and obstacle_recall > .99
    assert normal_error < .1 and height_error < .001
    checks = dict(ground_recall=ground_recall, wall_false_ground_fraction=wall_false_ground,
                  obstacle_recall=obstacle_recall, plane_normal_error_deg=normal_error,
                  ground_height_error_m=height_error, ground_rms_m=result['rms'],
                  iterations=result['iterations'], wall_points=wall_count,
                  floor_points=ground_count)
    return result, checks


def render_cloud(data, result):
    image = np.full((500, 1000, 3), 22, np.uint8)
    points = data['points']
    colors = np.full((len(points), 3), [90, 90, 90], np.uint8)
    colors[result['ground']] = [65, 210, 65]
    colors[result['obstacles']] = [50, 85, 240]
    colors[result['below']] = [220, 120, 50]
    for panel, axes in enumerate(((0, 1), (0, 2))):
        origin = np.array([250+500*panel, 250])
        projected = np.rint(points[:, axes]*100+origin).astype(int)
        for (x, y), color in zip(projected, colors):
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (int(x), int(y)), 1, tuple(int(c) for c in color), -1)
        cv2.putText(image, 'XY' if panel == 0 else 'XZ', (20+panel*500, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (230, 230, 230), 1)
    cv2.putText(image, 'Green: ground | Red: obstacle', (28, 472),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (230, 230, 230), 1)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/ground_obstacles')
    args = parser.parse_args()
    data = make_scene()
    result, checks = evaluate(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'cloud.png', render_cloud(data, result))
    report = dict(scope='Synthetic metric scene with 2x as many wall as floor points. Geometric obstacle classification, not traversability.',
                  checks=checks, normal=result['normal'].tolist(), offset=result['offset'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Gravity-constrained ground</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Ground and obstacles</h1><p>6,000 wall points versus 3,000 floor points. Up direction and a height prior determine the ground candidate.</p><img src="cloud.png">',
        encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
