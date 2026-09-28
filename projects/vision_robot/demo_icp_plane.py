"""Known-pose multi-surface point-to-plane registration with target normals."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.icp_plane_ops import icp_point_to_plane
from operators.image_io import write_image


def make_scene(seed=521, face_count=600, outliers=120):
    rng = np.random.default_rng(seed)
    faces, normals = [], []
    for axis in range(3):
        face = rng.uniform(.05, 1., (face_count, 3))
        face[:, axis] = 0.
        normal = np.zeros(3)
        normal[axis] = -1.
        faces.append(face)
        normals.append(np.tile(normal, (face_count, 1)))
    clean, normals = np.vstack(faces), np.vstack(normals)
    truth = np.eye(4)
    truth[:3, :3] = cv2.Rodrigues(np.array([.025, -.02, .015]))[0]
    truth[:3, 3] = [.02, -.014, .01]
    chosen = np.concatenate([np.arange(i*face_count, i*face_count+int(.8*face_count))
                             for i in range(3)])
    source = (clean[chosen]-truth[:3, 3])@truth[:3, :3]
    source += rng.normal(0, .0003, source.shape)
    source = np.vstack((source, rng.uniform(2., 3., (outliers, 3))))
    target = clean+rng.normal(0, .0003, clean.shape)
    return dict(source=source, target=target, normals=normals, truth=truth,
                outlier_count=outliers)


def evaluate(data):
    result = icp_point_to_plane(data['source'], data['target'], data['normals'],
                                max_distance=.09, trim_fraction=.9, min_overlap=.7)
    assert result['converged'], result['status']
    relative = result['transform'][:3, :3]@data['truth'][:3, :3].T
    angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(relative)[0])))
    distance = float(np.linalg.norm(result['transform'][:3, 3]-data['truth'][:3, 3]))
    rejected = int(result['inliers'][-data['outlier_count']:].sum())
    assert angle < .02 and distance < .0002 and rejected == 0
    checks = dict(rotation_error_deg=angle, translation_error_m=distance,
                  point_plane_rms_m=result['rms'], iterations=result['iterations'],
                  deliberate_outliers_accepted=rejected, overlap=result['overlap'])
    return result, checks


def visualize(data, result):
    canvas = np.full((480, 960, 3), 22, np.uint8)
    clouds = (data['target'], data['source'][:-data['outlier_count']],
              result['transformed_source'][:-data['outlier_count']])
    colors = ((140, 140, 140), (70, 90, 235), (65, 220, 70))
    for panel, axes in enumerate(((0, 1), (0, 2))):
        for cloud, color in zip(clouds, colors):
            xy = np.rint(cloud[:, axes]*250+[100+panel*480, 130]).astype(int)
            for x, y in xy:
                if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                    cv2.circle(canvas, (int(x), int(y)), 1, color, -1)
        cv2.putText(canvas, 'XY' if panel == 0 else 'XZ', (20+panel*480, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (230, 230, 230), 1)
    cv2.putText(canvas, 'Gray target | red source | green aligned', (25, 455),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/icp_plane')
    args = parser.parse_args()
    data = make_scene()
    result, checks = evaluate(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'registration.png', visualize(data, result))
    report = dict(scope='Synthetic three orthogonal observed faces with known normals, noisy partial overlap and distant outliers. Requires target normals and a near initial pose.',
                  checks=checks, T_target_from_source=result['transform'].tolist())
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Point-to-plane ICP</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Point-to-plane ICP</h1><p>Three independent surface normal directions constrain 6-DoF pose. Distant outliers are rejected.</p><img src="registration.png">',
        encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
