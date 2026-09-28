"""Known-truth distortion-aware stereo depth with deliberately wrong matches."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from operators.image_io import write_image
from operators.robot_ops import Intrinsics
from operators.stereo_ops import triangulate_stereo


def make_scene(seed=58, count=2000, mismatches=120):
    rng = np.random.default_rng(seed)
    truth = rng.uniform([-.45, -.3, 1.5], [.45, .3, 4.], (count, 3))
    camera = Intrinsics(1280, 960, 900., 910., 640., 480.)
    matrix = np.array([[900., 0., 640.], [0., 910., 480.], [0., 0., 1.]])
    distortion = np.array([-.13, .025, .001, -.002, .004])
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.array([.008, -.013, .005]))[0]
    pose[:3, 3] = [-.18, .004, .002]
    right_truth = truth @ pose[:3, :3].T + pose[:3, 3]
    left = cv2.projectPoints(truth, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
    right = cv2.projectPoints(right_truth, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
    left += rng.normal(0., .1, left.shape)
    right += rng.normal(0., .1, right.shape)
    right[:mismatches, 1] += 40.
    return dict(truth=truth, left=left, right=right, camera=camera, pose=pose,
                distortion=distortion, mismatches=mismatches)


def evaluate(data):
    result = triangulate_stereo(data['left'], data['right'], data['camera'], data['camera'],
                                data['pose'], distortion_left=data['distortion'],
                                distortion_right=data['distortion'])
    bad_accepted = int(result['valid'][:data['mismatches']].sum())
    correct_mask = result['valid'][data['mismatches']:]
    errors = np.linalg.norm(result['points_left'][data['mismatches']:][correct_mask]-
                            data['truth'][data['mismatches']:][correct_mask], axis=1)
    assert bad_accepted == 0
    assert correct_mask.mean() > .99
    assert np.median(errors) < .006
    checks = dict(correspondences=len(data['left']), deliberate_mismatches=data['mismatches'],
                  wrong_matches_accepted=bad_accepted, correct_recall=float(correct_mask.mean()),
                  median_3d_error_m=float(np.median(errors)), p95_3d_error_m=float(np.percentile(errors, 95)),
                  valid_count=result['valid_count'])
    return result, checks


def visual(data, result):
    image = np.full((960, 1280, 3), 22, np.uint8)
    for i, (x, y) in enumerate(data['left']):
        if not 0 <= x < 1280 or not 0 <= y < 960:
            continue
        color = (50, 210, 75) if result['valid'][i] else (70, 70, 235)
        cv2.circle(image, (round(x), round(y)), 2, color, -1)
    cv2.putText(image, 'Green: triangulated   Red: rejected', (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (230, 230, 230), 2)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/stereo')
    args = parser.parse_args()
    data = make_scene()
    result, checks = evaluate(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'stereo_points.png', visual(data, result))
    (args.output/'report.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Stereo triangulation</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Calibrated stereo depth</h1><p>Known 3D truth and deliberately wrong vertical matches.'
        ' Green points pass two-camera geometry; red points are rejected.</p><img src="stereo_points.png">',
        encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
