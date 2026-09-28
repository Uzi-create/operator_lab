"""Three unorganized point clusters with sparse clutter and voxel connectivity."""
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
from operators.voxel_cluster_ops import voxel_clusters


def make_scene(seed=613, per_object=6000, outliers=500):
    rng = np.random.default_rng(seed)
    centers = np.array([[-1., 0., 0.], [1., 0., 0.], [0., 1.5, .5]])
    groups = [center+rng.normal(0., .13, (per_object, 3)) for center in centers]
    clutter = rng.uniform([-2.5, -1.5, -1.], [2.5, 2.5, 1.5], (outliers, 3))
    return dict(points=np.vstack((*groups, clutter)), true_centers=centers,
                per_object=per_object, outliers=outliers)


def evaluate(data, backend='native'):
    result = voxel_clusters(data['points'], .09, connectivity=26,
                            min_voxels=8, min_points=100, backend=backend)
    assert result['cluster_count'] == 3, result['cluster_count']
    labels = result['labels']
    mapping = []
    recalls = []
    center_errors = []
    for i, true_center in enumerate(data['true_centers']):
        assigned = labels[i*data['per_object']:(i+1)*data['per_object']]
        majority = int(np.bincount(assigned).argmax())
        assert majority > 0
        recall = float(np.mean(assigned == majority))
        error = float(np.linalg.norm(result['clusters'][majority-1]['centroid']-true_center))
        assert recall > .985 and error < .015, (recall, error)
        mapping.append(majority)
        recalls.append(recall)
        center_errors.append(error)
    assert len(set(mapping)) == 3
    false_accepted = int(np.count_nonzero(labels[-data['outliers']:]))
    assert false_accepted < .1*data['outliers']
    checks = dict(cluster_count=3, true_cluster_recalls=recalls,
                  centroid_errors_m=center_errors,
                  noise_points_accepted=false_accepted,
                  cluster_point_counts=[item['point_count'] for item in result['clusters']],
                  occupied_voxels=len(result['voxel_keys']))
    return result, checks


def render(data, result):
    canvas = np.full((480, 960, 3), 23, np.uint8)
    palette = np.array([[105, 105, 105], [50, 200, 70], [65, 190, 235],
                        [235, 145, 60]], np.uint8)
    for panel, axes in enumerate(((0, 1), (0, 2))):
        xy = np.rint(data['points'][:, axes]*95+[240+480*panel, 240]).astype(int)
        for (x, y), label in zip(xy, result['labels']):
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.circle(canvas, (int(x), int(y)), 1, tuple(int(c) for c in palette[label]), -1)
        cv2.putText(canvas, 'XY' if panel == 0 else 'XZ', (20+480*panel, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (230, 230, 230), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/voxel_clusters')
    parser.add_argument('--backend', choices=('native', 'numpy', 'auto'), default='native')
    args = parser.parse_args()
    data = make_scene()
    result, checks = evaluate(data, backend=args.backend)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output/'clusters.png', render(data, result))
    report = dict(scope='Synthetic unorganized 3D Gaussian clusters plus uniform clutter. Voxel adjacency is not an exact Euclidean-radius criterion.',
                  backend=result['backend'], checks=checks,
                  clusters=[{'label': item['label'], 'point_count': item['point_count'],
                             'voxel_count': item['voxel_count'],
                             'centroid': item['centroid'].tolist()} for item in result['clusters']])
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Voxel cluster graph</title>'
        '<style>body{background:#17232e;color:#eef;font:18px system-ui;margin:28px}img{max-width:100%}</style>'
        '<h1>Unorganized cloud voxel clusters</h1><p>Three colored clusters; gray indicates points filtered by minimum region size.</p><img src="clusters.png">',
        encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
