import numpy as np
from sklearn.cluster import DBSCAN
from math import atan2, degrees
from config import DBSCAN_EPS if hasattr(__import__('config'), 'DBSCAN_EPS') else 0.8, DBSCAN_MIN_SAMPLES if hasattr(__import__('config'), 'DBSCAN_MIN_SAMPLES') else 5

# Simple DBSCAN-based clustering + centroid extractor

def cluster_point_cloud(points, eps=0.8, min_samples=5):
    if points.shape[0] < min_samples:
        return []
    xy = points[:, :2]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
    labels = clustering.labels_
    clusters = {}
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        clusters.setdefault(lab, []).append(points[i])
    return list(clusters.values())


def extract_object_info(cluster):
    pts = np.array(cluster)
    centroid = np.mean(pts[:, :3], axis=0)
    distance = float(np.linalg.norm(centroid[:2]))
    z_mean = float(np.mean(pts[:, 2]))
    if z_mean > 1.2:
        obj_type = 'vehicle'
    elif z_mean > 0.4:
        obj_type = 'bicycle'
    else:
        obj_type = 'pedestrian'
    bearing = float(degrees(atan2(centroid[1], centroid[0])))
    return {
        'type': obj_type,
        'distance_m': distance,
        'relative_speed_mps': 0.0,  # updated by tracker
        'bearing_deg': bearing,
        'centroid': [float(x) for x in centroid]
    }
