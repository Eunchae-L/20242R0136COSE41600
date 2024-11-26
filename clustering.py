import open3d as o3d
import numpy as np

def apply_dbscan(pcd, eps, min_points):
    """DBSCAN 클러스터링 수행"""
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    print(f"clustering complete")
    return labels