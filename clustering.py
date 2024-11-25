import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def apply_dbscan(pcd, eps=0.6, min_points=11):
    """DBSCAN 클러스터링 수행"""
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters.")
    return labels, max_label


def colorize_clusters(pcd, labels, max_label):
    """클러스터별 색상 적용"""
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # 노이즈를 검정색으로 처리
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd


def cluster_and_colorize(pcd, eps=0.6, min_points=11):
    """DBSCAN 클러스터링과 색상 적용"""
    labels, max_label = apply_dbscan(pcd, eps=eps, min_points=min_points)
    return colorize_clusters(pcd, labels, max_label)