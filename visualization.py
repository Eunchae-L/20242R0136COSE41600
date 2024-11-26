import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

def process_pcd_for_rendering(pcd, labels, point_size=1.0, min_points=10, max_clusters=10):
    """
    PCD 처리 및 Bounding Box 생성

    Parameters:
        pcd (o3d.geometry.PointCloud): 포인트 클라우드 데이터.
        labels (np.ndarray): 클러스터 레이블 배열.
        point_size (float): 시각화 포인트 크기.
        min_points (int): 클러스터 내 최소 포인트 수.
        max_clusters (int): 최대 클러스터 개수.

    Returns:
        current_clusters (dict): 각 클러스터의 중심점과 포인트 클라우드.
    """
    print(f"Processing {len(np.unique(labels))} clusters")

    # 클러스터 색상 시각화
    colors = plt.get_cmap("tab20")(labels / max(labels.max(), 1))
    colors[labels < 0] = 0  # 노이즈는 검정색
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    current_clusters = {}
    for cluster_label in range(labels.max() + 1):
        cluster_indices = np.where(labels == cluster_label)[0]
        if len(cluster_indices) >= min_points:  # 최소 포인트 수 필터링
            cluster_pcd = pcd.select_by_index(cluster_indices)
            centroid = np.mean(np.asarray(cluster_pcd.points), axis=0)
            current_clusters[cluster_label] = (centroid, cluster_pcd)

    return current_clusters