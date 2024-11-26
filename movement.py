import numpy as np

# 전역 변수: 이전 프레임의 클러스터 정보 저장
previous_centroids = None

def calculate_movement(pcd, labels):
    """
    프레임 간 클러스터의 이동 거리 계산.
    Args:
        pcd (o3d.geometry.PointCloud): 현재 프레임의 포인트 클라우드.
        labels (np.ndarray): 현재 프레임의 클러스터 레이블.
    Returns:
        움직임이 감지된 클러스터의 레이블 리스트.
    """
    global previous_centroids
    
    current_centroids = []
    moving_clusters = []
    
    # 현재 클러스터의 중심 위치 계산
    for cluster_label in range(labels.max() + 1):
        cluster_indices = np.where(labels == cluster_label)[0]
        if len(cluster_indices) > 0:
            cluster_pcd = pcd.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            centroid = points.mean(axis=0)  # 클러스터 중심 좌표
            current_centroids.append((cluster_label, centroid))
    
    # 이전 프레임과 비교
    if previous_centroids is not None:
        for curr_label, curr_centroid in current_centroids:
            for prev_label, prev_centroid in previous_centroids:
                if curr_label == prev_label:
                    distance = np.linalg.norm(curr_centroid - prev_centroid)
                    if distance > 0.1:  # 특정 거리 이상 이동한 클러스터만 선택
                        moving_clusters.append(curr_label)
    
    # 현재 클러스터 정보를 저장
    previous_centroids = current_centroids
    return moving_clusters
