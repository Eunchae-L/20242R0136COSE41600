import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

# 이전 상태 저장 변수
previous_clusters = {}  # {cluster_id: centroid}
previous_bbox = None
cluster_movements = defaultdict(float)  # 클러스터 이동 누적량

def process_pcd_for_rendering(pcd, labels, point_size=1.0, min_points=10, max_clusters=10, movement_threshold=0.5):
    """
    PCD 처리 및 Bounding Box 생성
    """
    print(f"Processing {len(np.unique(labels))} clusters")
    
    # 클러스터 색상 시각화
    colors = plt.get_cmap("tab20")(labels / max(labels.max(), 1))
    colors[labels < 0] = 0  # 노이즈는 검정색
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    global previous_clusters, previous_bbox, cluster_movements

    # 현재 프레임 클러스터 중심점 계산
    current_clusters = {}
    for cluster_label in range(labels.max() + 1):
        cluster_indices = np.where(labels == cluster_label)[0]
        if len(cluster_indices) >= min_points:  # 최소 포인트 수 필터링
            cluster_pcd = pcd.select_by_index(cluster_indices)
            centroid = np.mean(np.asarray(cluster_pcd.points), axis=0)
            current_clusters[cluster_label] = (centroid, cluster_pcd)

    # 클러스터 개수 제한
    if len(current_clusters) > max_clusters:
        sorted_clusters = sorted(current_clusters.items(), key=lambda x: len(x[1][1].points), reverse=True)
        current_clusters = dict(sorted_clusters[:max_clusters])

    # 프레임 간 클러스터 매칭 (헝가리안 알고리즘)
    matched_clusters = {}
    if previous_clusters:
        current_ids = list(current_clusters.keys())
        previous_ids = list(previous_clusters.keys())
        
        # 거리 행렬 계산
        cost_matrix = np.zeros((len(previous_ids), len(current_ids)))
        for i, prev_id in enumerate(previous_ids):
            for j, curr_id in enumerate(current_ids):
                cost_matrix[i, j] = np.linalg.norm(previous_clusters[prev_id] - current_clusters[curr_id][0])
        
        # 최소 비용 매칭
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            prev_id = previous_ids[r]
            curr_id = current_ids[c]
            matched_clusters[prev_id] = current_clusters[curr_id]
    else:
        # 이전 클러스터가 없는 첫 번째 프레임 처리
        matched_clusters = {label: data for label, data in current_clusters.items()}

    # 업데이트된 클러스터 저장
    previous_clusters = {label: data[0] for label, data in matched_clusters.items()}

    # 모든 클러스터에 대해 Bounding Box 생성
    bounding_boxes = []
    for label, (centroid, cluster_pcd) in matched_clusters.items():
        if label in previous_clusters:
            # 이동량 누적 계산
            movement = np.linalg.norm(centroid - previous_clusters[label])
            cluster_movements[label] += movement  # 누적 이동량 업데이트
            print(f"Cluster {label}: Movement = {movement:.2f}, Cumulative Movement = {cluster_movements[label]:.2f}")

            # Bounding Box 생성
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            if cluster_movements[label] > movement_threshold:  # 이동량이 기준 초과인 경우
                bbox.color = (1, 0, 0)  # 빨간색
                bounding_boxes.append(bbox)

    # 최종적으로 Bounding Box 유지
    if bounding_boxes:
        previous_bbox = bounding_boxes[0]  # 첫 번째 Bounding Box를 유지
        return pcd, bounding_boxes

    # Bounding Box가 없으면 이전 상태 유지
    if previous_bbox:
        print("No significant movement detected. Retaining previous bounding box.")
        return pcd, [previous_bbox]

    return pcd, []