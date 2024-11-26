import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

root_dir = "data"

def process_pcd_for_rendering(pcd, labels, point_size=1.0,
                              min_points=5, max_points=40,
                              min_z=-1.5, max_z=2.5, min_height=0.5, max_height=2.0,
                              max_distance=30.0):
    """
    클러스터에 색상을 적용하고 바운딩 박스를 생성하여 렌더링 준비.
    Args:
        pcd (o3d.geometry.PointCloud): 입력 포인트 클라우드.
        labels (np.ndarray): 클러스터 레이블.
        point_size (float): 포인트 크기.
        기타 클러스터 필터링 조건 (바운딩 박스 생성용).
    Returns:
        tuple: 색상이 적용된 pcd와 바운딩 박스 리스트.
    """
    # 클러스터 시각화 (색상 적용)
    print(f"Unique labels: {np.unique(labels)}")
    total_points = len(labels)
    noise_points = np.sum(labels == -1)
    print(f"Total points: {total_points}, Noise points: {noise_points}, Noise ratio: {noise_points / total_points:.2%}")

    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # 노이즈를 검정색으로 처리
    rgb_colors = colors[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

    # 바운딩 박스 생성
    bounding_boxes = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points <= len(cluster_indices) <= max_points:
            cluster_pcd = pcd.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()
            if min_z <= z_min and z_max <= max_z:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    distances = np.linalg.norm(points, axis=1)
                    if distances.max() <= max_distance:
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        bbox.color = (1, 0, 0)  # 빨간색 바운딩 박스
                        bounding_boxes.append(bbox)

    return pcd, bounding_boxes
