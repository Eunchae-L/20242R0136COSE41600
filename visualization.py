import open3d as o3d
import matplotlib.pyplot as plt

def visualize_clusters(pcd, labels):
    """클러스터를 시각화하고 색상을 적용"""
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def draw_bounding_boxes(pcd, labels):
    """각 클러스터에 대해 바운딩 박스를 생성"""
    unique_labels = set(labels)
    bounding_boxes = []
    for label in unique_labels:
        if label == -1:  # Noise 클러스터 제외
            continue
        cluster = pcd.select_by_index(np.where(labels == label)[0])
        bbox = cluster.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # 빨간색
        bounding_boxes.append(bbox)
    return bounding_boxes