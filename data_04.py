import open3d as o3d
import numpy as np
import os
import cv2
from sklearn.cluster import DBSCAN
from collections import defaultdict

# PCD 파일 로드
def load_pcd_files(pcd_dir):
    pcd_files = sorted([os.path.join(pcd_dir, file) for file in os.listdir(pcd_dir) if file.endswith('.pcd')])
    return pcd_files

# Point Cloud 전처리: SOR, ROR, 다운샘플링, 클러스터링
def preprocess_point_cloud(pcd, voxel_size=0.3, eps=0.2, min_samples=10):
    # SOR (Statistical Outlier Removal)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # ROR (Radius Outlier Removal)
    pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=1.2)

    # 다운샘플링
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # RANSAC을 사용하여 평면 추정
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)

    # 평면에 속하지 않는 포인트 추출
    pcd = pcd.select_by_index(inliers, invert=True)

    # 클러스터링
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    return pcd, labels

# 축 정렬을 통한 배경 안정화
def align_point_cloud(pcd, reference_pcd):
    # ICP (Iterative Closest Point) 정렬로 축 정렬
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd, reference_pcd, max_correspondence_distance=1.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation = reg_p2p.transformation
    pcd.transform(transformation)
    return pcd

# 모션 기반 클러스터 식별
movement_vectors = defaultdict(list)

def is_person_by_motion(cluster_id, current_centroid, direction_threshold=0.6, N=5):
    if cluster_id not in movement_vectors:
        movement_vectors[cluster_id].append(current_centroid)
        return False

    previous_centroid = movement_vectors[cluster_id][-1]
    movement_vector = current_centroid - previous_centroid
    movement_vectors[cluster_id].append(current_centroid)

    if len(movement_vectors[cluster_id]) > N:
        movement_vectors[cluster_id].pop(0)

    if len(movement_vectors[cluster_id]) >= N:
        avg_direction = np.mean(np.diff(movement_vectors[cluster_id], axis=0), axis=0)
        if np.linalg.norm(avg_direction) > direction_threshold:
            return True

    return False

# PCD 렌더링 및 동영상 저장
def render_pcd_and_save_video(pcd_files, output_dir, video_name, frame_selection_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.get_render_option().point_size = 2.0

    frame_width, frame_height = 1280, 720
    fps = 30 // frame_selection_step
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}.mp4"), fourcc, fps, (frame_width, frame_height))

    # 기준 축을 정렬하기 위한 첫 번째 PCD
    reference_pcd = o3d.io.read_point_cloud(pcd_files[0])

    for idx, pcd_file in enumerate(pcd_files):
        if idx % frame_selection_step != 0:
            continue
        print(f"Processing frame {idx + 1}/{len(pcd_files)}: {pcd_file}")
        # PCD 로드 및 전처리
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd = align_point_cloud(pcd, reference_pcd)  # 축 정렬
        pcd, labels = preprocess_point_cloud(pcd)

        # 클러스터별 Bounding Box 생성
        cluster_ids = np.unique(labels)
        bounding_boxes = []
        for cluster_id in cluster_ids:
            if cluster_id == -1:
                continue  # 노이즈는 건너뛰기

            cluster_points = np.asarray(pcd.points)[labels == cluster_id]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

            current_centroid = np.mean(cluster_points, axis=0)

            if is_person_by_motion(cluster_id, current_centroid):
                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                expand_size = np.array([1,1,1])  # 각 축 방향으로 0.5씩 확장

                # min_bound와 max_bound 복사 후 수정
                new_min_bound = bbox.min_bound - expand_size
                new_max_bound = bbox.max_bound + expand_size

                # 수정된 값을 다시 할당
                bbox.min_bound = new_min_bound
                bbox.max_bound = new_max_bound

                bbox.color = (1, 0, 0)
                bounding_boxes.append(bbox)

        # 렌더링
        vis.add_geometry(pcd)
        for bbox in bounding_boxes:
            vis.add_geometry(bbox)

        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 프레임 저장
        video_writer.write(frame)
        vis.clear_geometries()

    vis.destroy_window()
    video_writer.release()
    print(f"Video saved to {os.path.join(output_dir, video_name)}.mp4")

# 시나리오 설정 및 실행
scenario = "04_zigzag_walk"
input_root_dir = "./data"
output_root_dir = "./output"
os.makedirs(output_root_dir, exist_ok=True)

print(f"Processing scenario: {scenario}")
pcd_dir = os.path.join(input_root_dir, scenario, 'pcd')
output_dir = os.path.join(output_root_dir, scenario)
os.makedirs(output_dir, exist_ok=True)

pcd_files = load_pcd_files(pcd_dir)
render_pcd_and_save_video(pcd_files, output_dir, scenario, frame_selection_step=10)