from collections import defaultdict
import os
import open3d as o3d
import numpy as np
import gc
import cv2
from noise_removal import remove_noise
from downsampling import downsample
from ground_removal import remove_ground
from clustering import apply_dbscan

os.environ["OPEN3D_DEVICE"] = "CPU"
print(f"OPEN3D_DEVICE: {os.getenv('OPEN3D_DEVICE')}")

# 시나리오 디렉토리 설정
root_dir = "data"
scenario_name = "01_straight_walk"
scenario_path = os.path.join(root_dir, scenario_name)

if not os.path.exists(scenario_path):
    print(f"Scenario path does not exist: {scenario_path}")
    exit()

pcd_files = sorted(
    [os.path.join(scenario_path, f) for f in os.listdir(scenario_path) if f.endswith(".pcd")],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
)

video_path = '/Users/eunchaelin/Desktop/MyFolder/Korea University/4-2/AutonomousVehicle/HW1/project/video'
frame_width, frame_height = 1280, 720
fps = 10
frame_selection_step = 2  # 프레임 간격 조정

movement_vectors = defaultdict(list)

def align_point_cloud(pcd, reference_pcd):
    """참조 PCD를 기반으로 축 정렬"""
    transformation = o3d.pipelines.registration.registration_icp(
        pcd, reference_pcd, max_correspondence_distance=1.0,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    ).transformation
    pcd.transform(transformation)
    return pcd

def preprocess_point_cloud(pcd):
    """PCD 전처리 및 클러스터 라벨 반환"""
    # 다운샘플링
    pcd_downsampled = downsample(pcd, voxel_size=0.05)
    # 노이즈 제거
    pcd_cleaned = remove_noise(pcd_downsampled, sor_neighbors=15, sor_std_ratio=1.5, ror_points=10, ror_radius=0.15)
    # 도로 제거
    _, non_ground = remove_ground(pcd_cleaned, distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    # 클러스터링
    labels = apply_dbscan(non_ground, eps=0.5, min_points=10)
    return non_ground, labels

def is_person_by_motion(cluster_id, current_centroid, direction_threshold=0.8, N=10):
    """모션 기반으로 클러스터가 사람인지 확인"""
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

# 단일 시나리오 처리
print(f"Processing scenario: {scenario_name}")
video_output_path = os.path.join(video_path, f"{scenario_name}_output.mp4")
out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

reference_pcd = None
if pcd_files:
    reference_pcd = o3d.io.read_point_cloud(pcd_files[0])  # 첫 번째 PCD를 기준으로 축 정렬

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
            expand_size = np.array([1, 1, 1])  # 각 축 방향으로 1씩 확장

            # 확장된 바운딩 박스 설정
            new_min_bound = bbox.min_bound - expand_size
            new_max_bound = bbox.max_bound + expand_size

            bbox.min_bound = new_min_bound
            bbox.max_bound = new_max_bound
            bbox.color = (1, 0, 0)
            bounding_boxes.append(bbox)

    print(f"Number of bounding boxes: {len(bounding_boxes)}")

    # 비디오 프레임 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=frame_width, height=frame_height)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.poll_events()
    vis.update_renderer()

    frame = np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255
    vis.clear_geometries()
    vis.destroy_window()

    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
    out.write(frame)

out.release()
print(f"Scenario video saved: {video_output_path}")
gc.collect()