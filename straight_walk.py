import os
import open3d as o3d
import cv2
import numpy as np
import gc
from noise_removal import remove_noise
from downsampling import downsample
from ground_removal import remove_ground
from clustering import apply_dbscan
from visualization import process_pcd_for_rendering
from scipy.optimize import linear_sum_assignment

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

def render_to_frame(pcd, bounding_boxes, width=1280, height=720):
    """PCD와 바운딩 박스를 렌더링하여 프레임으로 반환"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    # 고정된 뷰 설정
    view_control = vis.get_view_control()
    view_control.set_lookat([0, 0, 0])  # 카메라가 바라보는 중심점
    view_control.set_zoom(0.8)  # 고정된 줌 레벨
    view_control.set_up([0, -1, 0])  # 위쪽 방향 벡터
    
    vis.poll_events()
    vis.update_renderer()

    # 렌더링된 프레임 캡처
    frame = np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255
    vis.clear_geometries()
    vis.destroy_window()
    return frame.astype(np.uint8)

def calculate_cluster_movements(frames, movement_threshold=0.5):
    """
    여러 프레임의 클러스터 중심점 변화를 추적하고 이동량 계산.
    """
    tracked_clusters = []
    previous_clusters = {}
    cluster_movements = {}

    for frame_id, current_clusters in enumerate(frames):
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
                matched_clusters[curr_id] = current_clusters[curr_id]
                cluster_movements[curr_id] = cluster_movements.get(prev_id, 0) + np.linalg.norm(
                    previous_clusters[prev_id] - current_clusters[curr_id][0]
                )
        else:
            matched_clusters = {label: data for label, data in current_clusters.items()}

        # 이동량 기준 필터링: 이동량이 가장 큰 클러스터 선택
        if matched_clusters:
            max_movement_cluster = max(matched_clusters.items(), key=lambda x: cluster_movements.get(x[0], 0))
            tracked_clusters.append({max_movement_cluster[0]: max_movement_cluster[1]})

        # 현재 클러스터 갱신
        previous_clusters = {label: data[0] for label, data in matched_clusters.items()}

    return tracked_clusters

# 단일 시나리오 처리
print(f"Processing scenario: {scenario_name}")
video_output_path = os.path.join(video_path, f"{scenario_name}_output.mp4")
out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

all_frames = []

for pcd_file in pcd_files:
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 1. 다운샘플링
    pcd_downsampled = downsample(pcd, voxel_size=0.5)

    # 2. 노이즈 제거
    pcd_cleaned = remove_noise(pcd_downsampled, sor_neighbors=20, sor_std_ratio=1.0, ror_points=6, ror_radius=0.5)

    # 3. 도로 제거
    ground, non_ground = remove_ground(pcd_cleaned, distance_threshold=0.1, ransac_n=3, num_iterations=1000)

    # 4. 클러스터링
    labels = apply_dbscan(non_ground, eps=1.0, min_points=10)

    # 5. 클러스터 중심점 계산
    current_clusters = process_pcd_for_rendering(non_ground, labels)
    all_frames.append(current_clusters)

# 6. 프레임 간 클러스터 이동량 추적
tracked_clusters = calculate_cluster_movements(all_frames, movement_threshold=0.5)

# 7. 비디오 생성
for frame_id, frame_clusters in enumerate(tracked_clusters):
    pcd = o3d.io.read_point_cloud(pcd_files[frame_id])
    bounding_boxes = []

    for label, (centroid, cluster_pcd) in frame_clusters.items():
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)
        bounding_boxes.append(bbox)

    frame = render_to_frame(pcd, bounding_boxes, frame_width, frame_height)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)

out.release()  # 비디오 파일 저장
print(f"Scenario video saved: {video_output_path}")
gc.collect()  # 메모리 정리