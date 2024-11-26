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
from utils import get_pcd_files

os.environ["OPEN3D_DEVICE"] = "CPU"
print(f"OPEN3D_DEVICE: {os.getenv('OPEN3D_DEVICE')}")

# 시나리오 디렉토리 설정
root_dir = "data"
scenario_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

video_path = '/Users/eunchaelin/Desktop/MyFolder/Korea University/4-2/AutonomousVehicle/HW1/project/video'
frame_width, frame_height = 1280, 720
fps = 10
BATCH_SIZE=10

def render_to_frame(pcd, bounding_boxes, width=1280, height=720):
    """PCD와 바운딩 박스를 렌더링하여 프레임으로 반환"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.poll_events()
    vis.update_renderer()

    # 렌더링된 프레임 캡처
    frame = np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255
    vis.clear_geometries()
    vis.destroy_window()
    return frame.astype(np.uint8)

def process_batch(batch_files, batch_index, scenario_name):
    """PCD 배치를 처리하여 MP4 비디오로 저장"""
    video_output_path = os.path.join(video_path, f"{scenario_name}_batch_{batch_index}_output.mp4")
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for pcd_file in batch_files:
        print(f"Processing PCD file: {pcd_file}")

        # 1. PCD 파일 로드
        pcd = o3d.io.read_point_cloud(pcd_file)
        print(f"Point cloud has {len(pcd.points)} points.")

        # 2. 다운샘플링
        pcd_downsampled = downsample(pcd, voxel_size=0.5)
        print(f"Point cloud downsampled has {len(pcd_downsampled.points)} points.")

        # 3. 노이즈 제거
        pcd_cleaned = remove_noise(pcd_downsampled, sor_neighbors=20, sor_std_ratio=1.0, ror_points=6, ror_radius=1.0)
        print(f"Number of points in PCD: {len(np.asarray(pcd_cleaned.points))}")

        # 4. 도로 제거
        ground, non_ground = remove_ground(pcd_cleaned, distance_threshold=0.1, ransac_n=3, num_iterations=1000)
        print(f"Non-ground points: {np.asarray(non_ground.points).shape}")

        # 5. 클러스터링
        try:
            labels = apply_dbscan(non_ground, eps=0.8, min_points=5)
        except Exception as e:
            print(f"DBSCAN failed: {e}")
            labels = np.zeros(len(non_ground.points))

        # 6. 색상 적용 및 바운딩 박스 생성
        pcd_with_colors, bounding_boxes = process_pcd_for_rendering(non_ground, labels)
        # 7. 프레임 렌더링
        frame = render_to_frame(pcd_with_colors, bounding_boxes, frame_width, frame_height)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"Batch {batch_index} video saved: {video_output_path}")
    gc.collect()  # 메모리 정리


# 메인 루프
for scenario in scenario_dirs:
    print(f"Processing scenario: {scenario}")
    scenario_path = os.path.join(root_dir, scenario)
    pcd_files = [os.path.join(scenario_path, f) for f in os.listdir(scenario_path) if f.endswith(".pcd")]
    print(pcd_files)

    # 배치 처리
    for i in range(0, len(pcd_files), BATCH_SIZE):
        batch_files = pcd_files[i:i + BATCH_SIZE]
        batch_index = i // BATCH_SIZE + 1
        process_batch(batch_files, batch_index, scenario)

    print(f"Finished processing scenario: {scenario}")