import os
import open3d as o3d
import cv2
import numpy as np
from noise_removal import remove_noise
from downsampling import downsample
from ground_removal import remove_ground
from clustering import cluster_and_colorize
from visualization import visualize_clusters, draw_bounding_boxes
from utils import get_pcd_files

# 시나리오 디렉토리 설정
root_dir = "data"
scenario_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

for scenario in scenario_dirs:
    print(f"Processing scenario: {scenario}")
    scenario_path = os.path.join(root_dir, scenario, "pcd")
    pcd_files = get_pcd_files(scenario_path)

    # 영상 저장 초기화
    video_path = '/Users/eunchaelin/Desktop/MyFolder/Korea University/4-2/AutonomousVehicle/HW1/project/videoㅜ'
    video_output_path = os.path.join(video_path, "output_video.mp4")
    frame_width, frame_height = 1280, 720
    fps = 10
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Open3D 시각화 초기화
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=frame_width, height=frame_height)

    for pcd_file in pcd_files:
        # PCD 로드
        pcd = o3d.io.read_point_cloud(pcd_file)
        
        # 1. 노이즈 제거
        pcd_cleaned = remove_noise(pcd)
        
        # 2. 다운샘플링
        pcd_downsampled = downsample(pcd_cleaned)
        
        # 3. 도로 제거
        non_ground = remove_ground(pcd_downsampled)
        
        # 4. 클러스터링
        pcd = cluster_and_colorize(pcd, eps=0.6, min_points=11)
        
        # 5. 클러스터 시각화 및 바운딩 박스 생성
        visualize_clusters(non_ground, labels)
        bounding_boxes = draw_bounding_boxes(non_ground, labels)

        # Open3D 화면 렌더링
        vis.add_geometry(non_ground)
        for bbox in bounding_boxes:
            vis.add_geometry(bbox)
        vis.poll_events()
        vis.update_renderer()

        # Open3D 프레임 캡처
        frame = np.asarray(vis.capture_screen_float_buffer()) * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

        # Open3D 지오메트리 제거
        vis.clear_geometries()

    # 영상 저장 종료
    vis.destroy_window()
    out.release()
    print(f"Saved video: {video_output_path}")