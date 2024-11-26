import open3d as o3d
import numpy as np
import os
import cv2
from sklearn.cluster import DBSCAN
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

def load_pcd_files(pcd_dir):
    pcd_files = sorted([os.path.join(pcd_dir, file) for file in os.listdir(pcd_dir) if file.endswith('.pcd')])
    return pcd_files

def preprocess_point_cloud(pcd, voxel_size=0.5, eps=0.5, min_samples=10):
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(pcd.points)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    max_label = labels.max() + 1
    colors = np.random.uniform(0, 1, size=(max_label, 3))
    colors = np.vstack([colors, [0, 0, 0]])  # Noise color
    pcd.colors = o3d.utility.Vector3dVector(colors[labels])

    return pcd, labels

movement_vectors = defaultdict(list)

def is_person_by_motion(cluster_id, current_centroid, direction_threshold=0.1, N=10):
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

def extract_keyframes_by_ratio(pcd_files, ratio=0.3):
    total_frames = len(pcd_files)
    step = max(1, int(1 / ratio))
    keyframes = pcd_files[::step]
    print(f"Total frames: {total_frames}, Selected keyframes: {len(keyframes)}")
    return keyframes

def render_pcd_and_save_video(pcd_files, output_dir, video_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.get_render_option().point_size = 2.0

    frame_width = 1280
    frame_height = 720
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}.mp4"), fourcc, fps, (frame_width, frame_height))

    with ThreadPoolExecutor() as executor:
        future_results = list(executor.map(process_frame, pcd_files))

    for idx, (pcd, labels) in enumerate(future_results):
        print(f"Rendering frame {idx + 1}/{len(pcd_files)}")

        cluster_ids = np.unique(labels)
        bounding_boxes = []
        person_cluster = None
        max_motion = -1

        for cluster_id in cluster_ids:
            if cluster_id == -1:
                continue

            cluster_points = np.asarray(pcd.points)[labels == cluster_id]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

            current_centroid = np.mean(cluster_points, axis=0)

            if is_person_by_motion(cluster_id, current_centroid):
                motion_magnitude = np.linalg.norm(np.mean(np.diff(movement_vectors[cluster_id], axis=0), axis=0))
                if motion_magnitude > max_motion:
                    max_motion = motion_magnitude
                    person_cluster = cluster_pcd

        if person_cluster:
            aabb = person_cluster.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)
            bounding_boxes.append(aabb)

        vis.add_geometry(pcd)
        for bbox in bounding_boxes:
            vis.add_geometry(bbox)

        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        video_writer.write(frame)
        vis.clear_geometries()

    vis.destroy_window()
    video_writer.release()
    print(f"Video saved to {os.path.join(output_dir, video_name)}.mp4")

def process_frame(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    return preprocess_point_cloud(pcd)

# Directory setup
scenario = "03_straight_crawl"  
input_root_dir = "./data" 
output_root_dir = "./output"
os.makedirs(output_root_dir, exist_ok=True)

print(f"Processing scenario: {scenario}")
pcd_dir = os.path.join(input_root_dir, scenario, 'pcd')
output_dir = os.path.join(output_root_dir, scenario)
os.makedirs(output_dir, exist_ok=True)

pcd_files = load_pcd_files(pcd_dir)
keyframes = extract_keyframes_by_ratio(pcd_files, ratio=0.1)

render_pcd_and_save_video(keyframes, output_dir, scenario)