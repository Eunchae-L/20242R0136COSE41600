import open3d as o3d
import numpy as np
import os
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from collections import defaultdict

# Helper function to load PCD files
def load_pcd_files(pcd_dir):
    pcd_files = sorted([os.path.join(pcd_dir, file) for file in os.listdir(pcd_dir) if file.endswith('.pcd')])
    return pcd_files

# Function to preprocess point cloud: noise removal, downsampling, clustering
def preprocess_point_cloud(pcd):
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.3)
    points = np.asarray(pcd.points)

    # Perform clustering (DBSCAN)
    clustering = DBSCAN(eps=0.2, min_samples=10).fit(points)
    labels = clustering.labels_

    # Colorize clusters
    max_label = labels.max() + 1
    colors = np.random.uniform(0, 1, size=(max_label, 3))
    colors = np.vstack([colors, [0, 0, 0]])  # Add color for noise points
    pcd.colors = o3d.utility.Vector3dVector(colors[labels])

    return pcd, labels

movement_vectors = defaultdict(list)

def is_person_by_motion(cluster_id, current_centroid, direction_threshold=0.1, N=100):
    """프레임 간 이동 방향성을 기반으로 사람인지 판단"""
    global movement_vectors

    if cluster_id not in movement_vectors:
        movement_vectors[cluster_id].append(current_centroid)
        return False

    # Calculate movement vector
    previous_centroid = movement_vectors[cluster_id][-1]
    movement_vector = current_centroid - previous_centroid

    # Update movement history
    movement_vectors[cluster_id].append(current_centroid)
    if len(movement_vectors[cluster_id]) > N:
        movement_vectors[cluster_id].pop(0)

    # Check direction consistency
    if len(movement_vectors[cluster_id]) >= N:
        avg_direction = np.mean(np.diff(movement_vectors[cluster_id], axis=0), axis=0)
        if np.linalg.norm(avg_direction) > direction_threshold:
            return True

    return False

def render_pcd_and_save_video(pcd_files, output_dir, video_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.get_render_option().point_size = 2.0

    frame_width = 1280
    frame_height = 720
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}.mp4"), fourcc, fps, (frame_width, frame_height))

    previous_centroids = {}

    for idx, pcd_file in enumerate(pcd_files):
        print(f"Processing frame {idx + 1}/{len(pcd_files)}: {pcd_file}")

        # Load and preprocess point cloud
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd, labels = preprocess_point_cloud(pcd)

        # Generate bounding boxes for clusters
        cluster_ids = np.unique(labels)
        bounding_boxes = []
        for cluster_id in cluster_ids:
            if cluster_id == -1:
                continue  # Skip noise points

            cluster_points = np.asarray(pcd.points)[labels == cluster_id]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

            # Calculate current centroid
            current_centroid = np.mean(cluster_points, axis=0)

            # Check if cluster represents a person by motion
            if is_person_by_motion(cluster_id, current_centroid):
                bbox_color = (1, 0, 0)  # Red for person
            else:
                bbox_color = (0, 1, 0)  # Green for others

            # Generate bounding box
            aabb = cluster_pcd.get_axis_aligned_bounding_box()
            scale_factor = 4.0
            aabb = aabb.scale(scale_factor, aabb.get_center())
            aabb.color = bbox_color
            bounding_boxes.append(aabb)

        vis.add_geometry(pcd)
        vis.add_geometry(bbox)

        for bbox in bounding_boxes:
            vis.update_geometry(bbox)

        # Render and capture frame
        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write frame to video
        video_writer.write(frame)

        # Clear geometries
        vis.clear_geometries()

    vis.destroy_window()
    video_writer.release()
    print(f"Video saved to {os.path.join(output_dir, video_name)}.mp4")


# Directory setup
scenario = "02_straight_duck_walk"  
input_root_dir = "./data" 
output_root_dir = "./output"
os.makedirs(output_root_dir, exist_ok=True)

# Process a single scenario
print(f"Processing scenario: {scenario}")
pcd_dir = os.path.join(input_root_dir, scenario, 'pcd')
output_dir = os.path.join(output_root_dir, scenario)
os.makedirs(output_dir, exist_ok=True)

# Load PCD files
pcd_files = load_pcd_files(pcd_dir)

# Render PCD and save as a video
render_pcd_and_save_video(pcd_files, output_dir, scenario)