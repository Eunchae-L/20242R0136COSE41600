import open3d as o3d
import numpy as np
import os
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# Helper function to load PCD files
def load_pcd_files(pcd_dir):
    pcd_files = sorted([os.path.join(pcd_dir, file) for file in os.listdir(pcd_dir) if file.endswith('.pcd')])
    return pcd_files[:50]  # Limit to the first 20 files for testing

# Function to preprocess point cloud: noise removal, downsampling, clustering
def preprocess_point_cloud(pcd):
    # Remove noise using statistical outlier removal (SOR)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Downsample the point cloud
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # Convert point cloud to numpy array for clustering
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
# Global dictionary to track cluster trajectories
cluster_trajectories = {}

def match_clusters(previous_centroids, current_centroids, threshold=0.5):
    """클러스터 매칭: 이전 프레임과 현재 프레임 클러스터 중심을 매칭"""
    if len(previous_centroids) == 0:
        return {i: i for i in range(len(current_centroids))}

    distances = cdist(previous_centroids, current_centroids)
    matched_ids = {}
    for prev_id, row in enumerate(distances):
        current_id = np.argmin(row)
        if row[current_id] < threshold:
            matched_ids[prev_id] = current_id
    return matched_ids

def render_pcd_and_save_video(pcd_files, output_dir, video_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)

    frame_width = 1280
    frame_height = 720
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}.mp4"), fourcc, fps, (frame_width, frame_height))

    previous_centroids = []
    global cluster_trajectories

    for idx, pcd_file in enumerate(pcd_files):
        print(f"Processing frame {idx + 1}/{len(pcd_files)}: {pcd_file}")

        # Load and preprocess point cloud
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd, labels = preprocess_point_cloud(pcd)

        # Calculate centroids for current frame
        current_centroids = []
        cluster_ids = np.unique(labels)
        bounding_boxes = []
        for cluster_id in cluster_ids:
            if cluster_id == -1:
                continue  # Skip noise points

            cluster_points = np.asarray(pcd.points)[labels == cluster_id]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            centroid = np.mean(cluster_points, axis=0)
            current_centroids.append(centroid)

            # Generate bounding box
            aabb = cluster_pcd.get_axis_aligned_bounding_box()
            aabb.color = (0, 1, 0)  # Default: Green
            bounding_boxes.append((cluster_id, aabb))

        # Match clusters with previous centroids
        matched_ids = match_clusters(previous_centroids, current_centroids)

        # Update trajectories and color moving clusters
        for prev_id, current_id in matched_ids.items():
            if prev_id in cluster_trajectories:
                cluster_trajectories[prev_id].append(current_centroids[current_id])
            else:
                cluster_trajectories[prev_id] = [current_centroids[current_id]]

            # Check if trajectory is long enough to be a person
            trajectory_length = np.sum(np.linalg.norm(np.diff(cluster_trajectories[prev_id], axis=0), axis=1))
            if trajectory_length > 2.0:  # Threshold for person detection
                for cluster_id, aabb in bounding_boxes:
                    if cluster_id == current_id:
                        aabb.color = (1, 0, 0)  # Red for person

        # Update previous centroids
        previous_centroids = current_centroids

        # Add geometries to visualizer
        vis.add_geometry(pcd)
        for _, bbox in bounding_boxes:
            vis.add_geometry(bbox)

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
scenario = "01_straight_walk"  # Adjust this to the scenario you want to process
input_root_dir = "./data"  # Adjust this to the actual root directory of your data
output_root_dir = "./output"
os.makedirs(output_root_dir, exist_ok=True)

# Process a single scenario
print(f"Processing scenario: {scenario}")
pcd_dir = os.path.join(input_root_dir, scenario)
output_dir = os.path.join(output_root_dir, scenario)
os.makedirs(output_dir, exist_ok=True)

# Load PCD files
pcd_files = load_pcd_files(pcd_dir)

# Render PCD and save as a video
render_pcd_and_save_video(pcd_files, output_dir, scenario)