import open3d as o3d
from noise_removal import remove_noise
from downsampling import downsample
from ground_removal import remove_ground
from clustering import cluster_points
from bbox import draw_bounding_boxes
from visualize import visualize
import glob

def process_lidar(file_path):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Step 1: Noise Removal
    pcd = remove_noise(pcd)
    
    # Step 2: Downsampling
    pcd = downsample(pcd)
    
    # Step 3: Ground Removal
    _, pcd = remove_ground(pcd)
    
    # Step 4: Clustering
    labels = cluster_points(pcd)
    
    # Step 5: Draw Bounding Boxes
    geometries = draw_bounding_boxes(pcd, labels)
    
    # Step 6: Visualize and Save
    visualize(pcd, geometries)

if __name__ == "__main__":
    # Process all .pcd files in data directory
    file_paths = glob.glob("data/*/pcd/*.pcd")
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        process_lidar(file_path)
