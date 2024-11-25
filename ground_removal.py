import open3d as o3d

def remove_ground(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    ground_pcd = pcd.select_by_index(inliers)
    non_ground_pcd = pcd.select_by_index(inliers, invert=True)
    return ground_pcd, non_ground_pcd