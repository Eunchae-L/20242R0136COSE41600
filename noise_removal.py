import open3d as o3d

def remove_noise(pcd, sor_neighbors=20, sor_std_ratio=2.0, ror_points=16, ror_radius=0.5):
    # Statistical Outlier Removal (SOR)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio)
    pcd = pcd.select_by_index(ind)
    # Radius Outlier Removal (ROR)
    cl, ind = pcd.remove_radius_outlier(nb_points=ror_points, radius=ror_radius)
    pcd = pcd.select_by_index(ind)
    return pcd
