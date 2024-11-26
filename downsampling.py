import open3d as o3d

def downsample(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size=voxel_size)