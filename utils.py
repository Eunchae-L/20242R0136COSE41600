import os

def get_pcd_files(folder_path):
    """지정된 폴더에서 .pcd 파일 목록 가져오기"""
    return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pcd")])