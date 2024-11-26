import open3d as o3d
import numpy as np
from scipy.spatial.distance import pdist

# 1. PCD 파일 읽기
# pcd_file_path는 사용자가 가진 PCD 파일 경로로 변경
pcd_file_path = "/Users/eunchaelin/Desktop/MyFolder/Korea University/4-2/AutonomousVehicle/HW1/project/data/01_straight_walk/pcd_000001.pcd"
pcd = o3d.io.read_point_cloud(pcd_file_path)

# 2. 점 좌표 추출
points = np.asarray(pcd.points)  # N x 3 형태의 NumPy 배열로 변환

# 3. 점 간 거리 계산
distances = pdist(points, metric='euclidean')  # 모든 점 쌍의 유클리드 거리 계산
average_distance = np.mean(distances)          # 평균 거리 계산

# 4. 결과 출력
print(f"점 간 평균 거리: {average_distance:.4f}")