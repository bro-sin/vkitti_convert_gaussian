import numpy as np
from vkitti_convert_gaussian.utils.depth_to_point import write_point_plyfile


def create_rectangular_prism_point_cloud(length, width, height, num_points):
    # 生成均匀分布的点
    x = np.random.uniform(0, length, num_points)
    y = np.random.uniform(0, width, num_points)
    z = np.random.uniform(0, height, num_points)

    # 将点云数据合并为一个数组
    points = np.vstack((x, y, z)).T
    return points


# 设置车辆的长、宽、高和点云的数量
length = 4.0  # 例如4米
width = 1.8  # 例如1.8米
height = 1.6  # 例如1.6米
num_points = 10000  # 例如10000个点

# 生成点云
points = create_rectangular_prism_point_cloud(length, width, height, num_points)

# 保存为PLY文件
filename = "cuboid_point_cloud.ply"
write_point_plyfile(points, filename)

print(f"Point cloud saved to {filename}")
