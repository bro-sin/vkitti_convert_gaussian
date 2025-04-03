import numpy as np
import open3d as o3d
from vkitti_convert_gaussian.vkitti_with_lidar import VkittiWithLidar
from scipy.spatial.transform import Rotation
from vkitti_convert_gaussian.utils.lidar_extractor import extract_np_points_using_3dbbox

Changan_path = "/home/lsin/Data/vkitti/Scene18/clone/"

data = VkittiWithLidar(Changan_path)

# 测试读取pose文件中获取某一帧某一辆车的世界坐标和旋转，还有3D框
frame = 50
pose = data.pose[frame]
print(pose)

pcd = data.frames.lidar[frame]  # 这里点云是在ego car坐标系下的
print(pcd)

# 先将点云转到世界坐标系
R = data.extrinsics.get_R(frame)
T = data.extrinsics.get_t(frame)
world_to_car = np.eye(4)
world_to_car[:3, :3] = R
world_to_car[:3, 3] = T
print(world_to_car)
pcd_world = pcd.transform(np.linalg.inv(world_to_car))
print(pcd_world)

# 提取某一个车的点云
trackID = 0
pose_car = pose[pose["trackID"] == trackID]
print(pose_car,pose_car.dtype)

width = pose_car["width"][0]
print(width)
height = pose_car["height"][0]
length = pose_car["length"][0]
world_space_X = pose_car["world_space_X"][0]
world_space_Y = pose_car["world_space_Y"][0]
world_space_Z = pose_car["world_space_Z"][0]
rotation_world_space_y = pose_car["rotation_world_space_y"][0]
rotation_world_space_x = pose_car["rotation_world_space_x"][0]
rotation_world_space_z = pose_car["rotation_world_space_z"][0]
print(width)
points_car_w = extract_np_points_using_3dbbox(
    np.asarray(pcd_world.points),  # 这里的点云是在世界坐标系下的
    pose=pose_car,
    need=True,
)# 这里返回的点云是在世界坐标系下的

print(points_car_w)


#需要将点云转到目标车辆自身坐标系下
car2world=data.pose.get_car2w(index=frame,trackID=trackID)
points_car_o3d=o3d.geometry.PointCloud()
points_car_o3d.points=o3d.utility.Vector3dVector(points_car_w)
points_car_o3d.transform(car2world)
# o3d.visualization.draw_geometries([points_car_o3d])
print(points_car_o3d)