"""
lidar_extractor.py
读取frames/lidar下的数据，通过pose信息，分离点云中的车辆
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def extract_np_points_using_3dbbox(
    points: np.ndarray,  # 这里的点云是在世界坐标系下的
    pose: np.ndarray,
    # width: np.float64,
    # height: np.float64,
    # length: np.float64,
    # world_space_X: np.float64,
    # world_space_Y: np.float64,
    # world_space_Z: np.float64,
    # rotation_world_space_y: np.float64,
    # rotation_world_space_x: np.float64,
    # rotation_world_space_z: np.float64,
    need: bool = False,  # 如果需要3D框内的点云,就设置为True，此时返回的是3D框内的点云
):
    """
    通过3D框，剔除点云中的车辆
    :param points: 点云
    :param pose: 一帧中的pose信息，包括width, height, length, world_space_X, world_space_Y, world_space_Z, rotation_world_space_y, rotation_world_space_x, rotation_world_space_z
    :param need: 如果需要3D框内的点云,就设置为True，此时返回的是3D框内的点云
    :return: 剔除车辆后的点云
    """
    (width, height, length, world2car) = get_3dbbox_from_pose(pose)
    # 将点云转换到3D框的局部坐标系中
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(points)
    points_local_o3d = points_o3d.transform(world2car)

    points_local = np.asarray(points_local_o3d.points)

    # 判断点是否在3D框内
    mask = np.logical_and(
        np.logical_and(
            points_local[:, 0] > -width / 2, points_local[:, 0] < width / 2
        ),  # x in [-width/2, width/2]  3D框的宽度
        np.logical_and(
            points_local[:, 1] > -height / 2, points_local[:, 1] < height / 2
        ),  # y in [-height/2, height/2]  3D框的高度
        np.logical_and(
            points_local[:, 2] > -length / 2, points_local[:, 2] < length / 2
        ),  # z in [-length/2, length/2]  3D框的长度
    )
    return points_local[mask] if need else points[~mask]


def get_3dbbox_from_pose(pose: np.ndarray):
    """
    从pose中获取3D框的信息
    :param pose: 一帧中的pose信息
    :return: 3D框的信息
    """
    width = pose["width"]
    height = pose["height"]
    length = pose["length"]
    rotation = Rotation.from_euler(
        "xyz",
        [
            pose[f"rotation_world_space_x"],
            pose[f"rotation_world_space_y"],
            pose[f"rotation_world_space_z"],
        ],
        degrees=False,
    )
    t = np.array(
        [
            pose[f"world_space_X"],
            pose[f"world_space_Y"],
            pose[f"world_space_Z"],
        ]
    )  # 这里shape是(3,)
    car2world = np.zeros((4, 4))
    car2world[:3, :3] = rotation.as_matrix()
    car2world[:3, 3] = t
    car2world[3, 3] = 1

    world2car = np.linalg.inv(car2world)
    return (width, height, length, world2car)
