import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from typing import Tuple, List


def depth_image_to_point_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    cam2car: np.ndarray = None,
    rgb: np.ndarray = None,
    scale=100,
    DEPTH_MAX=50,
    use_gaussian_depth_def: bool = False,
):
    # depth中的数处以scale后是米为单位的
    # Get the intrinsic parameters
    fx = intrinsics["K[0,0]"]
    fy = intrinsics["K[1,1]"]
    cx = intrinsics["K[0,2]"]
    cy = intrinsics["K[1,2]"]
    # Get the extrinsic parameters
    # r = np.vstack(
    #     (
    #         extrinsics[["r1,1", "r1,2", "r1,3"]].tolist(),
    #         extrinsics[["r2,1", "r2,2", "r2,3"]].tolist(),
    #         extrinsics[["r3,1", "r3,2", "r3,3"]].tolist(),
    #     )
    # )
    # t = np.array(extrinsics[["t1", "t2", "t3"]].tolist())
    # 这里相机外参是w2c的：唐yt试出来的
    #    r1,1 r1,2 r1,3 t1
    # Extrinsic = r2,1 r2,2 r2,3 t2
    #             r3,1 r3,2 r3,3 t3
    #             0    0    0    1
    if cam2car is None:
        extrinsics_w2c = np.vstack(
            (
                extrinsics[["r1,1", "r1,2", "r1,3", "t1"]].tolist(),
                extrinsics[["r2,1", "r2,2", "r2,3", "t2"]].tolist(),
                extrinsics[["r3,1", "r3,2", "r3,3", "t3"]].tolist(),
                [0, 0, 0, 1],
            )
        )
        # print(f"extrinsics_w2c: {extrinsics_w2c.shape},type {type(extrinsics_w2c)}")
        # print(f"extrinsics_w2c: {extrinsics_w2c}")
        # print(f"extrinsics_w2c: {extrinsics_w2c.dtype}")
        c2w = np.linalg.inv(extrinsics_w2c)
    else:
        c2w = cam2car
    # print(f"c2w: {c2w}")

    # 生成像素坐标
    u, v = range(0, depth.shape[1]), range(0, depth.shape[0])
    u, v = np.meshgrid(u, v)
    # print(f"u: {u.shape},{u.dtype}, v: {v.shape},{v.dtype}")
    # print(f"u: {u}, v: {v}")
    u = u.astype(float)
    v = v.astype(float)

    # 计算相机坐标
    Z = depth.astype(float) / scale
    if use_gaussian_depth_def:
        Z = (
            Z
            * fx
            * fy
            / np.sqrt(fx**2 * (u - cx) ** 2 + fy**2 * (v - cy) ** 2 + fx**2 * fy**2)
        )
    X = (u - cx) * Z / fx  # 每个分量的乘法
    Y = (v - cy) * Z / fy

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z < DEPTH_MAX  # 只取深度小于DEPTH_MAX的点

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    # 计算世界坐标
    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(c2w, position)

    # 把其次坐标转换为三维坐标
    if rgb is None:
        points = np.transpose(position[0:3, :])
    else:
        # opencv读取的图像是BGR的,
        B = rgb[:, :, 0].ravel()[valid]
        G = rgb[:, :, 1].ravel()[valid]
        R = rgb[:, :, 2].ravel()[valid]
        points = np.transpose(np.vstack((position[0:3, :], R, G, B)))
    return points


def write_point_cloud(points: np.ndarray, filename: str):
    if filename.endswith(".ply"):
        write_point_ply(points, filename)
    elif filename.endswith(".txt"):
        write_point_txt(points, filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")


def write_point_ply(points: np.ndarray, filename: str):
    property_num = len(points[0])
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if property_num == 6:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}")
            if property_num == 6:
                # 后面属性是RGB,需要先转为整数
                f.write(f" {int(point[3])} {int(point[4])} {int(point[5])}")
            f.write("\n")
    return


def write_point_plyfile(points: np.ndarray, filename: str):
    # 使用plyfile库
    if len(points) == 0:
        with open(filename.replace("ply", "txt"), "w") as f:
            pass
        return
    property_num = len(points[0])
    # assert property_num == 6
    # 当没有rgb值时,只有3个属性
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
    ]
    vertex = np.zeros(len(points), dtype=dtype)
    for i, point in enumerate(points):
        # point属性后面需要加3个0,作为法向量
        if property_num == 6:
            vertex[i] = tuple(np.concatenate([point, [0, 0, 0]]))
        elif property_num == 3:
            vertex[i] = tuple(np.concatenate([point, [0, 0, 0, 0, 0, 0]]))
        else:
            raise ValueError(f"Unsupported property_num: {property_num}")
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(filename)
    return


def write_point_txt(points: np.ndarray, filename: str):
    # 写txt格式的,格式要求如下
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # Number of points: 4309985, mean track length: 0
    # 4309985 133.69400024414062 -106.01100158691406 157.88999938964844 150 145 117 -1
    # 4309984 46.384799957275391 -106 45.891799926757812 81 77 48 -1
    # 4309983 46.620098114013672 -107.19599914550781 47.362499237060547 5
    property_num = len(points[0])
    with open(filename, "w") as f:
        for i, point in enumerate(points[::-1]):
            f.write(f"{i+1} {point[0]} {point[1]} {point[2]}")
            if property_num == 6:
                # 后面属性是RGB,需要先转为整数
                f.write(f" {int(point[3])} {int(point[4])} {int(point[5])}")
            f.write(" -1\n")
    return


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from vkitti_convert_gaussian.vkitti import VkittiSceneData

    VkittiSceneData_path = "/home/lsin/Public/vkitti/Scene18/clone"
    vkitti_scene_data = VkittiSceneData(VkittiSceneData_path)
    cam_id = 0
    frame_id = 0
    depth_image = vkitti_scene_data.frames.depth[cam_id][frame_id]
    point_cloud = depth_image_to_point_cloud(
        depth_image,
        vkitti_scene_data.intrinsics[frame_id][cam_id],
        vkitti_scene_data.extrinsics[frame_id][cam_id],
        rgb=vkitti_scene_data.frames.rgb[cam_id][frame_id],
    )
    print(len(point_cloud))
    # write_point_cloud(point_cloud, "point_cloud.ply")

    # # 把点云保存为ply文件
    # with open("point_cloud.ply", "w") as f:
    #     f.write("ply\n")
    #     f.write("format ascii 1.0\n")
    #     f.write(f"element vertex {len(point_cloud)}\n")
    #     f.write("property float x\n")
    #     f.write("property float y\n")
    #     f.write("property float z\n")
    #     f.write("end_header\n")
    #     for point in point_cloud:
    #         f.write(f"{point[0]} {point[1]} {point[2]}\n")

    # 测试保存为txt
    # write_point_cloud(point_cloud, "point_cloud.txt")

    # 测试保存为plyfile
    write_point_cloud(point_cloud, "point_cloud.ply")
