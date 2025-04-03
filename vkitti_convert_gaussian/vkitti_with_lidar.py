"""
vkitti_with_lidar.py
自定义需求的vkitti数据集类，相比官方的vkitti2类，增加了对lidar数据的读取
"""

import os
import re
from typing import List
import open3d as o3d

from vkitti_convert_gaussian.vkitti import VkittiSceneData, FramesData, ExtrinsicsData


class LidarData:
    def __init__(self, dir_path: str):
        self.endwith: str = "pcd"
        self.dir_path: str = dir_path
        self.lidar_paths: List[str] = os.listdir(dir_path)
        self.lidar_paths.sort(
            key=lambda _pcd_name: int(re.match(r"(\d+).pcd", _pcd_name).group(1))
        )

    def get_lidar_path(self, index: int) -> str:
        return os.path.join(self.dir_path, self.lidar_paths[index])

    def get_lidar(self, index: int) -> o3d.geometry.PointCloud:
        return o3d.io.read_point_cloud(self.get_lidar_path(index))

    def __getitem__(self, index: int) -> o3d.geometry.PointCloud:
        return self.get_lidar(index)

    def __len__(self) -> int:
        return len(self.lidar_paths)


class FramesDataWithLidar(FramesData):
    def __init__(self, dir_path: str):
        super().__init__(dir_path)
        try:
            lidar = LidarData(os.path.join(dir_path, "lidar"))
        except FileNotFoundError as e:
            lidar = None
            print("由于错误：", e, "不能使用--use-lidar选项，self.lidar将为None")
        self.lidar: LidarData = lidar


class VkittiWithLidar(VkittiSceneData):
    def __init__(self, scene_path: str):
        super().__init__(scene_path)
        self.frames: FramesDataWithLidar = FramesDataWithLidar(
            os.path.join(scene_path, "frames")
        )
        try:
            lidar_extrinsics = ExtrinsicsData(
                os.path.join(scene_path, "lidar_extrinsic.txt")
            )
        except FileNotFoundError as e:
            lidar_extrinsics = None
            print(
                "由于错误：",
                e,
                "不能使用--use-lidar选项，self.lidar_extrinsics将为None",
            )

        self.lidar_extrinsics: ExtrinsicsData = lidar_extrinsics
