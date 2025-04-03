import os
import shutil
import sys
import numpy as np
import cv2
import open3d as o3d
from rich.progress import Progress
from scipy.spatial.transform import Rotation
from PIL import Image
import tyro

from typing import Tuple, TYPE_CHECKING, List

if __name__ == "__main__":
    # 把当前文件夹加入到系统路径中
    sys.path.append(".")
from vkitti_convert_gaussian.vkitti_with_lidar import VkittiWithLidar as VkittiSceneData
from vkitti_convert_gaussian.utils.extractor import (
    extract_instance_image,
    extract_instance_depth,
    extract_class_image,
    extract_class_depth,
)
from vkitti_convert_gaussian.utils.depth_to_point import (
    depth_image_to_point_cloud,
    write_point_cloud,
    write_point_plyfile,
)

from vkitti_convert_gaussian.utils.lidar_extractor import extract_np_points_using_3dbbox


def convert_extrinsic_to_quaternion_and_t(extrinsic):
    """
    将外参转为四元数和平移向量
    :param extrinsic: 4x4的外参矩阵
    :return: quaternion, t
    """
    r = Rotation.from_matrix(extrinsic[:3, :3])
    quaternion = r.as_quat()
    t = extrinsic[:3, 3]
    return quaternion, t


class GaussianConverter:
    def __init__(
        self,
        scene_data: VkittiSceneData,
        output_dir: str = None,
        start_frame: int = 0,
        end_frame: int = -1,
    ):
        self.scene_data = scene_data
        self.output_dir: str = (
            os.path.join(output_dir, "gaussian")
            if output_dir
            else os.path.join(scene_data.dir_path, "gaussian")
        )

        assert start_frame in range(len(self.scene_data.frames.rgb[0]))
        assert end_frame in range(len(self.scene_data.frames.rgb[0])) or end_frame == -1
        self.start_frame = start_frame
        self.end_frame = (
            len(self.scene_data.frames.rgb[0]) - 1 if end_frame == -1 else end_frame
        )

        self.output_dir = os.path.join(
            self.output_dir, f"{self.start_frame}-{self.end_frame}"
        )

        self.gaussian_dir = self.output_dir
        self.colmap_sparse_dir = os.path.join(self.gaussian_dir, "sparse", "0")
        if os.path.exists(self.gaussian_dir):
            # import shutil

            # shutil.rmtree(self.gaussian_dir)
            print(f"{self.gaussian_dir}已经存在，将会覆盖")
        # os.makedirs(self.colmap_sparse_dir, exist_ok=True)
        return

    def _set_output_dir(self, gaussian_output_name: str):
        self.gaussian_dir = os.path.join(self.output_dir, gaussian_output_name)
        self.colmap_sparse_dir = os.path.join(self.gaussian_dir, "sparse", "0")
        if os.path.exists(self.gaussian_dir):
            import shutil

            shutil.rmtree(self.gaussian_dir)
            pass
        os.makedirs(self.colmap_sparse_dir, exist_ok=True)
        return

    def get_bgr_target(
        self, trackID: int = -1, cam_id: int = 0, search_frames: int = 100
    ):
        # 输入trackID，返回这辆车在实例分割中的bgr值
        # 如果trackID为-1，返回背景的bgr值
        if trackID == -1:
            return (0, 0, 0)
        instance_path = self.scene_data.frames.instanceSegmentation[
            cam_id
        ].get_pic_path(0)
        src = Image.open(instance_path)
        assert src.mode == "P"
        patelle = src.getpalette()
        # breakpoint()
        # All instance segmentation images are encoded as 8bit indexed PNG files.
        # The value of a pixel is equal to (trackID+1). A value of 0 means it is not a vehicle.
        index = trackID + 1
        r, g, b = patelle[index * 3 : index * 3 + 3]
        return (b, g, r)

    def get_rgb_of_category(self, category: str):
        # 输入类别名，返回类别的rgb值
        return self.scene_data.colors[category]

    def convert_depth_to_pointclouds(
        self,
        bgr_target: Tuple[int, int, int] = (0, 0, 0),
        trackID: int = -1,
        need: bool = True,
        with_rgb: bool = True,
        frames: List[int] = None,
    ):
        if trackID != -1:
            bgr_target = self.get_bgr_target(trackID=trackID)
        point_clouds_dir = os.path.join(self.gaussian_dir, "point_clouds")
        os.makedirs(point_clouds_dir, exist_ok=True)
        for cam_id in range(self.scene_data.frames.depth.cam_num):
            depth_data = self.scene_data.frames.depth[cam_id]
            if frames is None:
                frames = range(len(depth_data))
            with Progress() as progress:
                task = progress.add_task(
                    f"Converting depth to pointclouds for cam{cam_id}",
                    total=len(frames),
                )
                for i in frames:
                    extracted_depth = extract_instance_depth(
                        depth_data[i],
                        self.scene_data.frames.instanceSegmentation[cam_id][i],
                        bgr_target=bgr_target,
                        need=need,
                    )
                    if bgr_target != (0, 0, 0) and all(
                        extracted_depth.flatten() == 65536
                    ):
                        # print(f"第{i}张图片没有实例，跳过")
                        progress.update(task, advance=1)
                        continue
                    point_cloud = depth_image_to_point_cloud(
                        extracted_depth,
                        self.scene_data.intrinsics[i][cam_id],
                        self.scene_data.extrinsics[i][cam_id],
                        rgb=self.scene_data.frames.rgb[cam_id][i] if with_rgb else None,
                        DEPTH_MAX=400,
                    )
                    if len(point_cloud) == 0:
                        # print(f"第{i}张图片没有点云，跳过")
                        progress.update(task, advance=1)
                        continue
                    write_point_cloud(
                        point_cloud,
                        os.path.join(
                            point_clouds_dir,
                            f"pointcloud_{cam_id}_{i}.ply",
                        ),
                    )
                    progress.update(task, advance=1)
        return

    def convert_depth_merge_point_clouds_for_car(
        self,
        cam_id: int = 0,
        trackID: int = 0,
        downsample: bool = True,
        voxel_size: float = 0.01,
        focous_frame: int = 26,
    ):
        # 将点云转到第focous_frame帧的车辆坐标系下
        # 需要找到每一帧到focous_frame的外参，然后将点云转到第focous_frame帧的世界坐标系下
        # 初始版本，可以直接返回第focus_frame帧的点云，后续再考虑合并的功能

        # TODO:将每一帧的点云转到车辆坐标系，然后合并
        # 后续对车辆点云的训练渲染以及操作，以车辆坐标系为准
        depth_data = self.scene_data.frames.depth[cam_id]
        instance_data = self.scene_data.frames.instanceSegmentation[cam_id]
        bgr_target = self.get_bgr_target(trackID=trackID)
        intrinsics = self.scene_data.intrinsics
        # 外参要考虑使用cam2car的矩阵，这样提取出来的点就是在车辆坐标系的，不用从世界坐标系再次进行转换
        rgb_data = self.scene_data.frames.rgb[cam_id]

        merged_point_cloud_o3d = o3d.geometry.PointCloud()
        with Progress() as progress:
            task = progress.add_task(
                f"Converting depth images to point clouds for car{trackID} in cam{cam_id}",
                total=self.end_frame - self.start_frame + 1,
            )
            for _frame in range(self.start_frame, self.end_frame + 1):
                extracted_depth = extract_instance_depth(
                    depth_img=depth_data[_frame],
                    instance_seg_img=instance_data[_frame],
                    bgr_target=bgr_target,
                    need=True,
                )
                if all(extracted_depth.flatten() >= 65535):
                    progress.update(task, advance=1)
                    # 在该深度图中没有提取到车子的点
                    continue
                _extrinsics = self.scene_data.pose.get_cam2car(
                    index=_frame, trackID=trackID, cam_id=cam_id
                )
                point_cloud = depth_image_to_point_cloud(
                    depth=extracted_depth,
                    intrinsics=intrinsics[_frame][cam_id],
                    extrinsics=None,
                    cam2car=_extrinsics,
                    rgb=rgb_data[_frame],
                    DEPTH_MAX=200,  # 由于车的数据比较少，就给多一点距离
                )
                if len(point_cloud) == 0:
                    progress.update(task, advance=1)
                    # 没有提取到点，跳过
                    continue
                point_cloud_o3d = o3d.geometry.PointCloud()
                point_cloud_o3d.points = o3d.utility.Vector3dVector(
                    point_cloud[:, 0:3]  # 前面维度是x,y,z
                )
                if len(point_cloud[0]) == 6:
                    point_cloud_o3d.colors = o3d.utility.Vector3dVector(
                        point_cloud[:, 3:6] / 255.0
                    )
                    pass  # end if
                merged_point_cloud_o3d += point_cloud_o3d
                progress.update(task, advance=1)
                pass  # end for _frame
            pass  # end progress

        # 降采样
        if downsample:
            print("Downsampling depth images")
            downsampled_point_cloud = merged_point_cloud_o3d.voxel_down_sample(
                voxel_size=voxel_size
            )
        else:
            print("Downsampling depth images")
            downsampled_point_cloud = merged_point_cloud_o3d

        # 保存点云
        if len(downsampled_point_cloud.colors) > 0:
            points = np.hstack(
                [
                    np.asarray(downsampled_point_cloud.points),
                    np.asarray(downsampled_point_cloud.colors) * 255,
                ]
            )
        else:
            points = np.asarray(downsampled_point_cloud.points)
        write_point_plyfile(
            points, os.path.join(self.colmap_sparse_dir, "points3D.ply")
        )
        return points

    def convert_depth_merge_point_clouds(
        self,
        cam_ids: List[int] = [0],
        bgr_target: Tuple = (0, 0, 0),
        need: bool = True,
        downsample: bool = True,
        voxel_size: float = 0.1,
        with_rgb: bool = True,
        point_cloud_format: str = "ply",
        DEPTH_MAX: float = 655.35,
    ):
        print("Converting depth to pointclouds and merging...")
        # 读取深度图,转为点云，将所有点云合并，并且进行降采样
        merged_point_cloud_o3d = (
            o3d.geometry.PointCloud()
        )  # 由于numpy数据太大，所以使用open3d的数据结构
        for cam_id in cam_ids:
            depth_data = self.scene_data.frames.depth[cam_id]
            # 获取静止的车辆的ID
            static_car_ids = self.scene_data.bbox.get_all_static_car(cam_id=cam_id)
            with Progress() as progress:
                task = progress.add_task(
                    f"Converting depth to pointclouds for cam{cam_id}",
                    total=self.end_frame - self.start_frame + 1,
                )
                for i in range(self.start_frame, self.end_frame + 1):
                    this_instance_data = self.scene_data.frames.instanceSegmentation[
                        cam_id
                    ][i]
                    if bgr_target == (0, 0, 0):
                        # 如果是背景，那么需要考虑背景中的静态车辆，
                        # 需要把静态车辆在实例分割中的bgr_target去掉，变成0,0,0
                        # 遍历所有的trackID，如果是静态车辆，就把它的bgr_target去掉
                        for _trackID in static_car_ids:
                            _static_mask = np.all(
                                this_instance_data == self.get_bgr_target(_trackID),
                                axis=2,
                            )
                            this_instance_data[_static_mask] = 0

                    extracted_depth = extract_instance_depth(
                        depth_data[i],
                        this_instance_data,
                        bgr_target=bgr_target,
                        need=need,
                    )
                    point_cloud = depth_image_to_point_cloud(
                        extracted_depth,
                        self.scene_data.intrinsics[i][cam_id],
                        self.scene_data.extrinsics[i][cam_id],
                        rgb=self.scene_data.frames.rgb[cam_id][i] if with_rgb else None,
                        DEPTH_MAX=DEPTH_MAX,
                    )
                    if len(point_cloud) > 0:
                        point_cloud_o3d = o3d.geometry.PointCloud()
                        point_cloud_o3d.points = o3d.utility.Vector3dVector(
                            point_cloud[:, 0:3]
                        )
                        if len(point_cloud[0]) == 6:
                            # 有颜色信息
                            point_cloud_o3d.colors = o3d.utility.Vector3dVector(
                                point_cloud[:, 3:6] / 255.0
                            )
                            pass
                        merged_point_cloud_o3d += point_cloud_o3d
                    else:
                        pass
                    progress.update(task, advance=1)
                    pass  # end for i
                pass  # end with
            pass  # end for cam_id

        # 降采样
        if downsample:
            print("Downsampling...")
            downsampled_point_cloud = merged_point_cloud_o3d.voxel_down_sample(
                voxel_size
            )
        else:
            print("不降采样了")
            downsampled_point_cloud = merged_point_cloud_o3d

        # 保存点云
        print(f"Saving point cloud as {point_cloud_format} file...")
        if point_cloud_format == "ply":
            # o3d.io.write_point_cloud(
            #     os.path.join(self.colmap_sparse_dir, "points3D.ply"),
            #     downsampled_point_cloud,
            #     write_ascii=True,
            #     print_progress=True,
            # )
            if len(downsampled_point_cloud.colors) > 0:
                points = np.hstack(
                    [
                        np.asarray(downsampled_point_cloud.points),
                        np.asarray(downsampled_point_cloud.colors) * 255,
                    ]
                )
            else:
                points = np.asarray(downsampled_point_cloud.points)
            write_point_plyfile(
                points,
                os.path.join(self.colmap_sparse_dir, "points3D.ply"),
            )

        elif point_cloud_format == "txt":
            # 把points和colors合并到一起
            if len(downsampled_point_cloud.colors) > 0:
                points = np.hstack(
                    [
                        np.asarray(downsampled_point_cloud.points),
                        np.asarray(downsampled_point_cloud.colors) * 255,
                    ]
                )
            else:
                points = np.asarray(downsampled_point_cloud.points)
            write_point_cloud(
                points,
                os.path.join(self.colmap_sparse_dir, "points3D.txt"),
            )
        else:
            raise ValueError(f"Unknown point cloud format: {point_cloud_format}")
        return

    def merge_lidar_point_clouds(
        self,
        downsample: bool = True,
        voxel_size: float = 0.1,
    ):
        # 读取激光雷达点云，将所有点云合并，并且进行降采样
        merged_point_cloud_o3d = o3d.geometry.PointCloud()
        with Progress() as progress:
            task = progress.add_task(
                f"merging lidar point clouds!",
                total=len(self.scene_data.frames.lidar),
            )
            for lidar in self.scene_data.frames.lidar:
                merged_point_cloud_o3d += lidar
                progress.update(task, advance=1)
        if downsample:
            print("Downsampling!")
            downsampled_point_cloud = merged_point_cloud_o3d.voxel_down_sample(
                voxel_size=voxel_size
            )
        else:
            downsampled_point_cloud = merged_point_cloud_o3d
        print("Writing points to points3D.ply!")
        write_point_plyfile(
            np.asarray(downsampled_point_cloud.points),
            os.path.join(self.colmap_sparse_dir, "points3D.ply"),
        )
        return

    def extract_lidar_points_using_3dbbox(
        self,
        trackID: int = -1,
        downsample: bool = True,
        voxel_size: float = 0.1,
    ):
        lidar_data = self.scene_data.frames.lidar
        extrinsics = self.scene_data.lidar_extrinsics  # 这里外参应当使用world_to_liadr
        pose = self.scene_data.pose
        # 当trackID为-1时，提取背景点云
        # 提取背景点云的时候，遍历所有lidar点云，去除其中车辆的部分，然后合并
        merged_point_cloud_o3d = o3d.geometry.PointCloud()
        if trackID == -1:
            with Progress() as progress:
                task = progress.add_task(
                    "Extracting background point cloud!",
                    total=self.end_frame - self.start_frame + 1,
                )
                for frame in range(self.start_frame, self.end_frame + 1):
                    progress.update(task, advance=1)
                    # 将点云从ego car坐标系转到世界坐标系
                    R = extrinsics.get_R(frame)
                    T = extrinsics.get_t(frame)
                    world_to_car = np.eye(4)
                    world_to_car[:3, :3] = R
                    world_to_car[:3, 3] = T
                    car_to_world = np.linalg.inv(world_to_car)
                    points_world_o3d = lidar_data[frame].transform(car_to_world)

                    # 将这一帧点云中的动态车辆部分去除
                    # 遍历每一个3d框，将3d框内的点云去除
                    np_points = np.asarray(points_world_o3d.points)
                    for _tmp_pose in pose[frame]:
                        np_points = extract_np_points_using_3dbbox(
                            np_points,
                            pose=_tmp_pose,
                            need=False,
                        )
                    points_world_o3d.points = o3d.utility.Vector3dVector(np_points)
                    merged_point_cloud_o3d += points_world_o3d
        else:
            # 提取车辆点云
            with Progress() as progress:
                task = progress.add_task(
                    "Extracting car point cloud!",
                    total=self.end_frame - self.start_frame + 1,
                )
                for frame in range(self.start_frame, self.end_frame + 1):
                    progress.update(task, advance=1)
                    # 先判断这一帧是否有这个trackID的车辆
                    if trackID not in pose[frame]["trackID"]:
                        continue
                    # 先将点云转到世界坐标系
                    R = extrinsics.get_R(frame)
                    T = extrinsics.get_t(frame)
                    world_to_car = np.eye(4)
                    world_to_car[:3, :3] = R
                    world_to_car[:3, 3] = T
                    car_to_world = np.linalg.inv(world_to_car)
                    points_world_o3d = lidar_data[frame].transform(car_to_world)

                    # 获取这一帧中的车辆点云
                    np_points = np.asarray(points_world_o3d.points)
                    np_points = extract_np_points_using_3dbbox(
                        np_points,
                        pose=pose[frame][pose[frame]["trackID"] == trackID][0],
                        need=True,
                    )

                    points_car_o3d = o3d.geometry.PointCloud()
                    points_car_o3d.points = o3d.utility.Vector3dVector(np_points)
                    merged_point_cloud_o3d += points_car_o3d

        if downsample:
            print("Downsampling!")
            downsampled_point_cloud = merged_point_cloud_o3d.voxel_down_sample(
                voxel_size=voxel_size
            )
        else:
            downsampled_point_cloud = merged_point_cloud_o3d
        print("Writing points to points3D.ply!")
        write_point_plyfile(
            np.asarray(downsampled_point_cloud.points),
            os.path.join(self.colmap_sparse_dir, "points3D.ply"),
        )
        return np.asarray(downsampled_point_cloud.points)

    def extract_instance_images(
        self,
        bgr_target: Tuple[int, int, int] = (65, 137, 0),
        trackID: int = -1,
        cam_id_list: List[int] = [0],
        white_bg: bool = False,
    ):
        if trackID != -1:
            bgr_target = self.get_bgr_target(trackID=trackID)
        destination_images_path = os.path.join(self.gaussian_dir, "images")
        os.makedirs(destination_images_path, exist_ok=True)
        for cam_id in cam_id_list:
            rgb_data = self.scene_data.frames.rgb[cam_id]
            instance_data = self.scene_data.frames.instanceSegmentation[cam_id]
            static_car_ids = self.scene_data.bbox.get_all_static_car(cam_id=cam_id)
            with Progress() as progress:
                task = progress.add_task(
                    f"Extracting instance images for car{trackID} in cam{cam_id}",
                    total=self.end_frame - self.start_frame + 1,
                )
                for i in range(self.start_frame, self.end_frame + 1):
                    progress.update(task, advance=1)
                    # 当目标不是背景，并且需要提取目标的时候，提取的图片会有大量空白，此时进行判断，空白图片不保存
                    # 目标是提取背景的时候，所有图片都会保存，跳过这个判断，减少计算量
                    if bgr_target != (0, 0, 0) and not np.any(
                        np.all(
                            # 判断实例分割中是否存在bgr_target
                            instance_data[i] == bgr_target,
                            axis=2,
                            keepdims=True,
                        )
                    ):  # 如果全都不等于bgr_target，就跳过
                        # 判断背景的核心逻辑
                        # print(f"第{i}张图片没有实例，跳过")
                        continue

                    this_instance_data = instance_data[i]
                    if trackID == -1:
                        # 如果是背景，那么需要考虑背景中的静态车辆，
                        # 需要把静态车辆在实例分割中的bgr_target去掉，变成0,0,0
                        # 遍历所有的trackID，如果是静态车辆，就把它的bgr_target去掉
                        for _trackID in static_car_ids:
                            _static_mask = np.all(
                                this_instance_data == self.get_bgr_target(_trackID),
                                axis=2,
                            )  # 需要保证所有的通道都相等，才能确定这个索引是静态车辆在实例分割中对应的位置
                            this_instance_data[_static_mask] = 0

                    output_img = extract_instance_image(
                        rgb_data[i],
                        this_instance_data,
                        bgr_target=bgr_target,
                        white_bg=white_bg,
                    )
                    cv2.imwrite(
                        os.path.join(
                            destination_images_path,
                            rgb_data.pic_paths[i].replace(
                                "jpg", "png"
                            ),  # 保存为png格式
                        ),
                        output_img,
                    )
        return

    def copy_images_to_sparse(self, cam_id: int = 0, target_suffix="png"):
        # 把图片复制到gaussian/images文件夹下
        rgb = self.scene_data.frames.rgb[cam_id]
        rgb_data_path = rgb.dir_path
        print(f"正在复制{rgb_data_path}下的图片到{self.gaussian_dir}/images文件夹下")
        assert os.path.exists(rgb_data_path)
        destination_path = os.path.join(self.gaussian_dir, "images")
        if os.path.exists(destination_path):
            print(f"{destination_path}已经存在，将会覆盖")
        else:
            os.makedirs(destination_path, exist_ok=True)
        for _frame in range(self.start_frame, self.end_frame + 1):
            src = rgb.get_pic_path(index=_frame)
            dst = os.path.join(
                destination_path, os.path.basename(src).replace("jpg", target_suffix)
            )
            shutil.copy(src=src, dst=dst)
        return

    def generate_colmap_images_txt(self, cam_id: int = 0):
        # 生成colmap的images.txt文件
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        images_txt_path = os.path.join(self.colmap_sparse_dir, "images.txt")
        rgb = self.scene_data.frames.rgb[cam_id]
        extrinsics = self.scene_data.extrinsics
        with open(images_txt_path, "w") as f:
            for i in range(self.start_frame, self.end_frame + 1):
                quaternion, t = extrinsics.convert_extrinsic_to_quaternion_and_t(
                    index=i, cam_id=cam_id
                )
                IMAGE_ID = i + 1  # 照片的id从1开始
                # scipy返回四元数的顺序是x,y,z,w,https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html#scipy.spatial.transform.Rotation.as_quat
                QX, QY, QZ, QW = quaternion  # 这里QW是实数部分
                TX, TY, TZ = t
                CAMERA_ID = i + 1  # 相机的id从1开始
                NAME = rgb.pic_paths[i].replace("jpg", "png")
                f.write(
                    f"{IMAGE_ID} {QW} {QX} {QY} {QZ} {TX} {TY} {TZ} {CAMERA_ID} {NAME}\n\n"
                )
        return

    def prepare_colmap_images_txt_for_only_car(
        self, cam_id: int = 0, trackID: int = 0, focous_frame: int = 26
    ):
        # 只生成车辆的colmap的images.txt文件
        images_txt_path = os.path.join(self.colmap_sparse_dir, "images.txt")
        rgb = self.scene_data.frames.rgb[cam_id]
        pose = self.scene_data.pose
        extrinsic_dict = dict()
        _extrinsic_data = self.scene_data.extrinsics
        with open(images_txt_path, "w") as f:
            for i in range(self.start_frame, self.end_frame + 1):
                # # 使用复杂的公式
                # car2w = pose.get_car2w(index=i, trackID=trackID)
                # if car2w is None:
                #     continue
                # w2cam = np.zeros((4, 4))
                # w2cam[:3, :3] = _extrinsic_data.get_R(index=i, cam_id=cam_id)
                # w2cam[:3, 3] = _extrinsic_data.get_t(index=i, cam_id=cam_id)
                # w2cam[3, 3] = 1
                # new_extrinsic = np.dot(w2cam, np.dot(car2w, w2car_focous))

                # 使用简单的公式
                car2cam = pose.get_car2cam(index=i, trackID=trackID, cam_id=cam_id)
                if car2cam is None:
                    continue
                # new_extrinsic = np.dot(car2cam, w2car_focous)

                # 对于车辆点云使用车辆坐标系来说，新的外参应该使用car2cam
                new_extrinsic = car2cam

                extrinsic_dict[i] = new_extrinsic
                quaternion, t = convert_extrinsic_to_quaternion_and_t(new_extrinsic)
                IMAGE_ID = i + 1  # 照片的id从1开始
                QX, QY, QZ, QW = quaternion
                TX, TY, TZ = t
                CAMERA_ID = i + 1  # 相机的id从1开始
                NAME = rgb.pic_paths[i].replace("jpg", "png")
                f.write(
                    f"{IMAGE_ID} {QW} {QX} {QY} {QZ} {TX} {TY} {TZ} {CAMERA_ID} {NAME}\n\n"
                )
        return extrinsic_dict

    def generate_colmap_cameras_txt(self, cam_id: int = 0, model: str = "PINHOLE"):
        # 生成colmap的cameras.txt文件
        # Camera list with one line of data per camera:
        #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # Number of cameras: 3
        # 1 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
        # 2 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
        # 3 SIMPLE_RADIAL 3072 2304 2559.69 1536 1152 -0.0218531
        cameras_txt_path = os.path.join(self.colmap_sparse_dir, "cameras.txt")
        intrinsics = self.scene_data.intrinsics
        _tmp_img = self.scene_data.frames.rgb[cam_id][0]
        WIDTH, HEIGHT = _tmp_img.shape[1], _tmp_img.shape[0]

        cameras_dict = dict()
        with open(cameras_txt_path, "w") as f:
            # vkitti的数据考虑采用pinhole模型
            for i in range(len(intrinsics)):
                CAMERA_ID = i + 1
                MODEL = model
                PARAMS = intrinsics[i][cam_id][["K[0,0]", "K[1,1]", "K[0,2]", "K[1,2]"]]
                cameras_dict[CAMERA_ID] = PARAMS
                # print(PARAMS)
                f.write(
                    f"{CAMERA_ID} {MODEL} {WIDTH} {HEIGHT} {' '.join(map(str, PARAMS))}\n"
                )
        return cameras_dict

    def generate_colmap_content(
        self,
        cam_id: int = 0,
        white_bg: bool = False,
        use_lidar: bool = False,
        DEPTH_MAX: float = 655.35,
        downsample: bool = True,
        voxel_size: float = 0.1,
        background_image_only: bool = True,
    ):
        # 生成colmap的sparse文件夹下的内容
        if background_image_only:
            self.extract_instance_images(
                bgr_target=(0, 0, 0), cam_id_list=[cam_id], white_bg=white_bg
            )
        else:
            self.copy_images_to_sparse(cam_id)
        self.generate_colmap_images_txt(cam_id)
        self.generate_colmap_cameras_txt(cam_id)
        if use_lidar:
            # self.merge_lidar_point_clouds()
            self.extract_lidar_points_using_3dbbox(downsample=downsample)
        else:
            self.convert_depth_merge_point_clouds(
                cam_ids=[cam_id],
                bgr_target=(0, 0, 0),
                need=True,
                downsample=downsample,
                voxel_size=voxel_size,
                with_rgb=True,
                point_cloud_format="ply",
                DEPTH_MAX=DEPTH_MAX,
            )  # 生成全背景的点云
        return

    def generate_colmap_content_for_only_car(
        self,
        cam_id: int = 0,
        trackID: int = 0,
        verify_extrinsic: bool = False,
        focous_frame: int = 26,
        white_bg: bool = False,
        car_downsample: bool = True,
        use_lidar: bool = False,
    ):
        # TODO：做一个交互，当不输入focous_frame的时候，根据trackID，输出所有可行的frames供选择
        # 生成colmap的sparse文件夹下的内容
        extrinsic_dict = self.prepare_colmap_images_txt_for_only_car(
            cam_id, trackID, focous_frame
        )
        cameras_dict = self.generate_colmap_cameras_txt(cam_id)
        self.extract_instance_images(
            trackID=trackID, cam_id_list=[cam_id], white_bg=white_bg
        )

        if use_lidar:
            point_cloud = self.extract_lidar_points_using_3dbbox(
                trackID=trackID,
                downsample=car_downsample,
                voxel_size=0.01,
            )
        else:
            point_cloud = self.convert_depth_merge_point_clouds_for_car(
                cam_id=cam_id,
                trackID=trackID,
                focous_frame=focous_frame,
                downsample=car_downsample,
            )
        print(f"car{trackID} point_cloud.shape:{point_cloud.shape}")

        # 验证点云使用外参投影到像素坐标是否与原图匹配
        if verify_extrinsic:
            self.generate_images_from_point_cloud_with_new_extrinsic(
                extrinsic_dict=extrinsic_dict,
                cameras_dict=cameras_dict,
                point_cloud=point_cloud[:, 0:3],
                trackID=trackID,
                cam_id=cam_id,
            )
        return

    def generate_colmap_content_for_one_class(
        self,
        category: str,
        cam_id: int = 0,
        downsample: bool = True,
        white_bg: bool = False,
    ):
        # 生成特定类别的colmap内容
        self._set_output_dir(category)
        self.prepare_colmap_images_txt_for_one_class(category, cam_id)
        self.generate_colmap_cameras_txt_for_one_class(category, cam_id)
        self.extract_instance_images_for_one_class(category, cam_id, white_bg=white_bg)
        self.convert_depth_merge_point_clouds_for_one_class(
            category, cam_id, downsample
        )

    def prepare_colmap_images_txt_for_one_class(
        self,
        category: str,
        cam_id: int = 0,
    ):
        images_txt_path = os.path.join(self.colmap_sparse_dir, "images.txt")
        rgb = self.scene_data.frames.rgb[cam_id]
        extrinsics = self.scene_data.extrinsics
        class_segmentation = self.scene_data.frames.classSegmentation[cam_id]
        class_rgb = self.get_rgb_of_category(category)
        with open(images_txt_path, "w") as f:
            with Progress() as progress:
                task = progress.add_task(
                    f"Generating colmap images.txt for {category} in cam{cam_id}",
                    total=self.end_frame - self.start_frame + 1,
                )
                for i in range(self.start_frame, self.end_frame + 1):
                    progress.update(task, advance=1)
                    _class_segmentation_rgb = cv2.cvtColor(
                        class_segmentation[i], cv2.COLOR_BGR2RGB
                    )
                    if not np.any(
                        np.all(
                            _class_segmentation_rgb == class_rgb, axis=2, keepdims=True
                        )
                    ):
                        # 如果这一帧没有这个类别，跳过
                        continue
                    quaternion, t = extrinsics.convert_extrinsic_to_quaternion_and_t(
                        index=i, cam_id=cam_id
                    )
                    IMAGE_ID = i + 1
                    QX, QY, QZ, QW = quaternion
                    TX, TY, TZ = t
                    CAMERA_ID = i + 1
                    NAME = rgb.pic_paths[i].replace("jpg", "png")
                    f.write(
                        f"{IMAGE_ID} {QW} {QX} {QY} {QZ} {TX} {TY} {TZ} {CAMERA_ID} {NAME}\n\n"
                    )

    def generate_colmap_cameras_txt_for_one_class(
        self, category: str, cam_id: int = 0, model: str = "PINHOLE"
    ):
        # 返回所有的相机内参
        self.generate_colmap_cameras_txt(cam_id, model)

    def extract_instance_images_for_one_class(
        self,
        category: str,
        cam_id: int = 0,
        white_bg: bool = False,
    ):
        # 提取特定类别的图片
        destination_images_path = os.path.join(self.gaussian_dir, "images")
        os.makedirs(destination_images_path, exist_ok=True)
        rgb_data = self.scene_data.frames.rgb[cam_id]
        class_segmentation = self.scene_data.frames.classSegmentation[cam_id]
        class_rgb = np.asarray(self.get_rgb_of_category(category))
        print("class_rgb:", class_rgb)
        with Progress() as progress:
            task = progress.add_task(
                f"Extracting instance images for {category} in cam{cam_id}",
                total=self.end_frame - self.start_frame + 1,
            )
            for i in range(self.start_frame, self.end_frame + 1):
                _class_segmentation_rgb = cv2.cvtColor(
                    class_segmentation[i], cv2.COLOR_BGR2RGB
                )
                if not np.any(
                    np.all(_class_segmentation_rgb == class_rgb, axis=2, keepdims=True)
                ):
                    # 判断是否有这个类别，很重要,内层的all是针对像素的rgb要all相等，外层的any是针对是否有这个类别
                    # 如果这一帧没有这个类别，跳过
                    # print("没有这个类别")
                    progress.update(task, advance=1)
                    continue

                output_img = extract_class_image(
                    rgb_data[i], _class_segmentation_rgb, class_rgb, white_bg=white_bg
                )
                cv2.imwrite(
                    os.path.join(
                        destination_images_path,
                        rgb_data.pic_paths[i].replace("jpg", "png"),
                    ),
                    output_img,
                )
                progress.update(task, advance=1)

    def convert_depth_merge_point_clouds_for_one_class(
        self,
        category: str,
        cam_id: int = 0,
        downsample: bool = True,
        voxel_size: float = 0.1,
    ):
        # 生成特定类别的点云
        class_segmentation = self.scene_data.frames.classSegmentation[cam_id]
        class_rgb = np.asarray(self.get_rgb_of_category(category))
        depth_data = self.scene_data.frames.depth[cam_id]
        intrinsics = self.scene_data.intrinsics
        extrinsics = self.scene_data.extrinsics
        rgb_data = self.scene_data.frames.rgb[cam_id]
        merged_point_cloud_o3d = o3d.geometry.PointCloud()
        with Progress() as progress:
            task = progress.add_task(
                f"Converting depth to pointclouds for {category} in cam{cam_id}",
                total=self.end_frame - self.start_frame + 1,
            )
            for i in range(self.start_frame, self.end_frame + 1):
                _class_segmentation_rgb = cv2.cvtColor(
                    class_segmentation[i], cv2.COLOR_BGR2RGB
                )
                if not np.any(
                    np.all(_class_segmentation_rgb == class_rgb, axis=2, keepdims=True)
                ):
                    # 如果这一帧没有这个类别，跳过
                    progress.update(task, advance=1)
                    continue
                extracted_depth = extract_class_depth(
                    depth_data[i], _class_segmentation_rgb, class_rgb
                )
                point_cloud = depth_image_to_point_cloud(
                    extracted_depth,
                    intrinsics[i][cam_id],
                    extrinsics[i][cam_id],
                    rgb=rgb_data[i],
                )
                if len(point_cloud) > 0:
                    point_cloud_o3d = o3d.geometry.PointCloud()
                    point_cloud_o3d.points = o3d.utility.Vector3dVector(
                        point_cloud[:, 0:3]
                    )
                    if len(point_cloud[0]) == 6:
                        point_cloud_o3d.colors = o3d.utility.Vector3dVector(
                            point_cloud[:, 3:6] / 255.0
                        )
                        pass
                    merged_point_cloud_o3d += point_cloud_o3d
                progress.update(task, advance=1)
                pass

        # 降采样
        if downsample:
            print("Downsampling depth images")
            downsampled_point_cloud = merged_point_cloud_o3d.voxel_down_sample(
                voxel_size=voxel_size
            )
        else:
            print("Downsampling depth images")
            downsampled_point_cloud = merged_point_cloud_o3d

        # 保存点云
        if len(downsampled_point_cloud.colors) > 0:
            points = np.hstack(
                [
                    np.asarray(downsampled_point_cloud.points),
                    np.asarray(downsampled_point_cloud.colors) * 255,
                ]
            )
        else:
            points = np.asarray(downsampled_point_cloud.points)
        write_point_plyfile(
            points, os.path.join(self.colmap_sparse_dir, "points3D.ply")
        )

    def prepare_all_colmap_contents(
        self,
        cam_id: int = 0,
        white_bg: bool = False,
        car_downsample: bool = True,
        background_downsample: bool = True,
        background_downsample_voxel_size: float = 0.1,
        use_lidar=False,
        BACKGROUND_DEPTH_MAX: float = 655.35,
        background_image_only: bool = True,
    ):
        # 生成背景的colmap内容
        self._set_output_dir("background")
        self.generate_colmap_content(
            cam_id,
            white_bg=white_bg,
            use_lidar=use_lidar,
            DEPTH_MAX=BACKGROUND_DEPTH_MAX,
            downsample=background_downsample,
            voxel_size=background_downsample_voxel_size,
            background_image_only=background_image_only,
        )

        # 在bbox中搜索一共有多少个trackID
        dynamic_car_ids = self.scene_data.bbox.get_all_moving_car(cam_id=cam_id)
        print("Generating colmap content for all dynamic trackIDs")
        for _trackID in dynamic_car_ids:
            print(_trackID)
            self._set_output_dir(f"trackID_{_trackID}")
            self.generate_colmap_content_for_only_car(
                cam_id=cam_id,
                trackID=_trackID,
                white_bg=white_bg,
                car_downsample=car_downsample,
                use_lidar=use_lidar,
            )
            # 如果生成的动态车辆点云为空，删除这个文件夹
            # 因为这个车的点云数量太少了（基本上都是远景的视图），几乎没有训练的价值
            # 通过images文件夹为空来判断，在vkitti数据上，基本与点云数量为空等价
            if len(os.listdir(os.path.join(self.gaussian_dir, "images"))) == 0:
                shutil.rmtree(self.gaussian_dir)
                pass
            pass  # end for _trackID

        return

    def generate_images_from_point_cloud_with_new_extrinsic(
        self,
        extrinsic_dict: dict,
        cameras_dict: dict,
        point_cloud: np.ndarray,
        trackID: int = 0,
        cam_id: int = 0,
    ):
        # 生成新的图片
        images_new_path = os.path.join(self.gaussian_dir, "images_new_extrinsic")
        os.makedirs(images_new_path, exist_ok=True)
        rgb_data = self.scene_data.frames.rgb[cam_id]
        # 将点云按外参，内参，投影到像素坐标

        assert point_cloud.shape[1] == 3
        # 世界坐标系下的点云
        _points_world = point_cloud
        # 转为齐次坐标
        _points_world = np.hstack([_points_world, np.ones((_points_world.shape[0], 1))])
        assert _points_world.shape[1] == 4
        print(f"shape of _points_world: {_points_world.shape}")
        _points_world = _points_world.T  # 为了后面使用矩阵乘法，从n*4变为4*n
        b, g, r = self.get_bgr_target(trackID=trackID)
        print(f"bgr: {b}, {g}, {r}")
        with Progress() as progress:
            task = progress.add_task(
                f"Generating new images from point cloud for track{trackID}",
                total=len(extrinsic_dict),
            )
            for (
                frame_id,
                _extrinsic,
            ) in extrinsic_dict.items():  # 这个字典只存了有车的帧的外参
                # _extrinsic 是world2cam的4x4矩阵
                fx, fy, cx, cy = cameras_dict[frame_id]
                _instrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                # 获得相机坐标系下的点云
                _points_camera = np.dot(_extrinsic, _points_world)
                # 投影到像素坐标
                _points_pixel = np.dot(_instrinsic, _points_camera[0:3, :])
                # 归一化
                # print(f"shape of _points_pixel: {_points_pixel.shape}, {np.int16(_points_pixel)}")
                _points_pixel = _points_pixel / _points_pixel[2, :]
                # print(f"shape of _points_pixel: {_points_pixel.shape}, {np.int16(_points_pixel)}")
                # return
                # 生成图片
                # 在本来的rgb图片对应的位置上，填充为实例分割的颜色
                # print(f"正在生成第{frame_id}帧填充好的的图片")
                img = rgb_data[frame_id].copy()
                # img=np.zeros((rgb_data[frame_id].shape[0],rgb_data[frame_id].shape[1],3),dtype=np.uint8)
                for i in range(_points_pixel.shape[1]):
                    x, y = int(_points_pixel[0, i]), int(_points_pixel[1, i])
                    # print(f"x: {x}, y: {y}")
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        # print(f"填充点：{x},{y}")
                        img[y, x] = np.array([b, g, r])
                        # exit()
                        pass  # end if
                    pass  # end for i
                # 保存为jpg
                cv2.imwrite(
                    os.path.join(images_new_path, rgb_data.pic_paths[frame_id]), img
                )
                progress.update(task, advance=1)
                pass  # end for frame_id
        return


def _extract_car(
    scene_path: str,
    trackID: int,
    cam_id: int = 0,
    white_bg: bool = False,
    car_downsample: bool = True,
    use_lidar: bool = False,
    verify_extrinsic: bool = False,
    output_dir: str = None,
    start_frame: int = 0,
    end_frame: int = -1,
):
    """
    将指定车辆的数据转为colmap格式
    Args:
        scene_path:  path to vkitti scene, example: /home/lsin/Public/vkitti/Scene18/clone
        trackID:  trackID of the car
        cam_id:  camera id
    """
    scene_data = VkittiSceneData(scene_path)
    gaussian_converter = GaussianConverter(
        scene_data, output_dir=output_dir, start_frame=start_frame, end_frame=end_frame
    )
    gaussian_converter._set_output_dir(f"trackID_{trackID}")
    # gaussian_converter.extract_instance_images(trackID=trackID, cam_id_list=[cam_id])
    gaussian_converter.generate_colmap_content_for_only_car(
        cam_id=cam_id,
        trackID=trackID,
        white_bg=white_bg,
        car_downsample=car_downsample,
        use_lidar=use_lidar,
        verify_extrinsic=verify_extrinsic,
    )
    return


def extract_car():
    tyro.cli(_extract_car)


def _extract_background(
    scene_path: str,
    cam_id: int = 0,
    white_bg: bool = False,
    use_lidar: bool = False,
    BACKGROUND_DEPTH_MAX: float = 655.35,
    output_dir: str = None,
    downsample: bool = True,
    voxel_size: float = 0.1,
    start_frame: int = 0,
    end_frame: int = -1,
    background_image_only: bool = True,
):
    """
    传入scene-path和cam-id（可选），
    会在scene-path下生成只包含背景的colmap格式数据集，存在scene-path/gaussian文件夹，包含images文件夹和sparse文件夹。
    scene-path示例：/home/lsin/Public/vkitti/Scene18/clone
    """
    scene_data = VkittiSceneData(scene_path)
    gaussian_converter = GaussianConverter(
        scene_data, output_dir=output_dir, start_frame=start_frame, end_frame=end_frame
    )
    gaussian_converter._set_output_dir("background")
    gaussian_converter.generate_colmap_content(
        cam_id=cam_id,
        white_bg=white_bg,
        use_lidar=use_lidar,
        DEPTH_MAX=BACKGROUND_DEPTH_MAX,
        downsample=downsample,
        voxel_size=voxel_size,
        background_image_only=background_image_only,
    )
    return


def extract_background():
    tyro.cli(_extract_background)


def _convert_all_vkitti_to_gaussian(
    scene_path: str,
    cam_id: int = 0,
    white_bg: bool = False,
    car_downsample: bool = True,
    background_downsample: bool = True,
    background_downsample_voxel_size: float = 0.1,
    use_lidar: bool = False,
    BACKGROUND_DEPTH_MAX: float = 655.35,  # 背景深度最大值（单位是米）
    output_dir: str = None,
    start_frame: int = 0,
    end_frame: int = -1,
    background_image_only: bool = True,
):
    """
    传入scene-path和cam-id（可选），
    会在scene-path下生成只包含背景的colmap格式数据集，存在scene-path/background文件夹，包含images文件夹和sparse文件夹。
    同时会按照trackID生成包含车辆的colmap格式数据集，存在scene-path/trackID_{trackID}文件夹，包含images文件夹和sparse文件夹。
    scene-path示例：/home/lsin/Public/vkitti/Scene18/clone
    """
    scene_data = VkittiSceneData(scene_path)
    gaussian_converter = GaussianConverter(
        scene_data, output_dir=output_dir, start_frame=start_frame, end_frame=end_frame
    )
    gaussian_converter.prepare_all_colmap_contents(
        cam_id=cam_id,
        white_bg=white_bg,
        car_downsample=car_downsample,
        use_lidar=use_lidar,
        BACKGROUND_DEPTH_MAX=BACKGROUND_DEPTH_MAX,
        background_downsample=background_downsample,
        background_downsample_voxel_size=background_downsample_voxel_size,
        background_image_only=background_image_only,
    )
    return


def convert_all_vkitti_to_gaussian():
    tyro.cli(_convert_all_vkitti_to_gaussian)


def _convert_one_class_to_gaussian(
    scene_path: str,
    category: str,
    cam_id: int = 0,
    dowmsample: bool = True,
    white_bg: bool = False,
    ouput_dir: str = None,
    start_frame: int = 0,
    end_frame: int = -1,
):
    """
    Convert one class in classSegementation to Colmap like dataset
    Args:
        scene_path:  path to vkitti scene, example: /home/lsin/Public/vkitti/Scene18/clone
        category:  category name, example: "Terrain"
        cam_id:  camera id
        dowmsample:  downsample the point cloud
    """
    scene_data = VkittiSceneData(scene_path)
    if category not in scene_data.colors.data:
        print(f"Category {category} not in classSegmentation")
        print(f"Available categories: {scene_data.colors.data.keys()}")
        return
    gaussian_converter = GaussianConverter(
        scene_data, output_dir=ouput_dir, start_frame=start_frame, end_frame=end_frame
    )
    gaussian_converter.generate_colmap_content_for_one_class(
        category=category, cam_id=cam_id, downsample=dowmsample, white_bg=white_bg
    )


def convert_one_class_to_gaussian():
    tyro.cli(_convert_one_class_to_gaussian)


def _convert_all_classes_to_gaussian(
    scene_path: str,
    cam_id: int = 0,
    dowmsample: bool = True,
    white_bg: bool = False,
    output_dir: str = None,
    start_frame: int = 0,
    end_frame: int = -1,
):
    """
    Convert all classes in classSegementation to Colmap like dataset
    Args:
        scene_path:  path to vkitti scene, example: /home/lsin/Public/vkitti/Scene18/clone
        cam_id:  camera id
        dowmsample:  downsample the point cloud
    """
    scene_data = VkittiSceneData(scene_path)
    gaussian_converter = GaussianConverter(
        scene_data, output_dir=output_dir, start_frame=start_frame, end_frame=end_frame
    )
    for category in scene_data.colors.data.keys():
        gaussian_converter.generate_colmap_content_for_one_class(
            category=category,
            cam_id=cam_id,
            downsample=dowmsample,
            white_bg=white_bg,
        )


def convert_all_classes_to_gaussian():
    tyro.cli(_convert_all_classes_to_gaussian)
