import numpy as np
import cv2
from typing import Tuple, List, Dict
import abc
import os
import re
from scipy.spatial.transform import Rotation


class PictureData(abc.ABC):
    # 定义抽象类
    def __init__(self, dir_path: str, cam_id: int = 0) -> None:
        # 后缀为png
        self.endwith: str = "png"
        self.dir_path: str = dir_path
        self.cam_id: int = cam_id
        self.pic_paths: List[str] = os.listdir(dir_path)
        # 文件名形如：instancegt_00050.png
        self.pic_paths.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        return

    # 实现PictureData[]的方法
    def __getitem__(self, index: int) -> np.ndarray:
        return self.get_pic(index)

    def get_pic_path(self, index: int) -> str:
        return os.path.join(self.dir_path, self.pic_paths[index])

    def get_pic(self, index: int) -> np.ndarray:
        # 深度图是0-65535的uint16
        # RGB图是uint8的，三通道的
        return cv2.imread(
            self.get_pic_path(index), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
        )
        # if self.depth:
        #     return cv2.imread(self.get_pic_path(index), cv2.IMREAD_UNCHANGED)
        # else:
        #     return cv2.imread(self.get_pic_path(index))

    def __len__(self) -> int:
        return len(self.pic_paths)

    def cv_show(self, index: int) -> None:
        cv2.imshow("Output", self[index])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class StereoPictureData:
    def __init__(self, dir_path: str) -> None:
        self.dir_path: str = dir_path
        self.cam_num: int = len(os.listdir(dir_path))
        return

    def __len__(self) -> int:
        return self.cam_num

    def __getitem__(self, index: int) -> PictureData:
        return PictureData(os.path.join(self.dir_path, f"Camera_{index}"), cam_id=index)


class FramesData:
    def __init__(self, dir_path: str) -> None:
        self.dir_path: str = dir_path
        # 有这些类型的frames classSegmentation  depth  instanceSegmentation  rgb
        self.classSegmentation: StereoPictureData = StereoPictureData(
            os.path.join(dir_path, "classSegmentation")
        )
        self.depth: StereoPictureData = StereoPictureData(
            os.path.join(dir_path, "depth")
        )
        self.instanceSegmentation: StereoPictureData = StereoPictureData(
            os.path.join(dir_path, "instanceSegmentation")
        )
        self.rgb: StereoPictureData = StereoPictureData(os.path.join(dir_path, "rgb"))
        return


class ExtrinsicsData:
    def __init__(self, extrinsic_txt_path: str) -> None:
        self.extrinsic_txt_path: str = extrinsic_txt_path
        self.data: np.ndarray = self.__loadtxt__()
        return

    def __loadtxt__(self) -> np.ndarray:
        with open(self.extrinsic_txt_path, "r") as f:
            # names:frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
            _names = tuple(f.readline().split()[:-4])  # 去掉最后四个
            pass
        _formats = tuple(["int"] * 2 + ["float"] * (len(_names) - 2))
        data = np.loadtxt(
            self.extrinsic_txt_path,
            delimiter=" ",
            skiprows=1,
            usecols=range(len(_names)),
            dtype={"names": _names, "formats": _formats},
        )
        return data

    def __len__(self) -> int:
        return self.data["frame"][-1] + 1

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0:
            index = len(self) + index
            pass
        _extrinsics = sorted(
            self.data[self.data["frame"] == index], key=lambda x: x[1]
        )  # 使用cameraID排序
        return _extrinsics

    def get_R(self, index: int, cam_id: int = 0):
        # 获取旋转矩阵
        extrinsic = self[index][cam_id]
        R = np.array(
            [[extrinsic[f"r{i},{j}"] for j in range(1, 4)] for i in range(1, 4)],
            dtype=np.float32,
        )
        return R

    def get_t(self, index: int, cam_id: int = 0):
        # 获取平移向量
        extrinsic = self[index][cam_id]
        t = np.array([extrinsic[f"t{i}"] for i in range(1, 4)], dtype=np.float32)
        return t

    def convert_extrinsic_to_quaternion_and_t(self, index: int, cam_id: int = 0):
        # 将外参转为四元数
        # colmap中四元数也是表示的 the projection from world to the camera coordinate system
        R = self.get_R(index, cam_id)
        T = self.get_t(index, cam_id)
        quaternion = Rotation.from_matrix(R).as_quat()
        t = T
        return quaternion, t


class IntrinsicsData:
    def __init__(self, intrinsics_txt_path: str) -> None:
        self.intrinsics_txt_path: str = intrinsics_txt_path
        self.data: np.ndarray = self.__loadtxt__()
        return

    def __loadtxt__(self) -> np.ndarray:
        with open(self.intrinsics_txt_path, "r") as f:
            _names = tuple(f.readline().split())
            pass
        _formats = ("int", "int", "float", "float", "float", "int")
        assert len(_names) == len(_formats)
        data = np.loadtxt(
            self.intrinsics_txt_path,
            delimiter=" ",
            skiprows=1,
            usecols=range(len(_names)),
            dtype={"names": _names, "formats": _formats},
        )
        return data

    def __len__(self) -> int:
        return self.data["frame"][-1] + 1

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0:
            index = len(self) + index
            pass
        _intrinsics = sorted(
            self.data[self.data["frame"] == index], key=lambda x: x[1]
        )  # 使用cameraID排序
        return _intrinsics


class BboxData:
    def __init__(self, bbox_txt_path: str) -> None:
        self.bbox_txt_path: str = bbox_txt_path
        self.data: np.ndarray = self.__loadtxt__()
        return

    def __loadtxt__(self) -> np.ndarray:
        with open(self.bbox_txt_path, "r") as f:
            _names = tuple(f.readline().split())
            pass
        # _names: frame cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio isMoving
        _dtype = [
            ("frame", np.int32),
            ("cameraID", np.int32),
            ("trackID", np.int32),
            ("left", np.int32),
            ("right", np.int32),
            ("top", np.int32),
            ("bottom", np.int32),
            ("number_pixels", np.int32),
            ("truncation_ratio", np.float32),
            ("occupancy_ratio", np.float32),
            ("isMoving", np.bool_),
        ]
        assert len(_names) == len(_dtype)
        data = np.genfromtxt(
            self.bbox_txt_path, delimiter=" ", skip_header=1, dtype=_dtype
        )
        return data

    def __len__(self) -> int:
        # 这里的长度很可能比实际的长度短，因为有的帧没有bbox
        return self.data["frame"][-1] + 1

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0:
            index = len(self) + index
            pass
        _bboxes = self.data[
            self.data["frame"] == index
        ]  # 可能有很多bbox,就不排序了，使用的时候引用cam_id即可
        # 对于没有bbox的帧，会返回空的ndarray
        return _bboxes

    def is_car_static_all(self, trackID: int = 0, cam_id: int = 0):
        # 判断车辆是否静止
        car_data = self.data[
            np.logical_and(
                self.data["trackID"] == trackID, self.data["cameraID"] == cam_id
            )
        ]
        return np.all(car_data["isMoving"] == False)

    def is_car_moving_all(self, trackID: int = 0, cam_id: int = 0):
        # 判断车辆是否移动
        car_data = self.data[
            np.logical_and(
                self.data["trackID"] == trackID, self.data["cameraID"] == cam_id
            )
        ]
        return np.all(car_data["isMoving"] == True)

    def car_moving_ratio(self, trackID: int = 0, cam_id: int = 0):
        # 定义一个函数计算某一个car的isMoving为True的占比
        car_data = self.data[
            np.logical_and(
                self.data["trackID"] == trackID, self.data["cameraID"] == cam_id
            )
        ]
        return np.sum(car_data["isMoving"]) / len(car_data)

    def is_car_moving(self, trackID: int = 0, cam_id: int = 0, possibility=0.5):
        # 定义一个函数，如果车辆isMoving为True的占比超过possibility，则认为车辆在移动
        return self.car_moving_ratio(trackID, cam_id) > possibility

    def is_car_static(self, trackID: int = 0, cam_id: int = 0, possibility=0.5):
        # 定义一个函数，如果车辆isMoving为False的占比超过possibility，则认为车辆在静止
        return self.car_moving_ratio(trackID, cam_id) < possibility

    def get_all_static_car(self, cam_id: int = 0):
        # 获取所有静止的车辆
        car_ids = set(self.data["trackID"])
        return [
            _car_id for _car_id in car_ids if self.is_car_static_all(_car_id, cam_id)
        ]

    def get_all_moving_car(self, cam_id: int = 0):
        # 获取所有移动的车辆,这里只要是移动过的车辆，就算是移动的车辆
        car_ids = set(self.data["trackID"])
        return [
            _car_id
            for _car_id in car_ids
            if not self.is_car_static_all(_car_id, cam_id)
        ]


class PoseData:
    def __init__(self, pose_txt_path: str) -> None:
        self.pose_txt_path: str = pose_txt_path
        self.data: np.ndarray = self.__loadtxt__()
        return

    def __loadtxt__(self) -> np.ndarray:
        with open(self.pose_txt_path, "r") as f:
            _names = tuple(f.readline().split())
            pass
        # _names:frame cameraID trackID alpha width height length world_space_X world_space_Y world_space_Z rotation_world_space_y rotation_world_space_x rotation_world_space_z camera_space_X camera_space_Y camera_space_Z rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z
        # 第一行数据：25 0 0 1.69414 1.776562 1.421875 3.617188 53.18796 -106 79.85496 2.263542 0 0 -3.088518 1.056862 55.55215 1.638601 -0.01539355 0.03731001
        _dtype = [
            ("frame", np.int32),
            ("cameraID", np.int32),
            ("trackID", np.int32),
            ("alpha", np.float32),
            ("width", np.float32),
            ("height", np.float32),
            ("length", np.float32),
            ("world_space_X", np.float32),
            ("world_space_Y", np.float32),
            ("world_space_Z", np.float32),
            ("rotation_world_space_y", np.float32),
            ("rotation_world_space_x", np.float32),
            ("rotation_world_space_z", np.float32),
            ("camera_space_X", np.float32),
            ("camera_space_Y", np.float32),
            ("camera_space_Z", np.float32),
            ("rotation_camera_space_y", np.float32),
            ("rotation_camera_space_x", np.float32),
            ("rotation_camera_space_z", np.float32),
        ]
        assert len(_names) == len(_dtype)
        data = np.loadtxt(
            self.pose_txt_path,
            delimiter=" ",
            skiprows=1,
            usecols=range(len(_names)),
            dtype=_dtype,
        )
        return data

    def __len__(self) -> int:
        # 与bbox一样，这里的长度很可能比实际的长度短，因为有的帧没有pose
        return self.data["frame"][-1] + 1

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0:
            index = len(self) + index
            pass
        _pose = self.data[self.data["frame"] == index]
        # print(f"pose: {index} shape: {_pose.shape}, {_pose}")
        return _pose

    def get_w2car(self, index: int, trackID: int = 0, cam_id: int = 0):
        car2w = self.get_car2w(index, trackID, cam_id)
        if car2w is None:
            return None
        return np.linalg.inv(car2w)

    def get_car2w(self, index: int, trackID: int = 0, cam_id: int = 0):
        return self.get_car_to_other_matrix(index, "world", trackID, cam_id)

    def get_car_to_other_matrix(
        self, index: int, other_axis: str, trackID: int = 0, cam_id: int = 0
    ):
        # 返回车辆坐标系到其他坐标系的变换矩阵
        pose = self[index][
            np.logical_and(
                self[index]["trackID"] == trackID, self[index]["cameraID"] == cam_id
            )
        ]
        if len(pose) == 0:  # 不满足上面的筛选条件
            return None
        else:
            pose = pose[0]
        # 最后取0,原则上是一帧，同一个trackID，同一个cam_id只有一个这个结果
        rotation = Rotation.from_euler(
            "xyz",
            [
                pose[f"rotation_{other_axis}_space_x"],
                pose[f"rotation_{other_axis}_space_y"],
                pose[f"rotation_{other_axis}_space_z"],
            ],
            degrees=False,
        )
        t = np.array(
            [
                pose[f"{other_axis}_space_X"],
                pose[f"{other_axis}_space_Y"],
                pose[f"{other_axis}_space_Z"],
            ]
        )  # 这里shape是(3,)
        car2other = np.zeros((4, 4))
        car2other[:3, :3] = rotation.as_matrix()
        car2other[:3, 3] = t
        car2other[3, 3] = 1
        return car2other

    def get_car2cam(self, index: int, trackID: int = 0, cam_id: int = 0):
        return self.get_car_to_other_matrix(index, "camera", trackID, cam_id)

    def get_cam2car(self, index: int, trackID: int = 0, cam_id: int = 0):
        car2cam = self.get_car2cam(index, trackID, cam_id)
        if car2cam is None:
            return None
        return np.linalg.inv(car2cam)


class ColorData:
    def __init__(self, dir_path: str) -> None:
        self.dir_path: str = dir_path
        self.__loadtxt__()
        return

    def __len__(self) -> int:
        return len(self.data)

    def __loadtxt__(self):
        self.data: Dict[str:Tuple] = dict()
        data = np.loadtxt(self.dir_path, delimiter=" ", dtype=str, skiprows=1)

        for _category_info in data:
            _category, _r, _g, _b = _category_info
            _r, _g, _b = int(_r), int(_g), int(_b)
            self.data[_category] = (int(_r), int(_g), int(_b))

    def __getitem__(self, index: str) -> Tuple:
        return self.data[index]


class VkittiSceneData:
    def __init__(self, dir_path: str, scene_class: str = "clone") -> None:
        self.scene_class: str = scene_class
        self.dir_path: str = dir_path
        self.cam_num: int = 2
        self.frames: FramesData = FramesData(os.path.join(dir_path, "frames"))
        self.extrinsics: ExtrinsicsData = ExtrinsicsData(
            os.path.join(dir_path, "extrinsic.txt")
        )
        self.intrinsics: IntrinsicsData = IntrinsicsData(
            os.path.join(dir_path, "intrinsic.txt")
        )
        self.bbox: BboxData = BboxData(os.path.join(dir_path, "bbox.txt"))
        self.pose: PoseData = PoseData(os.path.join(dir_path, "pose.txt"))
        self.colors: ColorData = ColorData(os.path.join(dir_path, "colors.txt"))
        return

    def shwo_rgb_with_bbox(self, index: int, cam_id: int = 0) -> None:
        rgb = self.frames.rgb[cam_id][index]
        bboxes = self.bbox[index]
        if bboxes is not None:
            bboxes = bboxes[bboxes["cameraID"] == cam_id]
            for bbox in bboxes:
                cv2.rectangle(
                    rgb,
                    (bbox["left"], bbox["top"]),
                    (bbox["right"], bbox["bottom"]),
                    (0, 255, 0),
                    1,
                )
        cv2.imshow("rgb", rgb)
        # 按任意键或者点击关闭窗口
        while cv2.waitKey(1) & 0xFF != ord("q") and cv2.getWindowProperty(
            "rgb", cv2.WND_PROP_VISIBLE
        ):
            pass


if __name__ == "__main__":
    VkittiSceneData_path = "/home/lsin/Public/vkitti/Scene18/clone"
    vkitti_scene_data = VkittiSceneData(VkittiSceneData_path)
    # print(vkitti_scene_data.frames.rgb[0][0].shape)
    # print(vkitti_scene_data.extrinsics[0])
    # print(vkitti_scene_data.extrinsics[0][0])
    # print(vkitti_scene_data.extrinsics[0][1])
    # print(vkitti_scene_data.intrinsics[0])
    # print(len(vkitti_scene_data.intrinsics))

    # print(vkitti_scene_data.intrinsics[-1])

    # 测试bbox
    # print(vkitti_scene_data.bbox[0])
    # print(f"bbox length: {len(vkitti_scene_data.bbox)}")
    # print(vkitti_scene_data.bbox[0])
    # print(f"dtype: {vkitti_scene_data.bbox.data.dtype}")
    # print(f"shape of bbox: {vkitti_scene_data.bbox.data.shape}")

    # 测试show
    # vkitti_scene_data.shwo_rgb_with_bbox(80)

    # 测试pose
    # print(vkitti_scene_data.pose[0])
    # print(f"type of pose: {type(vkitti_scene_data.pose[0])}")
    # print(f"length of pose: {len(vkitti_scene_data.pose)}")

    # 测试四元数
    quaternion, t = vkitti_scene_data.extrinsics.convert_extrinsic_to_quaternion_and_t(
        0
    )
    print(quaternion)
    print(t)
    # 测试type,shape
    print(type(quaternion))
    print(quaternion.shape)
    print(type(t))
    print(t.shape)
