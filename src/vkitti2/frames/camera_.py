"""
camera_.py
This module provides a convenient way to retrieve images from the Virtual KITTI 2 dataset.
"""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, List, Type, Generic, TypeVar

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    pass


class ImageFile(ABC):
    """
    Single image file class for Virtual KITTI 2 dataset.
    """

    def __init__(self, file_path: str, resolution: float = 1) -> None:
        super().__init__()
        self.file_path: str = file_path
        self.resolution: float = resolution

    @property
    @abstractmethod
    def image(self) -> np.ndarray:
        """
        Return the image.
        """

    def get_image_file_pointer(self) -> Image:
        """
        This method is only for testing purposes.
        It is not recommended to use this method for production code.
        """
        return Image.open(self.file_path)

    def show(self) -> None:
        """
        Show the image.
        """
        with Image.open(self.file_path) as _img:
            _img.show()


SpecificImageFile = TypeVar("SpecificImageFile", bound=ImageFile)


class Camera(Generic[SpecificImageFile]):
    """
    Camera class for Virtual KITTI 2 dataset.
    The "SpecificImageFile" type parameter should be the subclass of ImageFile.
    """

    def __init__(
        self,
        dir_path: str,
        camera_index: int,
        pattern: str,
        image_file_class: Type[
            SpecificImageFile
        ],  # SpecificImageFile can be any of the classes like RGBImageFile, DepthImageFile, etc.
        suffix: Literal["png", "jpg"],
    ) -> None:
        self.dir_path: str = dir_path
        self.camera_index: int = camera_index
        self.suffix: Literal["png", "jpg"] = suffix
        self.pattern: str = pattern
        r"""
        The pattern of the image file path.
        For instance, if the subclass is for classSegmentation,
        the file name is like "classgt_%05d.png"
        this method should return r"classgt_(\d+)\.png"
        """
        self.image_file_class: Type[SpecificImageFile] = image_file_class
        self.image_file_paths: List[str] = [
            _image_file_path
            for _image_file_path in os.listdir(self.dir_path)
            if _image_file_path.endswith(suffix)
        ]
        # sort the image file paths using the frame index
        # extracted from the file name with the pattern
        self.image_file_paths.sort(
            key=lambda _image_file_path: int(
                re.match(pattern, _image_file_path).group(1)
            )
        )

    def __getitem__(self, frame_index: int) -> ImageFile:
        """
        Return the image file object created using SpecificImageFile class.
        """
        return self.image_file_class(
            file_path=os.path.join(self.dir_path, self.image_file_paths[frame_index])
        )

    def __len__(self) -> int:
        return len(self.image_file_paths)


@dataclass
class CameraDataType:
    """
    The data type of a general camera type.
    Must be extended to specify the image file class.
    For instance DepthDataType, ClassSegmentationDataType, etc.
    """

    dir_path: str
    pattern: str
    image_file_class: Type[ImageFile]
    suffix: Literal["png", "jpg"] = "png"
    camera_nums: int = 2
    camera_: List[Camera] = None

    def __post_init__(self):
        self.camera_ = [
            Camera(
                dir_path=os.path.join(self.dir_path, f"Camera_{_camera_index}"),
                camera_index=_camera_index,
                image_file_class=self.image_file_class,
                pattern=self.pattern,
                suffix=self.suffix,
            )
            for _camera_index in range(self.camera_nums)
        ]
