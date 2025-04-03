"""
rgb.py
This module provides a convenient way to retrieve RGB images
from the Virtual KITTI 2 dataset.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Type, Literal
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .camera_ import ImageFile, CameraDataType


if TYPE_CHECKING:
    pass


class RGBImageFile(ImageFile):
    """
    RGB image file class for Virtual KITTI 2 dataset.
    """

    def __init__(self, file_path: str, resolution: float = 1) -> None:
        super().__init__(file_path=file_path, resolution=resolution)

    @property
    def image(self) -> np.ndarray:
        """
        Return the image.
        """
        with Image.open(self.file_path) as _img:
            _img_np = np.asarray(_img)
        return _img_np


@dataclass
class RGB(CameraDataType):
    """
    Class representing the data in the rgb folder of Vkitti2 datasets.
    """

    pattern: str = r"rgb_(\d+)\.jpg"
    image_file_class: Type[RGBImageFile] = RGBImageFile
    suffix: Literal["png", "jpg"] = "jpg"
