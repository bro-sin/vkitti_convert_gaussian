"""
frames.py
This module provides a convenient way to retrieve images 
from the Virtual KITTI 2 dataset.
"""

import os
from dataclasses import dataclass

from .class_segmentation import ClassSegmentation
from .depth import Depth
from .instance_segmentation import InstanceSegmentation
from .rgb import RGB


@dataclass
class Frames:
    """
    Class representing the frames of the Virtual KITTI 2 dataset.
    """

    dir_path: str

    def __post_init__(self) -> None:
        self.class_segmentation: ClassSegmentation = ClassSegmentation(
            dir_path=os.path.join(self.dir_path, "classSegmentation")
        )
        self.depth: Depth = Depth(dir_path=os.path.join(self.dir_path, "depth"))
        self.instance_segmentation: InstanceSegmentation = InstanceSegmentation(
            dir_path=os.path.join(self.dir_path, "instanceSegmentation")
        )
        self.rgb: RGB = RGB(dir_path=os.path.join(self.dir_path, "rgb"))
