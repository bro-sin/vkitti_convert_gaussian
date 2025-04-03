"""
vkitti2.py
This module provides a convenient way to retrieve data from the Virtual KITTI 2 dataset.
"""

import os
from dataclasses import dataclass

# from typing import TYPE_CHECKING

from .frames import Frames


@dataclass
class VKitti2:
    """
    VKitti2 class for Virtual KITTI 2 dataset.
    """

    def __init__(
        self, scene_path: str, x_location: str = None, y_variation: str = None
    ) -> None:

        self._set_scene_path(
            scene_path=scene_path, x_location=x_location, y_variation=y_variation
        )
        self.frames: Frames = Frames(dir_path=self.scene_path)
        self.bbox = None
        self.colors = None
        self.extrinsic = None
        self.info = None
        self.intrinsic = None
        self.pose = None

    def _set_scene_path(
        self, scene_path: str, x_location: str, y_variation: str
    ) -> None:
        if x_location is None and y_variation is None:
            # get x_location and y_variation from scene_path
            _abs_scene_path = os.path.abspath(scene_path)
            y_variation = os.path.basename(_abs_scene_path)
            x_location = os.path.basename(os.path.dirname(_abs_scene_path))
        elif x_location is not None and y_variation is not None:
            scene_path = os.path.join(scene_path, x_location, y_variation)
        else:
            raise ValueError(
                f"Invalid arguments x_location={x_location},y_variation={y_variation}"
            )
        self.scene_path: str = scene_path
        self.x_location: str = x_location
        self.y_variation: str = y_variation

    def _check_scene_path_valid(self) ->None:
        assert os.path.exists(self.scene_path), f"self.scene_path={self.scene_path} does not exist"
        assert os.path.exists(os.path.join(self.scene_path, "frames")), f"frames folder does not exist in self.scene_path={self.scene_path}"
        assert os.path.exists(os.path.join(self.scene_path, "info.txt")), f"info.txt does not exist in self.scene_path={self.scene_path}"
        assert os.path.exists(os.path.join(self.scene_path, "bbox.txt")), f"bbox.txt does not exist in self.scene_path={self.scene_path}"
        assert os.path.exists(os.path.join(self.scene_path, "colors.txt")), f"colors.txt does not exist in self.scene_path={self.scene_path}"
        assert os.path.exists(os.path.join(self.scene_path, "extrinsic.txt")), f"extrinsic.txt does not exist in self.scene_path={self.scene_path}"
        assert os.path.exists(os.path.join(self.scene_path, "intrinsic.txt")), f"intrinsic.txt does not exist in self.scene_path={self.scene_path}"
        assert os.path.exists(os.path.join(self.scene_path, "pose.txt")), f"pose.txt does not exist in self.scene_path={self.scene_path}"