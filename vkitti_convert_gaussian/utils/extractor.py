# 输入RGB图片，实例分割图片，需要提取的实例在instance_segementation图片中的RGB值，输出提取的实例图片
import cv2
import numpy as np
from typing import Tuple


def extract_instance_image_using_path(
    rgb_path: str, instance_seg_path: str, bgr_target: Tuple, need: bool = True
):
    rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    instance_seg_img = cv2.imread(instance_seg_path, cv2.IMREAD_UNCHANGED)
    return extract_instance_image(rgb_img, instance_seg_img, bgr_target, need)


def extract_instance_image(
    rgb_img: np.ndarray,
    instance_seg_img: np.ndarray,
    bgr_target: Tuple,
    need: bool = True,
    white_bg: bool = False,
    use_original_bg: bool = False,
):
    # 把rgb_img扩展成4维,第四维度是alpha通道，默认是1
    rgba_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2BGRA)
    bg_rgb = (255, 255, 255) if white_bg else (0, 0, 0)
    bg_rgba = bg_rgb + (0,)
    if use_original_bg:
        bg_rgb = rgb_img.copy()
        # 扩展bg_rgb的维度，并且第四个维度设置为0.7
        alpha = np.ones((bg_rgb.shape[0], bg_rgb.shape[1], 1), dtype=np.uint8) * 10
        bg_rgba = np.concatenate((bg_rgb, alpha), axis=2)

    if need:
        x = rgba_img
        y = bg_rgba
    else:
        x = bg_rgba
        y = rgba_img
    # 把满足条件的点设置成x,不满足的设置成y
    output_img = np.where(
        np.all(instance_seg_img == np.asarray(bgr_target), axis=2, keepdims=True), x, y
    )
    return output_img


def extract_instance_depth(
    depth_img: np.ndarray,
    instance_seg_img: np.ndarray,
    bgr_target: Tuple,
    need: bool = True,
):
    depth_expanded_broadcasted = np.broadcast_to(
        np.expand_dims(depth_img, axis=-1), instance_seg_img.shape
    )
    if need:
        x = depth_expanded_broadcasted
        y = 65536
    else:
        x = 65536
        y = depth_expanded_broadcasted
    # 把满足条件的点设置成x,不满足的设置成y
    # 不需要的地方设置成65536，超出深度范围
    output_img = np.where(
        np.all(instance_seg_img == np.asarray(bgr_target), axis=2, keepdims=True), x, y
    )
    assert output_img.shape == instance_seg_img.shape
    output_img = output_img[:, :, 0]
    assert output_img.shape == depth_img.shape
    return output_img


def extract_class_image(
    rgb_img: np.ndarray,
    class_seg_img: np.ndarray,
    class_target_rgb: Tuple,
    need: bool = True,
    white_bg: bool = False,
):
    # 把rgb_img扩展成4维,第四维度是alpha通道，默认是1
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2BGRA)
    bg_rgb = (255, 255, 255) if white_bg else (0, 0, 0)
    bg_rgba = bg_rgb + (0,)

    if need:
        x = rgb_img
        y = bg_rgba
    else:
        x = bg_rgba
        y = rgb_img
    # 把满足条件的点设置成x,不满足的设置成y
    assert class_seg_img.shape[2] == 3
    output_img = np.where(
        np.all(class_seg_img == np.asarray(class_target_rgb), axis=2, keepdims=True),
        x,
        y,
    )
    return output_img


def extract_class_depth(
    depth_img: np.ndarray,
    class_seg_img: np.ndarray,
    class_target_rgb: Tuple,
    need: bool = True,
):
    depth_expanded_broadcasted = np.broadcast_to(
        np.expand_dims(depth_img, axis=-1), class_seg_img.shape
    )
    if need:
        x = depth_expanded_broadcasted
        y = 65536
    else:
        x = 65536
        y = depth_expanded_broadcasted
    # 把满足条件的点设置成x,不满足的设置成y
    # 不需要的地方设置成65536，超出深度范围
    output_img = np.where(
        np.all(class_seg_img == np.asarray(class_target_rgb), axis=2, keepdims=True),
        x,
        y,
    )
    # TODO:这里考虑用 rgb等于来限制 要取的depth_img,而不去扩展depth维度
    assert output_img.shape == class_seg_img.shape
    output_img = output_img[:, :, 0]
    assert output_img.shape == depth_img.shape
    return output_img


if __name__ == "__main__":
    instance_seg_path = "/home/lsin/Public/vkitti/Scene18/clone/frames/instanceSegmentation/Camera_1/instancegt_00080.png"
    rgb_path = (
        "/home/lsin/Public/vkitti/Scene18/clone/frames/rgb/Camera_1/rgb_00080.jpg"
    )
    depth_path = (
        "/home/lsin/Public/vkitti/Scene18/clone/frames/depth/Camera_1/depth_00080.png"
    )
    instance = cv2.imread(instance_seg_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # 输出shape
    # print(instance.shape)
    # print(depth.shape)
    # extract_instance_depth(depth, instance, (65, 137, 0))

    # 测试提取实例图片
    instance_img = extract_instance_image_using_path(
        rgb_path, instance_seg_path, (65, 137, 0)
    )
    # 保存
    cv2.imwrite("instance_img.png", instance_img)
