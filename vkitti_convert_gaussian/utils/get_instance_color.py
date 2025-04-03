import cv2
import typing


# 输入一张图片路径，返回这张图片所包含的所有颜色，返回的是一个列表，列表中的每个元素是一个颜色，颜色是一个三元组
def get_instance_color(instance_image_path: str):
    # 读取图片
    instance_image = cv2.imread(instance_image_path, cv2.IMREAD_UNCHANGED)

    # 获取图片的高和宽
    height, width = instance_image.shape[:2]
    # 创建一个空列表，用来存储颜色
    bgr_dict = dict()
    # 遍历图片的每个像素
    for i in range(height):
        for j in range(width):
            # 存储像素的bgr值
            bgr = instance_image[i][j]
            # 将bgr值转换为元组
            bgr = tuple(bgr)
            # 将bgr值存储到列表中
            if bgr not in bgr_dict:
                bgr_dict[bgr] = 1
            else:
                bgr_dict[bgr] += 1
    return bgr_dict


if __name__ == "__main__":
    image_path = "/home/lsin/Public/vkitti/Scene18/clone/frames/instanceSegmentation/Camera_1/instancegt_00080.png"
    colors = get_instance_color(image_path)
    print(colors)
