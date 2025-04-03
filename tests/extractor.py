from vkitti_convert_gaussian.utils.extractor import (
    extract_class_depth,
    extract_class_image,
    extract_instance_depth,
    extract_instance_image,
)
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

rgb_path = "/home/lsin/Public/vkitti/Scene18/clone/frames/rgb/Camera_0/rgb_00301.jpg"
class_seg_path = "/home/lsin/Public/vkitti/Scene18/clone/frames/classSegmentation/Camera_0/classgt_00301.png"
class_target_rgb = (250,100,255)

rgb_img = cv2.imread(rgb_path)
class_seg_img = np.asarray(Image.open(class_seg_path))
output_img = extract_class_image(rgb_img, class_seg_img, class_target_rgb, need=True)
output_img = cv2.cvtColor(output_img, cv2.COLOR_BGRA2RGBA)
plt.imshow(output_img)
plt.show()
