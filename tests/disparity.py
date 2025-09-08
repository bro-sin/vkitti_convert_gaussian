# 将深度图转成视差图
from vkitti_convert_gaussian.vkitti import VkittiSceneData
from pathlib import Path
import numpy as np
import cv2

scene_path = Path("/home/lsin/Data/vkitti2/Scene18/clone/")
vkitti = VkittiSceneData(scene_path)

cam_index = 0  # 视差图是以左边摄像头为准的
pic_index = 100

# 拿到一张深度图
depth = vkitti.frames.depth[cam_index]
depth.cv_show(pic_index)#展示深度图

# 获取焦距
f = vkitti.intrinsics[pic_index][cam_index][2]

extrinsics = vkitti.extrinsics
left = -extrinsics.get_R(pic_index, 0).transpose() @ extrinsics.get_t(pic_index, 0)
right = -extrinsics.get_R(pic_index, 1).transpose() @ extrinsics.get_t(pic_index, 1)

# 获取基线长度，官网说是0.532725m=53.2725cm
# 也就是左右相机的距离，由于外参用的是米，因此这里结果是0.532724
B = np.linalg.norm(left - right)
#由于相机相对位置不变，因此基线长度B是一个常数
print(B)

depth_data = depth.get_pic(pic_index)

disparity_data = f * B / depth_data

cv2.imshow("Disparity", disparity_data)
cv2.waitKey(0)
cv2.destroyAllWindows()
