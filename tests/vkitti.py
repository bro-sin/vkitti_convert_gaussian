from vkitti_convert_gaussian.vkitti import VkittiSceneData
import numpy as np

scene_path = "/home/lsin/Models/lsy_vkitti_combine_train_render/Scene20/clone/"
vkitti = VkittiSceneData(scene_path)
# x,y,z= vkitti.pose[26][["world_space_X", "world_space_Y", "world_space_Z"]][0]
# P_w=np.array([x,y,z,1]).reshape(4,1)
# # print(x,y,z)
# w2car=vkitti.pose.get_w2car(26)
# print(w2car)

# P_car=w2car.dot(P_w)
# print(P_car)
# assert all(P_car==np.array([0,0,0,1]).reshape(4,1))


# 测试car2w*w2cam=car2cam
# i=26
# cam_id=0
# car2w=vkitti.pose.get_car2w(index=i, cam_id=cam_id)
# print("car2w")
# print(car2w)
# _extrinsic_data = vkitti.extrinsics
# w2cam = np.zeros((4, 4))
# w2cam[:3, :3] = _extrinsic_data.get_R(index=i, cam_id=cam_id)
# w2cam[:3, 3] = _extrinsic_data.get_t(index=i, cam_id=cam_id)
# w2cam[3, 3] = 1
# print("w2cam")
# print(w2cam)

# car2cam=vkitti.pose.get_car2cam(i,cam_id)
# print("car2cam")
# print(car2cam)

# _diff=np.dot(w2cam,car2w)-car2cam
# print("diff")
# print(_diff)
# print(np.linalg.norm(_diff))

# rgb=vkitti.colors["Road"]
# print(rgb)
# print(len(vkitti.colors))

bbox = vkitti.bbox
car_ids = set(bbox.data["trackID"])
from matplotlib import pyplot as plt

car_moving_ratio = []
for i in car_ids:
    # static = bbox.is_car_static(i)
    # static_all = bbox.is_car_static_all(i)
    # moving = bbox.is_car_moving(i)
    # moving_all = bbox.is_car_moving_all(i)
    # if static and moving:
    #     print(
    #         f"car {i} is static: {static}, is static all: {static_all}, is moving: {moving}, is moving all: {moving_all}"
    #     )

    # if static == static_all:
    #     continue
    # print(f"car {i} is static: {static}, is static all: {static_all}")
    moving_ratio = bbox.car_moving_ratio(i)
    print(f"car {i} moving ratio: {moving_ratio}")
    car_moving_ratio.append(moving_ratio)

plt.hist(car_moving_ratio, bins=20)
plt.show()

# 用点来表示
plt.plot(car_moving_ratio, "ro")
plt.show()

# 输出moving_ratio为0,1和介于0,1之间的车辆,以及这几类车辆的数量
all_moving_car = [i for i in car_ids if bbox.car_moving_ratio(i) == 1]
all_static_car = [i for i in car_ids if bbox.car_moving_ratio(i) == 0]
other_car = [i for i in car_ids if 0 < bbox.car_moving_ratio(i) < 1]
print(f"There are {len(all_moving_car)} moving cars, respectively {all_moving_car}")
print(f"There are {len(all_static_car)} static cars, respectively {all_static_car}")
print(f"There are {len(other_car)} other cars, respectively {other_car}")
