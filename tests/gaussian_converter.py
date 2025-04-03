from vkitti_convert_gaussian.vkitti import VkittiSceneData
from vkitti_convert_gaussian.gaussian_converter import GaussianConverter

scene_data = VkittiSceneData("/home/lsin/Public/vkitti/Scene18/clone")
gaussian_converter = GaussianConverter(scene_data)
# gaussian_converter.extract_instance_images()
# gaussian_converter.prepare_colmap_images_txt_for_only_car()
# gaussian_converter.convert_depth_to_pointclouds(bgr_target=(65, 137, 0),frames=range(60,81))

# bgr_dic=dict()
# for i in range(21):
#     print("Processing track", i, "...")
#     bgr=gaussian_converter.get_bgr_target(i,search_frames=10)
#     bgr_dic[i]=bgr

# print(bgr_dic)

import random

# gaussian_converter.generate_colmap_content_for_only_car(trackID=random.randint(a=0,b=21),focous_frame=175)
# gaussian_converter.generate_colmap_content_for_only_car(trackID=2, focous_frame=308)
# gaussian_converter.convert_depth_merge_point_clouds_for_car(cam_id=0,trackID=19,downsample=False)

category = "Terrain"

gaussian_converter.generate_colmap_content_for_one_class(
    category=category, cam_id=0, downsample=True
)
