from vkitti2.frames.frames import Frames

f = Frames(dir_path="/home/lsin/Public/vkitti/Scene18/clone/frames")

frame = 80

rgb=f.rgb 
print(rgb.suffix)

rgb0 = f.rgb.camera_[1][frame]
rgb0.show()

classgt0 = f.class_segmentation.camera_[0][frame]
classgt0.show()


depth0 = f.depth.camera_[0][frame]
depth0.show()

instancegt0 = f.instance_segmentation.camera_[0][frame]
instancegt0.show()
