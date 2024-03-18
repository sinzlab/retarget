import bpy
import numpy as np
from os import listdir, path

def fbx2bvh(data_path, output_path, file):
    sourcepath = data_path+"/"+file
    bvh_path = output_path+"/"+file.split(".fbx")[0]+".bvh"

    bpy.ops.import_scene.fbx(filepath=sourcepath)

    frame_start = 9999
    frame_end = -9999
    action = bpy.data.actions[-1]
    if  action.frame_range[1] > frame_end:
      frame_end = action.frame_range[1]
    if action.frame_range[0] < frame_start:
      frame_start = action.frame_range[0]

    bpy.ops.export_anim.bvh(filepath=bvh_path,
                            frame_start=int(frame_start),
                            frame_end=int(frame_end), root_transform_only=True)
    bpy.data.actions.remove(bpy.data.actions[-1])
    print(data_path+"/"+file+" processed.")

if __name__ == '__main__':
    from pathlib import Path

    data_path = "./data/Mixamo/"
    fbx_path = data_path + "FBX/"
    bvh_path = data_path + "BVH/"

    Path(bvh_path).mkdir(exist_ok=True)

    directories = sorted([f for f in listdir(fbx_path) if not f.startswith(".")])
    for d in directories:
      Path(path.join(bvh_path,d)).mkdir(exist_ok=True)
      files = sorted([f for f in listdir(fbx_path+d) if f.endswith(".fbx")])
      for file in files:
        # print(path.join(fbx_path,d), path.join(bvh_path,d), file)
          fbx2bvh(path.join(fbx_path,d), path.join(bvh_path,d), file)
