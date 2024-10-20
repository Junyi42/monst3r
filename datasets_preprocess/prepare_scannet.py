import glob
import os
import shutil
import numpy as np

seq_list = sorted(os.listdir("data/scannetv2"))
for seq in seq_list:
    img_pathes = sorted(glob.glob(f"data/scannetv2/{seq}/color/*.jpg"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    depth_pathes = sorted(glob.glob(f"data/scannetv2/{seq}/depth/*.png"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    pose_pathes = sorted(glob.glob(f"data/scannetv2/{seq}/pose/*.txt"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(f"{seq}: {len(img_pathes)} {len(depth_pathes)}")

    new_color_dir = f"data/scannetv2/{seq}/color_90"
    new_depth_dir = f"data/scannetv2/{seq}/depth_90"

    new_img_pathes = img_pathes[:90*3:3]
    new_depth_pathes = depth_pathes[:90*3:3]
    new_pose_pathes = pose_pathes[:90*3:3]

    os.makedirs(new_color_dir, exist_ok=True)
    os.makedirs(new_depth_dir, exist_ok=True)

    for i, (img_path, depth_path) in enumerate(zip(new_img_pathes, new_depth_pathes)):
        shutil.copy(img_path, f"{new_color_dir}/frame_{i:04d}.jpg")
        shutil.copy(depth_path, f"{new_depth_dir}/frame_{i:04d}.png")

    pose_new_path = f"data/scannetv2/{seq}/pose_90.txt"
    with open(pose_new_path, 'w') as f:
        for i, pose_path in enumerate(new_pose_pathes):
            with open(pose_path, 'r') as pose_file:
                pose = np.loadtxt(pose_file)
                pose = pose.reshape(-1)
                f.write(f"{' '.join(map(str, pose))}\n")
