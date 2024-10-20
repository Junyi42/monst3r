# %%
import glob
import os
import shutil
import numpy as np

dirs = glob.glob("../data/tum/*/")
dirs = sorted(dirs)
# extract frames
for dir in dirs:
    frames = glob.glob(dir + 'rgb/*.png')
    frames = sorted(frames)
    # sample 90 frames at the stride of 3
    frames = frames[::3][:90]
    # cut frames after 90
    new_dir = dir + 'rgb_90/'

    for frame in frames:
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(frame, new_dir)
        # print(f'cp {frame} {new_dir}')

    depth_frames = glob.glob(dir + 'depth/*.png')
    depth_frames = sorted(depth_frames)
    # sample 90 frames at the stride of 3
    depth_frames = depth_frames[::3][:90]
    # cut frames after 90
    new_dir = dir + 'depth_90/'

    for frame in depth_frames:
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(frame, new_dir)
        # print(f'cp {frame} {new_dir}')

for dir in dirs:
    gt_path = "groundtruth.txt"
    gt = np.loadtxt(dir + gt_path)
    gt_90 = gt[::3][:90]
    np.savetxt(dir + 'groundtruth_90.txt', gt_90)



