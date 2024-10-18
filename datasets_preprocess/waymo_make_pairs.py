import glob
import os
from tqdm import tqdm
import cv2
import numpy as np

import numpy as np

file_path = "data/waymo/waymo_pairs.npz"

data = np.load(file_path)
# data.files # ['scenes', 'frames', 'pairs']

scenes, frames, pairs = data['scenes'], data['frames'], data['pairs']

new_scenes = glob.glob("data/waymo_processed/*.tfrecord/")
new_scenes_last = [scene.split("/")[-2] for scene in new_scenes]
img_lens = []
for path in tqdm(new_scenes):
    imgs = glob.glob(path + "/*.jpg")
    img_lens.append(len(imgs))

new_frames = list(frames)
new_pairs = []
strides = [2,3,4,5,6,7,8,9,10]
step = 1
for path in tqdm(new_scenes):
    imgs_track1 = glob.glob(path + "/*_1.jpg")
    imgs_track1.sort()
    imgs_track2 = glob.glob(path + "/*_2.jpg")
    imgs_track2.sort()
    imgs_track3 = glob.glob(path + "/*_3.jpg")
    imgs_track3.sort()
    imgs_track4 = glob.glob(path + "/*_4.jpg")
    imgs_track4.sort()
    imgs_track5 = glob.glob(path + "/*_5.jpg")
    imgs_track5.sort()
    for stride in strides:
        for i in range(0, len(imgs_track1)-stride, step):
            if os.path.exists(imgs_track1[i+stride]) and os.path.exists(imgs_track1[i]):
                new_pairs.append([new_scenes_last.index(path.split("/")[-2]), new_frames.index(imgs_track1[i].split('/')[-1].replace('.jpg','')), new_frames.index(imgs_track1[i+stride].split('/')[-1].replace('.jpg',''))])
        for i in range(0, len(imgs_track2)-stride, step):
            if os.path.exists(imgs_track2[i+stride]) and os.path.exists(imgs_track2[i]):
                new_pairs.append([new_scenes_last.index(path.split("/")[-2]), new_frames.index(imgs_track2[i].split('/')[-1].replace('.jpg','')), new_frames.index(imgs_track2[i+stride].split('/')[-1].replace('.jpg',''))])
        for i in range(0, len(imgs_track3)-stride, step):
            if os.path.exists(imgs_track3[i+stride]) and os.path.exists(imgs_track3[i]):
                new_pairs.append([new_scenes_last.index(path.split("/")[-2]), new_frames.index(imgs_track3[i].split('/')[-1].replace('.jpg','')), new_frames.index(imgs_track3[i+stride].split('/')[-1].replace('.jpg',''))])
        for i in range(0, len(imgs_track4)-stride, step):
            if os.path.exists(imgs_track4[i+stride]) and os.path.exists(imgs_track4[i]):
                new_pairs.append([new_scenes_last.index(path.split("/")[-2]), new_frames.index(imgs_track4[i].split('/')[-1].replace('.jpg','')), new_frames.index(imgs_track4[i+stride].split('/')[-1].replace('.jpg',''))])
        for i in range(0, len(imgs_track5)-stride, step):
            if os.path.exists(imgs_track5[i+stride]) and os.path.exists(imgs_track5[i]):
                new_pairs.append([new_scenes_last.index(path.split("/")[-2]), new_frames.index(imgs_track5[i].split('/')[-1].replace('.jpg','')), new_frames.index(imgs_track5[i+stride].split('/')[-1].replace('.jpg',''))])

print(len(new_pairs), "pairs")
save_path = "data/waymo_processed/waymo_pairs_video2_10.npz"
np.savez(save_path, scenes=np.array(new_scenes_last), frames=np.array(new_frames), pairs=np.array(new_pairs))