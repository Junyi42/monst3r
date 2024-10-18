# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed WayMo
# dataset at https://github.com/waymo-research/waymo-open-dataset
# See datasets_preprocess/preprocess_waymo.py
# --------------------------------------------------------
import sys
sys.path.append('.')
import os
import os.path as osp
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class Waymo (BaseStereoViewDataset):
    """ Dataset of outdoor street scenes, 5 images each time
    """

    def __init__(self, *args, ROOT, pairs_npz_name='waymo_pairs_video.npz', **kwargs):
        self.ROOT = ROOT
        self.pairs_npz_name = pairs_npz_name
        super().__init__(*args, **kwargs)
        self._load_data()

    def _load_data(self):
        with np.load(osp.join(self.ROOT, self.pairs_npz_name)) as data:
            self.scenes = data['scenes']
            self.frames = data['frames']
            self.inv_frames = {frame: i for i, frame in enumerate(data['frames'])}
            self.pairs = data['pairs']  # (array of (scene_id, img1_id, img2_id)
            assert self.pairs[:, 0].max() == len(self.scenes) - 1
        print(f'Loaded {self.get_stats()}')

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.scenes)} scenes'

    def _get_views(self, pair_idx, resolution, rng):
        seq, img1, img2 = self.pairs[pair_idx]
        seq_path = osp.join(self.ROOT, self.scenes[seq])

        views = []

        for view_index in [img1, img2]:
            impath = self.frames[view_index]
            image = imread_cv2(osp.join(seq_path, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(seq_path, impath + ".exr"))
            camera_params = np.load(osp.join(seq_path, impath + ".npz"))

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.float32(camera_params['cam2world'])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, impath))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='Waymo',
                label=osp.relpath(seq_path, self.ROOT),
                instance=impath))

        return views


if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = Waymo(split='train', ROOT="data/waymo_processed", resolution=512, aug_crop=16)
    idxs = np.arange(0, len(dataset)-1, (len(dataset)-1)//10)
    for idx in idxs:
        views = dataset[idx]
        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        os.makedirs('./tmp/waymo', exist_ok=True)
        path = f"./tmp/waymo/waymo_scene_{idx}.glb"
        viz.save_glb(path)
