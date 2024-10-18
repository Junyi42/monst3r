import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from torch._C import dtype, set_flush_denormal
import dust3r.utils.po_utils.basic
import dust3r.utils.po_utils.improc
from dust3r.utils.po_utils.misc import farthest_point_sample_py
from dust3r.utils.po_utils.geom import apply_4x4_py, apply_pix_T_cam_py
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
from functools import partial
import json

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


def convert_ndc_to_pixel_intrinsics(
    focal_length_ndc, principal_point_ndc, image_width, image_height, intrinsics_format='ndc_isotropic'
):
    f_x_ndc, f_y_ndc = focal_length_ndc
    c_x_ndc, c_y_ndc = principal_point_ndc

    # Compute half image size
    half_image_size_wh_orig = np.array([image_width, image_height]) / 2.0

    # Determine rescale factor based on intrinsics_format
    if intrinsics_format.lower() == "ndc_norm_image_bounds":
        rescale = half_image_size_wh_orig  # [image_width/2, image_height/2]
    elif intrinsics_format.lower() == "ndc_isotropic":
        rescale = np.min(half_image_size_wh_orig)  # scalar value
    else:
        raise ValueError(f"Unknown intrinsics format: {intrinsics_format}")

    # Convert focal length from NDC to pixel coordinates
    if intrinsics_format.lower() == "ndc_norm_image_bounds":
        focal_length_px = np.array([f_x_ndc, f_y_ndc]) * rescale
    elif intrinsics_format.lower() == "ndc_isotropic":
        focal_length_px = np.array([f_x_ndc, f_y_ndc]) * rescale

    # Convert principal point from NDC to pixel coordinates
    principal_point_px = half_image_size_wh_orig - np.array([c_x_ndc, c_y_ndc]) * rescale

    # Construct the intrinsics matrix in pixel coordinates
    K_pixel = np.array([
        [focal_length_px[0], 0,                principal_point_px[0]],
        [0,                 focal_length_px[1], principal_point_px[1]],
        [0,                 0,                 1]
    ])

    return K_pixel

def load_16big_png_depth(depth_png):
        with Image.open(depth_png) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
        return depth

class DynamicReplicaDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/dynamic_replica',
                 use_augs=False,
                 S=2,
                 strides=[1,2,3,4,5,6,7,8,9],
                 clip_step=2,
                 quick=False,
                 verbose=False,
                 dist_type=None,
                 clip_step_last_skip = 0,
                 *args, 
                 **kwargs
                 ):

        print('loading pointodyssey dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'pointodyssey'
        self.S = S # stride
        self.verbose = verbose

        self.use_augs = use_augs

        self.rgb_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location))

        anno_path = os.path.join(dataset_location, 'frame_annotations_train.json')
        with open(anno_path, 'r') as f:
            self.anno = json.load(f)

        #organize anno by 'sequence_name'
        anno_by_seq = {}
        for a in self.anno:
            seq_name = a['sequence_name']
            if seq_name not in anno_by_seq:
                anno_by_seq[seq_name] = []
            anno_by_seq[seq_name].append(a)

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                seq_name = seq.split('/')[-1]
                self.sequences.append(seq)

        self.sequences = anno_by_seq.keys()
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s ' % (len(self.sequences), dataset_location))


        if quick:
           self.sequences = self.sequences[1:2] 
        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            anno = anno_by_seq[seq]
            

            for stride in strides:
                for ii in range(0,len(anno)-self.S*max(stride,clip_step_last_skip)+1, clip_step):
                    full_idx = ii + np.arange(self.S)*stride
                    self.rgb_paths.append([os.path.join(dataset_location, anno[idx]['image']['path']) for idx in full_idx])
                    self.depth_paths.append([os.path.join(dataset_location, anno[idx]['depth']['path']) for idx in full_idx])
                    # check if all paths are valid, if not, skip
                    if not all([os.path.exists(p) for p in self.rgb_paths[-1]]) or not all([os.path.exists(p) for p in self.depth_paths[-1]]):
                        self.rgb_paths.pop()
                        self.depth_paths.pop()
                        continue
                    self.annotation_paths.append([anno[idx]['viewpoint'] for idx in full_idx])
                    self.full_idxs.append(full_idx)
                    self.sample_stride.append(stride)
                if self.verbose:
                    sys.stdout.write('.')
                    sys.stdout.flush()

        
        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        print('stride counts:', self.stride_counts)
        
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        print('collected %d clips of length %d in %s' % (
            len(self.rgb_paths), self.S, dataset_location,))

    def _resample_clips(self, strides, dist_type):

        # Get distribution of strides, and sample based on that
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        max_num_clips = self.stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [min(self.stride_counts[stride], int(dist[i]*max_num_clips)) for i, stride in enumerate(strides)]
        print('resampled_num_clips_each_stride:', num_clips_each_stride)
        resampled_idxs = []
        for i, stride in enumerate(strides):
            resampled_idxs += np.random.choice(self.stride_idxs[stride], num_clips_each_stride[i], replace=False).tolist()
        
        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        self.annotation_paths = [self.annotation_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)
    
    def _get_views(self, index, resolution, rng):

        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        full_idx = self.full_idxs[index]
        annotations = self.annotation_paths[index]
        focals = [np.array(annotations[i]['focal_length']).astype(np.float32) for i in range(2)]
        pp = [np.array(annotations[i]['principal_point']).astype(np.float32) for i in range(2)]
        intrinsics_format = [annotations[i]['intrinsics_format'] for i in range(2)]
        cams_T_world_R = [np.array(annotations[i]['R']).astype(np.float32) for i in range(2)]
        cams_T_world_t = [np.array(annotations[i]['T']).astype(np.float32) for i in range(2)]

        views = []
        for i in range(2):
            
            impath = rgb_paths[i]
            depthpath = depth_paths[i]

            # load camera params
            R = cams_T_world_R[i]
            t = cams_T_world_t[i]
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R.T
            camera_pose[:3,3] = -R.T @ t

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = load_16big_png_depth(depthpath)

            # load intrinsics
            intrinsics = convert_ndc_to_pixel_intrinsics(focals[i], pp[i], rgb_image.shape[1], rgb_image.shape[0],
                                                         intrinsics_format=intrinsics_format[i])
            intrinsics = intrinsics.astype(np.float32)


            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
            ))
        return views
        

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import gradio as gr
    import random

    use_augs = False
    S = 2
    strides = [1,2,3,4,5,6,7,8,9]
    clip_step = 2
    quick = False  # Set to True for quick testing

    def visualize_scene(idx):
        views = dataset[idx]
        assert len(views) == 2
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                        focal=views[view_idx]['camera_intrinsics'][0, 0],
                        color=(255, 0, 0),
                        image=colors,
                        cam_size=cam_size)
        os.makedirs('./tmp/replica', exist_ok=True)
        path = f"./tmp/replica/replica_scene_{idx}.glb"
        return viz.save_glb(path)

    dataset = DynamicReplicaDUSt3R(
        use_augs=use_augs,
        S=S,
        strides=strides,
        clip_step=clip_step,
        quick=quick,
        verbose=False,
        resolution=512, 
        aug_crop=16,
        dist_type='linear_1_9',
        aug_focal=1.0,
        z_far=80)

    idxs = np.arange(0, len(dataset)-1, (len(dataset)-1)//10)
    # idx = random.randint(0, len(dataset)-1)
    # idx = 0
    for idx in idxs:
        print(f"Visualizing scene {idx}...")
        visualize_scene(idx)
