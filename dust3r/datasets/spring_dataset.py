import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import glob

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution
import h5py

SPRING_BASELINE = 0.065
np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

def get_depth(disp1, intrinsics, baseline=SPRING_BASELINE):
    """
    get depth from reference frame disparity and camera intrinsics
    """
    return intrinsics[0] * baseline / disp1


def readDsp5Disp(filename):
    with h5py.File(filename, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
        return f["disparity"][()]


class SpringDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/spring',
                 dset='train',
                 use_augs=False,
                 S=2,
                 strides=[8],
                 clip_step=2,
                 quick=False,
                 verbose=False,
                 dist_type=None,
                 remove_seq_list=[],
                 *args, 
                 **kwargs
                 ):

        print('loading Spring dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'Spring'
        self.split = dset
        self.S = S # number of frames
        self.verbose = verbose

        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.depth_paths = []
        self.annotations = []
        self.intrinsics = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                seq_name = seq.split('/')[-2]
                print(f"Processing sequence {seq_name}")
                # remove_seq_list = ['0008', '0041', '0043']
                if seq_name in remove_seq_list:
                    print(f"Skipping sequence {seq_name}")
                    continue
                self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        if quick:
           self.sequences = self.sequences[1:2] 
        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            rgb_path = os.path.join(seq, 'frame_left')
            depth_path = os.path.join(seq, 'disp1_left')
            caminfo_path = os.path.join(seq, 'cam_data/extrinsics.txt')
            caminfo = np.loadtxt(caminfo_path)
            intrinsics_path = os.path.join(seq, 'cam_data/intrinsics.txt')
            intrinsics = np.loadtxt(intrinsics_path)
            
            for stride in strides:
                for ii in range(1,len(os.listdir(rgb_path))-self.S*stride+2, clip_step):
                    full_idx = ii + np.arange(self.S)*stride
                    self.rgb_paths.append([os.path.join(rgb_path, 'frame_left_%04d.png' % idx) for idx in full_idx])
                    self.depth_paths.append([os.path.join(depth_path, 'disp1_left_%04d.dsp5' % idx) for idx in full_idx])
                    self.annotations.append(caminfo[full_idx-1])
                    self.intrinsics.append(intrinsics[full_idx-1])
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

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))
    
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
        self.annotations = [self.annotations[i] for i in resampled_idxs]
        self.intrinsics = [self.intrinsics[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)
    
    def _get_views(self, index, resolution, rng):

        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        annotations = self.annotations[index]
        intrinsics = self.intrinsics[index]

        views = []
        for i in range(2):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]

            # load camera params
            extrinsic = np.reshape(annotations[i], (4,4)).astype(np.float32)
            camera_pose = np.linalg.inv(extrinsic)
            intrinsic_matrix = np.zeros((3, 3), dtype=np.float32)
            intrinsic_matrix[0, 0] = intrinsics[i][0]
            intrinsic_matrix[1, 1] = intrinsics[i][1]
            intrinsic_matrix[0, 2] = intrinsics[i][2]
            intrinsic_matrix[1, 2] = intrinsics[i][3]

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = get_depth(readDsp5Disp(depthpath), intrinsics[i]).astype(np.float32)
            depthmap = depthmap[::2, ::2]
            depthmap = np.where(np.isnan(depthmap), -1, depthmap)
            depthmap = np.where(np.isinf(depthmap), -1, depthmap)

            rgb_image, depthmap, intrinsic_matrix = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsic_matrix, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsic_matrix,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
            ))
        return views

if __name__ == "__main__":

    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    use_augs = False
    S = 2
    strides=[2,4,6,8,10,12,14,16,18]
    clip_step = 2
    quick = False  # Set to True for quick testing
    dist_type='linear_9_1'


    def visualize_scene(idx):
        views = dataset[idx]
        assert len(views) == 2
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 1)
        label = views[0]['label']
        instance = views[0]['instance']
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
        os.makedirs("./tmp/spring_scene", exist_ok=True)
        path = f"./tmp/spring_scene/spring_scene_{label}_{views[0]['instance']}_{views[1]['instance']}.glb"
        return viz.save_glb(path)

    dataset = SpringDUSt3R(
        use_augs=use_augs,
        S=S,
        strides=strides,
        clip_step=clip_step,
        quick=quick,
        verbose=False,
        resolution=(512,288), 
        aug_crop=16,
        dist_type=dist_type,
        z_far=80)

    idxs = np.arange(0, len(dataset)-1, (len(dataset)-1)//10)
    for idx in idxs:
        print(f"Visualizing scene {idx}...")
        visualize_scene(idx)