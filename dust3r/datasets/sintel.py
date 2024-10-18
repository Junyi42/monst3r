import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import glob
import PIL.Image
import torchvision.transforms as tvf

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2, crop_img
from dust3r.utils.misc import get_stride_distribution

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')
TAG_FLOAT = 202021.25
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ToTensor = tvf.ToTensor()

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

class SintelDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/sintel/training',
                 dset='clean',
                 use_augs=False,
                 S=2,
                 strides=[7],
                 clip_step=2,
                 quick=False,
                 verbose=False,
                 dist_type=None,
                 clip_step_last_skip = 0,
                 load_dynamic_mask=True,
                 *args, 
                 **kwargs
                 ):

        print('loading sintel dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'sintel'
        self.split = dset
        self.S = S # stride
        self.verbose = verbose
        self.load_dynamic_mask = load_dynamic_mask

        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.depth_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.dynamic_mask_paths = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')

        if quick:
           self.sequences = self.sequences[1:2] 
        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            rgb_path = seq
            depth_path = seq.replace(dset,'depth')
            caminfo_path = seq.replace(dset,'camdata_left')
            dynamic_mask_path = seq.replace(dset,'dynamic_label_perfect')
            
            for stride in strides:
                for ii in range(1,len(os.listdir(rgb_path))-self.S*max(stride,clip_step_last_skip)+1, clip_step):
                    full_idx = ii + np.arange(self.S)*stride
                    self.rgb_paths.append([os.path.join(rgb_path, 'frame_%04d.png' % idx) for idx in full_idx])
                    self.depth_paths.append([os.path.join(depth_path, 'frame_%04d.dpt' % idx) for idx in full_idx])
                    self.annotation_paths.append([os.path.join(caminfo_path, 'frame_%04d.cam' % idx) for idx in full_idx])
                    self.dynamic_mask_paths.append([os.path.join(dynamic_mask_path, 'frame_%04d.png' % idx) for idx in full_idx])
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
        self.annotation_paths = [self.annotation_paths[i] for i in resampled_idxs]
        self.dynamic_mask_paths = [self.dynamic_mask_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)
    
    def _get_views(self, index, resolution, rng):

        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        full_idx = self.full_idxs[index]
        annotations_paths = self.annotation_paths[index]
        dynamic_mask_paths = self.dynamic_mask_paths[index]

        views = []
        for i in range(2):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            dynamic_mask_path = dynamic_mask_paths[i]

            # load camera params
            intrinsics, extrinsics = cam_read(annotations_paths[i])
            intrinsics, extrinsics = np.array(intrinsics, dtype=np.float32), np.array(extrinsics, dtype=np.float32)
            R = extrinsics[:3,:3]
            t = extrinsics[:3,3]
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R.T
            camera_pose[:3,3] = -R.T @ t

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = depth_read(depthpath)

            # load dynamic mask
            if dynamic_mask_path is not None and os.path.exists(dynamic_mask_path):
                dynamic_mask = PIL.Image.open(dynamic_mask_path).convert('L')
                dynamic_mask = ToTensor(dynamic_mask).sum(0).numpy()
                _, dynamic_mask, _ = self._crop_resize_if_necessary(
                rgb_image, dynamic_mask, intrinsics, resolution, rng=rng, info=impath)
                dynamic_mask = dynamic_mask > 0.5
                assert not np.all(dynamic_mask), f"Dynamic mask is all True for {impath}"
            else:
                dynamic_mask = np.ones((resolution[1],resolution[0]), dtype=bool)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            
            if self.load_dynamic_mask:
                views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset=self.dataset_label,
                    label=rgb_paths[i].split('/')[-2],
                    instance=osp.split(rgb_paths[i])[1],
                    dynamic_mask=dynamic_mask,
                ))
            else:
                views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset=self.dataset_label,
                    label=rgb_paths[i].split('/')[-2],
                    instance=osp.split(rgb_paths[i])[1],
                ))
        return views
        

if __name__ == "__main__":

    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    use_augs = False
    S = 2
    strides = [1]
    clip_step = 1
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
        path = f"./tmp/sintel_scene_{idx}.glb"
        return viz.save_glb(path)

    dataset = SintelDUSt3R(
        use_augs=use_augs,
        S=S,
        strides=strides,
        clip_step=clip_step,
        quick=quick,
        verbose=False,
        resolution=(512,224), 
        seed = 777,
        clip_step_last_skip=0,
        aug_crop=16)

    idx = random.randint(0, len(dataset)-1)
    visualize_scene(idx)