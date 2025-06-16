import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import contextlib

from dust3r.cloud_opt.base_opt import BasePCOptimizer, edge_str
from dust3r.cloud_opt.pair_viewer import PairViewer
from dust3r.utils.geometry import xy_grid, geotrf, depthmap_to_pts3d
from dust3r.utils.device import to_cpu, to_numpy
from dust3r.utils.goem_opt import DepthBasedWarping, OccMask, WarpImage, depth_regularization_si_weighted, tum_to_pose_matrix
from third_party.raft import load_RAFT
from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

def smooth_L1_loss_fn(estimate, gt, mask, beta=1.0, per_pixel_thre=50.):
    loss_raw_shape = F.smooth_l1_loss(estimate*mask, gt*mask, beta=beta, reduction='none')
    if per_pixel_thre > 0:
        per_pixel_mask = (loss_raw_shape < per_pixel_thre) * mask
    else:
        per_pixel_mask = mask
    return torch.sum(loss_raw_shape * per_pixel_mask) / torch.sum(per_pixel_mask)

def mse_loss_fn(estimate, gt, mask):
    v = torch.sum((estimate*mask-gt*mask)**2) / torch.sum(mask)
    return v  # , v.item()

class PointCloudOptimizer(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, shared_focal=False, flow_loss_fn='smooth_l1', flow_loss_weight=0.0, 
                 depth_regularize_weight=0.0, num_total_iter=300, temporal_smoothing_weight=0, translation_weight=0.1, flow_loss_start_epoch=0.15, flow_loss_thre=50,
                 sintel_ckpt=False, use_self_mask=False, pxl_thre=50, sam2_mask_refine=True, motion_mask_thre=0.35, batchify=True,
                 window_wise=False, window_size=100, window_overlap_ratio=0.5, prev_video_results=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break
        self.num_total_iter = num_total_iter
        self.temporal_smoothing_weight = temporal_smoothing_weight
        self.translation_weight = translation_weight
        self.flow_loss_flag = False
        self.flow_loss_start_epoch = flow_loss_start_epoch
        self.flow_loss_thre = flow_loss_thre
        self.optimize_pp = optimize_pp
        self.pxl_thre = pxl_thre
        self.motion_mask_thre = motion_mask_thre
        self.batchify = batchify
        self.window_wise = window_wise

        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  # log(depth)
        self.im_poses = nn.ParameterList(self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs))  # camera poses
        self.shared_focal = shared_focal
        if self.shared_focal:
            self.im_focals = nn.ParameterList(torch.FloatTensor(
                [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes[:1])  # camera intrinsics
        else:
            self.im_focals = nn.ParameterList(torch.FloatTensor(
                [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes)  # camera intrinsics
        self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        self.im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(self.im_areas)

        # adding thing to global optimization
        if self.batchify:
            self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area) #(num_imgs, H*W)
            self.im_poses = ParameterStack(self.im_poses, is_param=True)
            self.im_focals = ParameterStack(self.im_focals, is_param=True)
            self.im_pp = ParameterStack(self.im_pp, is_param=True)
            self.register_buffer('_pp', torch.tensor([(w/2, h/2) for h, w in self.imshapes]))
            self.register_buffer('_grid', ParameterStack(
                [xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area))
            # pre-compute pixel weights
            self.register_buffer('_weight_i', ParameterStack(
                [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area))
            self.register_buffer('_weight_j', ParameterStack(
                [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area))
            # precompute aa
            self.register_buffer('_stacked_pred_i', ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
            self.register_buffer('_stacked_pred_j', ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
        # adding thing to window wise optimization
        elif self.window_wise:
            self.window_size = window_size
            self.window_overlap_ratio = window_overlap_ratio
            self.overlap_size = int(window_size * window_overlap_ratio)
            self.window_stride = window_size - self.overlap_size
            self.prev_video_results = prev_video_results
            # processing previous video results
            if self.prev_video_results is not None:
                self.n_prev_frames = len(prev_video_results['depths'])
                self._validate_prev_results()
                self._init_global_cache()
            else:
                self.n_prev_frames = 0
            self.window_starts = list(range(
                0,
                self.n_imgs - window_size + 1,
                self.window_stride
            ))
            if (self.n_imgs - window_size) % self.window_stride != 0:
                self.window_starts.append(self.n_imgs - window_size)
            
            self.num_windows = len(self.window_starts)
            # pre-compute pixel weights
            self.register_buffer('_weight_i', ParameterStack(
                [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area))
            self.register_buffer('_weight_j', ParameterStack(
                [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area))
            # precompute aa
            self.register_buffer('_stacked_pred_i', ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
            self.register_buffer('_stacked_pred_j', ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
            
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([self.im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([self.im_areas[j] for i, j in self.edges])

        self.depth_wrapper = DepthBasedWarping()
        self.backward_warper = WarpImage()
        self.depth_regularizer = depth_regularization_si_weighted
        if flow_loss_fn == 'smooth_l1':
            self.flow_loss_fn = smooth_L1_loss_fn
        elif flow_loss_fn == 'mse':
            self.low_loss_fn = mse_loss_fn

        self.flow_loss_weight = flow_loss_weight
        self.depth_regularize_weight = depth_regularize_weight
        if self.flow_loss_weight > 0:
            self.flow_ij, self.flow_ji, self.flow_valid_mask_i, self.flow_valid_mask_j = self.get_flow(sintel_ckpt) # (num_pairs, 2, H, W)
            if use_self_mask: self.get_motion_mask_from_pairs(*args)
            # turn off the gradient for the flow
            self.flow_ij.requires_grad_(False)
            self.flow_ji.requires_grad_(False)
            self.flow_valid_mask_i.requires_grad_(False)
            self.flow_valid_mask_j.requires_grad_(False)
            if sam2_mask_refine: 
                with torch.no_grad():
                    self.refine_motion_mask_w_sam2()
            else:
                self.sam2_dynamic_masks = None

    def _validate_prev_results(self):
        """Verify the format of the previous video data"""
        required_keys = ['depths', 'intrinsics', 'poses']
        assert all(k in self.prev_video_results for k in required_keys), \
            f"prev_video_results is missing required fields: {required_keys}"
        
        assert self.prev_video_results['depths'][0].shape == self.im_depthmaps[0].shape, \
            f"the depths of previous video do not match: {self.prev_video_results['depths'][0].shape} vs {self.im_depthmaps[0].shape}"
        
        assert self.overlap_size <= len(self.prev_video_results['depths']), \
            f"the frame number of previous video is insufficient, at least {self.overlap_size} frames are required"

    def _init_global_cache(self):
        """Initialize the global cache"""
        self.preset_intrinsics(self.prev_video_results['intrinsics'])
        self.preset_pose(self.prev_video_results['poses'])
        self.preset_depthmap(self.prev_video_results['depths'])
        for i in range(self.n_prev_frames):
            if not self.shared_focal:
                self.im_focals[i].requires_grad_(False)
            self.im_poses[i].requires_grad_(False)
            self.im_depthmaps[i].requires_grad_(False)
        if self.shared_focal:
            self.im_focals[0].requires_grad_(False)
            
    def _get_window_bounds(self, window_idx):
        """Get the window interval and the optimized interval"""
        start = self.window_starts[window_idx]
        end = start + self.window_size
        opt_start = start + int(self.window_size * self.window_overlap_ratio)
        return start, opt_start, end

    def load_window_params(self, window_idx):
        """Load window parameters dynamically, including frozen parameters"""
        # get the window bounds
        win_start, opt_start, win_end = self._get_window_bounds(window_idx)
        # process last window
        if window_idx == self.num_windows-1:
            opt_start = self._get_window_bounds(window_idx-1)[2]
        # Load optimization and frozen parameters from cache
        if window_idx == 0 and self.prev_video_results is None:
            self.opt_im_depthmaps = ParameterStack(self.im_depthmaps[win_start:win_end], is_param=True, fill=self.max_area).requires_grad_(True)
            self.opt_im_poses = ParameterStack(self.im_poses[win_start:win_end], is_param=True).requires_grad_(True)
            self.opt_im_focals = ParameterStack(self.im_focals[win_start:win_end], is_param=True).requires_grad_(True)
        else:
            self.frozen_im_depthmaps = ParameterStack(self.im_depthmaps[win_start:opt_start], is_param=True, fill=self.max_area).requires_grad_(False)
            self.opt_im_depthmaps = ParameterStack(self.im_depthmaps[opt_start:win_end], is_param=True, fill=self.max_area).requires_grad_(True)
            self.frozen_im_poses = ParameterStack(self.im_poses[win_start:opt_start], is_param=True).requires_grad_(False)
            self.opt_im_poses = ParameterStack(self.im_poses[opt_start:win_end], is_param=True).requires_grad_(True)
            self.opt_im_focals = ParameterStack(self.im_focals, is_param=True).requires_grad_(False)
            
        self.win_init_depthmap = self.init_depthmap[win_start:win_end]
        self.win_dynamic_masks = self.dynamic_masks[win_start:win_end]
        self.win_imshapes = self.imshapes[win_start:win_end]
        self.win_im_pp = ParameterStack(self.im_pp[win_start:win_end], is_param=True)
        self.register_buffer('_win_pp', torch.tensor([(w/2, h/2) for h, w in self.win_imshapes], device=self.device))
        self.register_buffer('_win_grid', ParameterStack(
            [xy_grid(W, H, device=self.device) for H, W in self.win_imshapes], fill=self.max_area))
        
        valid_i = (self._ei >= win_start) & (self._ei < win_end)
        valid_j = (self._ej >= win_start) & (self._ej < win_end)
        valid_edges = valid_i & valid_j

        self.opt_pw_poses = nn.Parameter(self.pw_poses[valid_edges])
        self.win_pw_adaptors = self.pw_adaptors[valid_edges]
        self._win_weight_i = self._weight_i[valid_edges]
        self._win_weight_j = self._weight_j[valid_edges]
        self._win_stacked_pred_i = self._stacked_pred_i[valid_edges]
        self._win_stacked_pred_j = self._stacked_pred_j[valid_edges]
        self._win_ei = self._ei[valid_edges] - win_start
        self._win_ej = self._ej[valid_edges] - win_start
        if self.flow_loss_weight > 0:
            self.win_flow_ij = self.flow_ij[valid_edges]
            self.win_flow_ji = self.flow_ji[valid_edges]
        self.win_total_area_i = sum([self.im_areas[i] for i, j in torch.tensor(self.edges, device=self.device)[valid_edges]])
        self.win_total_area_j = sum([self.im_areas[j] for i, j in torch.tensor(self.edges, device=self.device)[valid_edges]])
        self.window_idx = window_idx
        if self.verbose:
            print(f"Window {window_idx}: Total [{win_start}-{win_end}), Optimizing [{opt_start}-{win_end})")

    def save_window_params(self, window_idx):
        # get the window bounds
        win_start, opt_start, win_end = self._get_window_bounds(window_idx)
        # process last window
        if window_idx == self.num_windows-1:
            opt_start = self._get_window_bounds(window_idx-1)[2]

        if window_idx == 0 and self.prev_video_results is None:
            for idx, i in enumerate(range(win_start, win_end)):
                depth_param = self.im_depthmaps[i]
                depth_param.data[:] = self.opt_im_depthmaps[idx].view(self.imshapes[i][0], self.imshapes[i][1])
                pose_param = self.im_poses[i]
                pose_param.data[:] = self.opt_im_poses[idx]
                
                if not self.shared_focal:
                    focal_param = self.im_focals[i]
                    focal_param.data[:] = self.opt_im_focals[idx]

            if self.shared_focal:
                focal_param = self.im_focals[0]
                focal_param.data[:] = self.opt_im_focals[0]
        else:
            for idx, i in enumerate(range(opt_start, win_end)):
                depth_param = self.im_depthmaps[i]
                depth_param.data[:] = self.opt_im_depthmaps[idx].view(self.imshapes[i][0], self.imshapes[i][1])
                pose_param = self.im_poses[i]
                pose_param.data[:] = self.opt_im_poses[idx]
            
    def get_flow(self, sintel_ckpt=False): #TODO: test with gt flow
        print('precomputing flow...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        get_valid_flow_mask = OccMask(th=3.0)
        pair_imgs = [np.stack(self.imgs)[self._ei], np.stack(self.imgs)[self._ej]]

        flow_net = load_RAFT() if sintel_ckpt else load_RAFT("third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
        flow_net = flow_net.to(device)
        flow_net.eval()

        with torch.no_grad():
            chunk_size = 12
            flow_ij = []
            flow_ji = []
            num_pairs = len(pair_imgs[0])
            for i in tqdm(range(0, num_pairs, chunk_size)):
                end_idx = min(i + chunk_size, num_pairs)
                imgs_ij = [torch.tensor(pair_imgs[0][i:end_idx]).float().to(device),
                        torch.tensor(pair_imgs[1][i:end_idx]).float().to(device)]
                flow_ij.append(flow_net(imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                        imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                        iters=20, test_mode=True)[1])
                flow_ji.append(flow_net(imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                        imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                        iters=20, test_mode=True)[1])

            flow_ij = torch.cat(flow_ij, dim=0)
            flow_ji = torch.cat(flow_ji, dim=0)
            valid_mask_i = get_valid_flow_mask(flow_ij, flow_ji)
            valid_mask_j = get_valid_flow_mask(flow_ji, flow_ij)
        print('flow precomputed')
        # delete the flow net
        if flow_net is not None: del flow_net
        return flow_ij, flow_ji, valid_mask_i, valid_mask_j

    def get_motion_mask_from_pairs(self, view1, view2, pred1, pred2):
        assert self.is_symmetrized, 'only support symmetric case'
        symmetry_pairs_idx = [(i, i+len(self.edges)//2) for i in range(len(self.edges)//2)]
        intrinsics_i = []
        intrinsics_j = []
        R_i = []
        R_j = []
        T_i = []
        T_j = []
        depth_maps_i = []
        depth_maps_j = []
        for i, j in tqdm(symmetry_pairs_idx):
            new_view1 = {}
            new_view2 = {}
            for key in view1.keys():
                if isinstance(view1[key], list):
                    new_view1[key] = [view1[key][i], view1[key][j]]
                    new_view2[key] = [view2[key][i], view2[key][j]]
                elif isinstance(view1[key], torch.Tensor):
                    new_view1[key] = torch.stack([view1[key][i], view1[key][j]])
                    new_view2[key] = torch.stack([view2[key][i], view2[key][j]])
            new_view1['idx'] = [0, 1]
            new_view2['idx'] = [1, 0]
            new_pred1 = {}
            new_pred2 = {}
            for key in pred1.keys():
                if isinstance(pred1[key], list):
                    new_pred1[key] = [pred1[key][i], pred1[key][j]]
                elif isinstance(pred1[key], torch.Tensor):
                    new_pred1[key] = torch.stack([pred1[key][i], pred1[key][j]])
            for key in pred2.keys():
                if isinstance(pred2[key], list):
                    new_pred2[key] = [pred2[key][i], pred2[key][j]]
                elif isinstance(pred2[key], torch.Tensor):
                    new_pred2[key] = torch.stack([pred2[key][i], pred2[key][j]])
            pair_viewer = PairViewer(new_view1, new_view2, new_pred1, new_pred2, verbose=False)
            intrinsics_i.append(pair_viewer.get_intrinsics()[0])
            intrinsics_j.append(pair_viewer.get_intrinsics()[1])
            R_i.append(pair_viewer.get_im_poses()[0][:3, :3])
            R_j.append(pair_viewer.get_im_poses()[1][:3, :3])
            T_i.append(pair_viewer.get_im_poses()[0][:3, 3:])
            T_j.append(pair_viewer.get_im_poses()[1][:3, 3:])
            depth_maps_i.append(pair_viewer.get_depthmaps()[0])
            depth_maps_j.append(pair_viewer.get_depthmaps()[1])
        
        self.intrinsics_i = torch.stack(intrinsics_i).to(self.flow_ij.device)
        self.intrinsics_j = torch.stack(intrinsics_j).to(self.flow_ij.device)
        self.R_i = torch.stack(R_i).to(self.flow_ij.device)
        self.R_j = torch.stack(R_j).to(self.flow_ij.device)
        self.T_i = torch.stack(T_i).to(self.flow_ij.device)
        self.T_j = torch.stack(T_j).to(self.flow_ij.device)
        self.depth_maps_i = torch.stack(depth_maps_i).unsqueeze(1).to(self.flow_ij.device)
        self.depth_maps_j = torch.stack(depth_maps_j).unsqueeze(1).to(self.flow_ij.device)

        ego_flow_1_2, _ = self.depth_wrapper(self.R_i, self.T_i, self.R_j, self.T_j, 1 / (self.depth_maps_i + 1e-6), self.intrinsics_j, torch.linalg.inv(self.intrinsics_i))
        ego_flow_2_1, _ = self.depth_wrapper(self.R_j, self.T_j, self.R_i, self.T_i, 1 / (self.depth_maps_j + 1e-6), self.intrinsics_i, torch.linalg.inv(self.intrinsics_j))

        err_map_i = torch.norm(ego_flow_1_2[:, :2, ...] - self.flow_ij[:len(symmetry_pairs_idx)], dim=1)
        err_map_j = torch.norm(ego_flow_2_1[:, :2, ...] - self.flow_ji[:len(symmetry_pairs_idx)], dim=1)
        # normalize the error map for each pair
        err_map_i = (err_map_i - err_map_i.amin(dim=(1, 2), keepdim=True)) / (err_map_i.amax(dim=(1, 2), keepdim=True) - err_map_i.amin(dim=(1, 2), keepdim=True))
        err_map_j = (err_map_j - err_map_j.amin(dim=(1, 2), keepdim=True)) / (err_map_j.amax(dim=(1, 2), keepdim=True) - err_map_j.amin(dim=(1, 2), keepdim=True))
        self.dynamic_masks = [[] for _ in range(self.n_imgs)]

        for i, j in symmetry_pairs_idx:
            i_idx = self._ei[i]
            j_idx = self._ej[i]
            self.dynamic_masks[i_idx].append(err_map_i[i])
            self.dynamic_masks[j_idx].append(err_map_j[i])
        
        for i in range(self.n_imgs):
            self.dynamic_masks[i] = torch.stack(self.dynamic_masks[i]).mean(dim=0) > self.motion_mask_thre

    def refine_motion_mask_w_sam2(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Save previous TF32 settings
        if device == 'cuda':
            prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            prev_allow_cudnn_tf32 = torch.backends.cudnn.allow_tf32
            # Enable TF32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        try:
            autocast_dtype = torch.bfloat16 if device == 'cuda' else torch.float32
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                frame_tensors = torch.from_numpy(np.array((self.imgs))).permute(0, 3, 1, 2).to(device)
                inference_state = predictor.init_state(video_path=frame_tensors)
                mask_list = [self.dynamic_masks[i] for i in range(self.n_imgs)]
                
                ann_obj_id = 1
                self.sam2_dynamic_masks = [[] for _ in range(self.n_imgs)]
        
                # Process even frames
                predictor.reset_state(inference_state)
                for idx, mask in enumerate(mask_list):
                    if idx % 2 == 1:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state,
                            frame_idx=idx,
                            obj_id=ann_obj_id,
                            mask=mask,
                        )
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx in range(self.n_imgs):
                    if out_frame_idx % 2 == 0:
                        self.sam2_dynamic_masks[out_frame_idx] = video_segments[out_frame_idx][ann_obj_id]
        
                # Process odd frames
                predictor.reset_state(inference_state)
                for idx, mask in enumerate(mask_list):
                    if idx % 2 == 0:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state,
                            frame_idx=idx,
                            obj_id=ann_obj_id,
                            mask=mask,
                        )
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx in range(self.n_imgs):
                    if out_frame_idx % 2 == 1:
                        self.sam2_dynamic_masks[out_frame_idx] = video_segments[out_frame_idx][ann_obj_id]
        
                # Update dynamic masks
                for i in range(self.n_imgs):
                    self.sam2_dynamic_masks[i] = torch.from_numpy(self.sam2_dynamic_masks[i][0]).to(device)
                    self.dynamic_masks[i] = self.dynamic_masks[i].to(device)
                    self.dynamic_masks[i] = self.dynamic_masks[i] | self.sam2_dynamic_masks[i]
        
                # Clean up
                del predictor
        finally:
            # Restore previous TF32 settings
            if device == 'cuda':
                torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
                torch.backends.cudnn.allow_tf32 = prev_allow_cudnn_tf32


    def _check_all_imgs_are_selected(self, msk):
        self.msk = torch.from_numpy(np.array(msk, dtype=bool)).to(self.device)
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'
        pass

    def preset_pose(self, known_poses, pose_msk=None, requires_grad=False):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        if known_poses[0].shape[-1] == 7: # xyz wxyz
            known_poses = [tum_to_pose_matrix(pose) for pose in known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)
        if len(known_poses) == self.n_imgs:
            if requires_grad:
                self.im_poses.requires_grad_(True)
            else:
                self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_intrinsics(self, known_intrinsics, msk=None):
        if isinstance(known_intrinsics, torch.Tensor) and known_intrinsics.ndim == 2:
            known_intrinsics = [known_intrinsics]
        for K in known_intrinsics:
            assert K.shape == (3, 3)
        self.preset_focal([K.diagonal()[:2].mean() for K in known_intrinsics], msk)
        if self.optimize_pp:
            self.preset_principal_point([K[:2, 2] for K in known_intrinsics], msk)

    def preset_focal(self, known_focals, msk=None, requires_grad=False):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))
        if len(known_focals) == self.n_imgs:
            if requires_grad:
                self.im_focals.requires_grad_(True)
            else:
                self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        if self.shared_focal:
            log_focals = torch.stack([self.im_focals[0]] * self.n_imgs, dim=0)
        else:
            log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()
    
    def get_win_focals(self):
        if self.shared_focal:
            log_focals = torch.stack([self.opt_im_focals[0]] * self.window_size, dim=0)
        else:
            log_focals = torch.stack(list(self.opt_im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points_non_batch(self):
        return torch.stack([pp.new((W/2, H/2))+10*pp for pp, (H, W) in zip(self.im_pp, self.imshapes)])

    def get_principal_points_batch(self):
        return self._pp + 10 * self.im_pp
    
    def get_win_principal_points(self):
        return self._win_pp + 10 * self.win_im_pp

    def get_principal_points(self):
        if self.batchify:
            return self.get_principal_points_batch()
        else:
            return self.get_principal_points_non_batch()

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K
    
    def get_win_intrinsics(self):
        K = torch.zeros((self.window_size, 3, 3), device=self.device)
        focals = self.get_win_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_win_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses_batch(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world
    
    def get_win_im_poses(self):  # cam to world
        if self.window_idx == 0 and self.prev_video_results is None:
            cam2world = self._get_poses(self.opt_im_poses)
        else:
            frozen_cam2world = self._get_poses(self.frozen_im_poses)
            opt_cam2world = self._get_poses(self.opt_im_poses)
            cam2world = torch.cat([frozen_cam2world, opt_cam2world])
        return cam2world
    
    def get_opt_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.opt_im_poses)
        return cam2world

    def get_im_poses_non_batch(self):  # cam to world
        cam2world = self._get_poses(torch.stack(list(self.im_poses)))
        return cam2world

    def get_im_poses(self):
        if self.batchify:
            return self.get_im_poses_batch()
        else:
            return self.get_im_poses_non_batch()

    def _set_depthmap_batch(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def _set_depthmap_non_batch(self, idx, depth, force=False):
        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def _set_depthmap(self, idx, depth, force=False):
        if self.batchify:
            return self._set_depthmap_batch(idx, depth, force)
        else:
            return self._set_depthmap_non_batch(idx, depth, force)
    
    def preset_depthmap(self, known_depthmaps, msk=None, requires_grad=False):
        self._check_all_imgs_are_selected(msk)

        for idx, depth in zip(self._get_msk_indices(msk), known_depthmaps):
            if self.verbose:
                print(f' (setting depthmap #{idx})')
            self._no_grad(self._set_depthmap(idx, depth))

        if len(known_depthmaps) == self.n_imgs:
            if requires_grad:
                self.im_depthmaps.requires_grad_(True)
            else:
                self.im_depthmaps.requires_grad_(False)
    
    def _set_init_depthmap(self):
        depth_maps = self.get_depthmaps(raw=True)
        self.init_depthmap = [dm.detach().clone() for dm in depth_maps]

    def get_init_depthmaps(self, raw=False):
        res = self.init_depthmap
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res
    
    def get_win_init_depthmaps(self, raw=False):
        res = self.win_init_depthmap
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.win_imshapes)]
        return res

    def get_depthmaps_batch(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res
    
    def get_win_depthmaps(self, raw=False):
        if self.window_idx == 0 and self.prev_video_results is None:
            res = self.opt_im_depthmaps.exp()
        else:
            frozen_res = self.frozen_im_depthmaps.exp()
            opt_res = self.opt_im_depthmaps.exp()
            res = torch.cat([frozen_res, opt_res])
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.win_imshapes)]
        return res

    def get_depthmaps_non_batch(self):
        return [d.exp() for d in self.im_depthmaps]

    def get_depthmaps(self, raw=False):
        if self.batchify:
            return self.get_depthmaps_batch(raw)
        else:
            return self.get_depthmaps_non_batch()

    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)
    
    def depth_to_win_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_win_focals()
        pp = self.get_win_principal_points()
        im_poses = self.get_win_im_poses()
        depth = self.get_win_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._win_grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)

    def depth_to_pts3d_partial(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps()

        # convert focal to (1,2,H,W) constant field
        def focal_ex(i): return focals[i][..., None, None].expand(1, *focals[i].shape, *self.imshapes[i])
        # get pointmaps in camera frame
        rel_ptmaps = [depthmap_to_pts3d(depth[i][None], focal_ex(i), pp=pp[i:i+1])[0] for i in range(im_poses.shape[0])]
        # project to world frame
        return [geotrf(pose, ptmap) for pose, ptmap in zip(im_poses, rel_ptmaps)]
    
    def get_pts3d_batch(self, raw=False, **kwargs):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res
    
    def get_win_pts3d(self, raw=False, **kwargs):
        res = self.depth_to_win_pts3d()
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.win_imshapes)]
        return res

    def get_pts3d(self, raw=False, **kwargs):
        if self.batchify:
            return self.get_pts3d_batch(raw, **kwargs)
        else:
            return self.depth_to_pts3d_partial()
        
    def get_win_pw_scale(self):
        scale = self.opt_pw_poses[:, -1].exp()  # (n_edges,)
        scale = scale * self.get_pw_norm_scale_factor()
        return scale
        
    def get_win_pw_poses(self):  # cam to world
        RT = self._get_poses(self.opt_pw_poses)
        scaled_RT = RT.clone()
        scaled_RT[:, :3] *= self.get_win_pw_scale().view(-1, 1, 1)  # scale the rotation AND translation
        return scaled_RT

    def get_win_adaptors(self):
        adapt = self.win_pw_adaptors
        adapt = torch.cat((adapt[:, 0:1], adapt), dim=-1)  # (scale_xy, scale_xy, scale_z)
        if self.norm_pw_scale:  # normalize so that the product == 1
            adapt = adapt - adapt.mean(dim=1, keepdim=True)
        return (adapt / self.pw_break).exp()

    def forward_batchify(self, epoch=9999):
        pw_poses = self.get_pw_poses()  # cam-to-world

        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        # camera temporal loss
        if self.temporal_smoothing_weight > 0:
            temporal_smoothing_loss = self.relative_pose_loss(self.get_im_poses()[:-1], self.get_im_poses()[1:]).sum()
        else:
            temporal_smoothing_loss = 0

        if self.flow_loss_weight > 0 and epoch >= self.num_total_iter * self.flow_loss_start_epoch: # enable flow loss after certain epoch
            R_all, T_all = self.get_im_poses()[:,:3].split([3, 1], dim=-1)
            R1, T1 = R_all[self._ei], T_all[self._ei]
            R2, T2 = R_all[self._ej], T_all[self._ej]
            K_all = self.get_intrinsics()
            inv_K_all = torch.linalg.inv(K_all)
            K_1, inv_K_1 = K_all[self._ei], inv_K_all[self._ei]
            K_2, inv_K_2 = K_all[self._ej], inv_K_all[self._ej]
            depth_all = torch.stack(self.get_depthmaps(raw=False)).unsqueeze(1)
            depth1, depth2 = depth_all[self._ei], depth_all[self._ej]
            disp_1, disp_2 = 1 / (depth1 + 1e-6), 1 / (depth2 + 1e-6)
            ego_flow_1_2, _ = self.depth_wrapper(R1, T1, R2, T2, disp_1, K_2, inv_K_1)
            ego_flow_2_1, _ = self.depth_wrapper(R2, T2, R1, T1, disp_2, K_1, inv_K_2)
            dynamic_masks_all = torch.stack(self.dynamic_masks).to(self.device).unsqueeze(1)
            dynamic_mask1, dynamic_mask2 = dynamic_masks_all[self._ei], dynamic_masks_all[self._ej]

            flow_loss_i = self.flow_loss_fn(ego_flow_1_2[:, :2, ...], self.flow_ij, ~dynamic_mask1, per_pixel_thre=self.pxl_thre)
            flow_loss_j = self.flow_loss_fn(ego_flow_2_1[:, :2, ...], self.flow_ji, ~dynamic_mask2, per_pixel_thre=self.pxl_thre)
            flow_loss = flow_loss_i + flow_loss_j

            if flow_loss.item() > self.flow_loss_thre and self.flow_loss_thre > 0: 
                flow_loss = 0
                self.flow_loss_flag = True
        else:    
            flow_loss = 0

        if self.depth_regularize_weight > 0:
            init_depthmaps = torch.stack(self.get_init_depthmaps(raw=False)).unsqueeze(1)
            depthmaps = torch.stack(self.get_depthmaps(raw=False)).unsqueeze(1)
            dynamic_masks_all = torch.stack(self.dynamic_masks).to(self.device).unsqueeze(1)
            depth_prior_loss = self.depth_regularizer(depthmaps, init_depthmaps, dynamic_masks_all)
        else:
            depth_prior_loss = 0

        loss = (li + lj) * 1 + self.temporal_smoothing_weight * temporal_smoothing_loss + \
                self.flow_loss_weight * flow_loss + self.depth_regularize_weight * depth_prior_loss

        return loss, flow_loss
    
    def forward_window_wise(self, epoch=9999):
        pw_poses = self.get_win_pw_poses()  # cam-to-world

        pw_adapt = self.get_win_adaptors().unsqueeze(1)
        proj_pts3d = self.get_win_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._win_stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._win_stacked_pred_j)

        # compute the loss
        li = self.dist(proj_pts3d[self._win_ei], aligned_pred_i, weight=self._win_weight_i).sum() / self.win_total_area_i
        lj = self.dist(proj_pts3d[self._win_ej], aligned_pred_j, weight=self._win_weight_j).sum() / self.win_total_area_j

        # camera temporal loss
        if self.temporal_smoothing_weight > 0:
            temporal_smoothing_loss = self.relative_pose_loss(self.get_win_im_poses()[:-1], self.get_win_im_poses()[1:]).sum()
        else:
            temporal_smoothing_loss = 0

        if self.flow_loss_weight > 0 and epoch >= self.num_total_iter * self.flow_loss_start_epoch: # enable flow loss after certain epoch
            R_all, T_all = self.get_win_im_poses()[:,:3].split([3, 1], dim=-1)
            R1, T1 = R_all[self._win_ei], T_all[self._win_ei]
            R2, T2 = R_all[self._win_ej], T_all[self._win_ej]
            K_all = self.get_win_intrinsics()
            inv_K_all = torch.linalg.inv(K_all)
            K_1, inv_K_1 = K_all[self._win_ei], inv_K_all[self._win_ei]
            K_2, inv_K_2 = K_all[self._win_ej], inv_K_all[self._win_ej]
            depth_all = torch.stack(self.get_win_depthmaps(raw=False)).unsqueeze(1)
            depth1, depth2 = depth_all[self._win_ei], depth_all[self._win_ej]
            disp_1, disp_2 = 1 / (depth1 + 1e-6), 1 / (depth2 + 1e-6)
            ego_flow_1_2, _ = self.depth_wrapper(R1, T1, R2, T2, disp_1, K_2, inv_K_1)
            ego_flow_2_1, _ = self.depth_wrapper(R2, T2, R1, T1, disp_2, K_1, inv_K_2)
            dynamic_masks_all = torch.stack(self.win_dynamic_masks).to(self.device).unsqueeze(1)
            dynamic_mask1, dynamic_mask2 = dynamic_masks_all[self._win_ei], dynamic_masks_all[self._win_ej]

            flow_loss_i = self.flow_loss_fn(ego_flow_1_2[:, :2, ...], self.win_flow_ij, ~dynamic_mask1, per_pixel_thre=self.pxl_thre)
            flow_loss_j = self.flow_loss_fn(ego_flow_2_1[:, :2, ...], self.win_flow_ji, ~dynamic_mask2, per_pixel_thre=self.pxl_thre)
            flow_loss = flow_loss_i + flow_loss_j

            if flow_loss.item() > self.flow_loss_thre and self.flow_loss_thre > 0: 
                flow_loss = 0
                self.flow_loss_flag = True
        else:    
            flow_loss = 0

        if self.depth_regularize_weight > 0:
            init_depthmaps = torch.stack(self.get_win_init_depthmaps(raw=False)).unsqueeze(1)
            depthmaps = torch.stack(self.get_win_depthmaps(raw=False)).unsqueeze(1)
            dynamic_masks_all = torch.stack(self.win_dynamic_masks).to(self.device).unsqueeze(1)
            depth_prior_loss = self.depth_regularizer(depthmaps, init_depthmaps, dynamic_masks_all)
        else:
            depth_prior_loss = 0

        loss = (li + lj) * 1 + self.temporal_smoothing_weight * temporal_smoothing_loss + \
                self.flow_loss_weight * flow_loss + self.depth_regularize_weight * depth_prior_loss
        
        return loss, flow_loss
    
    def forward_non_batchify(self, epoch=9999):

        # --(1) Perform the original pairwise 3D consistency loss (pairwise 3D consistency)--
        pw_poses = self.get_pw_poses()  # pair-wise poses (or adaptive poses)
        pw_adapt = self.get_adaptors()
        proj_pts3d = self.get_pts3d()   # 3D point clouds for each image
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0.0
        for e, (i, j) in enumerate(self.edges):
            i_j = edge_str(i, j)
            # Transform the pairwise predictions to the world coordinate system
            aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j])
            aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])
            # Compute the distance loss between the projected point clouds and the predictions
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[i_j]).mean()
            loss += (li + lj)

        # Average the loss
        loss /= self.n_edges

        # --(2) Add temporal smoothing constraint between adjacent frames (temporal smoothing)--
        temporal_smoothing_loss = 0.0
        if self.temporal_smoothing_weight > 0:
            # Get the global poses (4x4) for all images
            im_poses = self.get_im_poses()  # shape: (n_imgs, 4, 4)
            # Stack the relative poses between adjacent frames and use the existing relative_pose_loss function
            rel_RT1, rel_RT2 = [], []
            for idx in range(self.n_imgs - 1):
                rel_RT1.append(im_poses[idx])
                rel_RT2.append(im_poses[idx + 1])
            if len(rel_RT1) > 0:
                rel_RT1 = torch.stack(rel_RT1, dim=0)  # shape: (n_imgs-1, 4, 4)
                rel_RT2 = torch.stack(rel_RT2, dim=0)
                # Compute the pose difference between adjacent frames
                temporal_smoothing_loss = self.relative_pose_loss(rel_RT1, rel_RT2).sum()
                loss += self.temporal_smoothing_weight * temporal_smoothing_loss

        # --(3) Add flow constraint (flow_loss), similar to forward_batchify--
        flow_loss = 0.0
        if self.flow_loss_weight > 0 and epoch >= self.num_total_iter * self.flow_loss_start_epoch:
            # Iterate through each pair of images and compute the depth map and flow comparison
            im_poses = self.get_im_poses()   # (n_imgs, 4, 4)
            K_all = self.get_intrinsics()    # (n_imgs, 3, 3)
            inv_K_all = torch.linalg.inv(K_all)
            depthmaps = self.get_depthmaps(raw=False)  # list of depth maps (H, W)

            for e, (i, j) in enumerate(self.edges):
                # Get the rotation, translation, and intrinsics for the two frames
                R1 = im_poses[i][:3, :3].unsqueeze(0)  # shape: (1, 3, 3)
                T1 = im_poses[i][:3, 3].unsqueeze(-1).unsqueeze(0)  # (1, 3, 1)
                R2 = im_poses[j][:3, :3].unsqueeze(0)
                T2 = im_poses[j][:3, 3].unsqueeze(-1).unsqueeze(0)
                K1 = K_all[i].unsqueeze(0)     # (1, 3, 3)
                K2 = K_all[j].unsqueeze(0)
                inv_K1 = inv_K_all[i].unsqueeze(0)
                inv_K2 = inv_K_all[j].unsqueeze(0)

                # Construct disparity: disp = 1/depth
                depth1 = depthmaps[i].unsqueeze(0).unsqueeze(1)  # (1, 1, H, W)
                depth2 = depthmaps[j].unsqueeze(0).unsqueeze(1)
                disp_1 = 1.0 / (depth1 + 1e-6)
                disp_2 = 1.0 / (depth2 + 1e-6)

                # Compute "ego-motion flow" by projecting using DepthBasedWarping
                # Note that DepthBasedWarping expects batch dimension, so add unsqueeze(0)
                ego_flow_1_2, _ = self.depth_wrapper(R1, T1, R2, T2, disp_1, K2, inv_K1)
                ego_flow_2_1, _ = self.depth_wrapper(R2, T2, R1, T1, disp_2, K1, inv_K2)

                # Get the corresponding dynamic region masks (if any)
                dynamic_mask_i = self.dynamic_masks[i]  # shape: (H, W)
                dynamic_mask_j = self.dynamic_masks[j]

                # When computing flow loss, exclude or ignore dynamic regions
                flow_loss_i = self.flow_loss_fn(
                    ego_flow_1_2[0, :2, ...],   # shape: (2, H, W)
                    self.flow_ij[e],           # shape: (2, H, W),  i->j
                    ~dynamic_mask_i,           # mask: True = keep, False = ignore
                    per_pixel_thre=self.pxl_thre
                )
                flow_loss_j = self.flow_loss_fn(
                    ego_flow_2_1[0, :2, ...],
                    self.flow_ji[e],           # j->i
                    ~dynamic_mask_j,
                    per_pixel_thre=self.pxl_thre
                )
                flow_loss += (flow_loss_i + flow_loss_j)

            # Optional: handle cases where the flow loss is too large (e.g., early stop)
            # divide by the number of edges
            flow_loss /= self.n_edges
            print(f'flow loss: {flow_loss.item()}')
            if flow_loss.item() > self.flow_loss_thre and self.flow_loss_thre > 0:
                flow_loss = 0.0

            loss += self.flow_loss_weight * flow_loss

        # --(4) Add depth regularization (depth_prior_loss) to constrain the initial depth--
        if self.depth_regularize_weight > 0:
            init_depthmaps = self.get_init_depthmaps(raw=False)  # initial depth maps
            current_depthmaps = self.get_depthmaps(raw=False)     # current optimized depth maps
            depth_prior_loss = 0.0
            for i in range(self.n_imgs):
                # Apply constraints on static regions (ignore dynamic regions)
                # Make sure the shape has the batch dimension (B,1,H,W)
                depth_prior_loss += self.depth_regularizer(
                    current_depthmaps[i].unsqueeze(0).unsqueeze(1),
                    init_depthmaps[i].unsqueeze(0).unsqueeze(1),
                    self.dynamic_masks[i].unsqueeze(0).unsqueeze(1)
                )
            loss += self.depth_regularize_weight * depth_prior_loss

        return loss, flow_loss

    def forward(self, epoch=9999):
        if self.batchify:
            return self.forward_batchify(epoch)
        elif self.window_wise:
            return self.forward_window_wise(epoch)
        else:
            return self.forward_non_batchify(epoch)
        
    def clean_prev_results(self):
        self.n_imgs = self.n_imgs - self.n_prev_frames
        self.im_poses = self.im_poses[self.n_prev_frames:]
        self.im_pp = self.im_pp[self.n_prev_frames:]
        self.imshapes = self.imshapes[self.n_prev_frames:]
        self.im_depthmaps = self.im_depthmaps[self.n_prev_frames:]
        self.im_conf = self.im_conf[self.n_prev_frames:]
        self.init_conf_maps = self.init_conf_maps[self.n_prev_frames:]
        self.imgs = self.imgs[self.n_prev_frames:]
        self.dynamic_masks = self.dynamic_masks[self.n_prev_frames:]
        if getattr(self, 'sam2_dynamic_masks', None) is not None:
            self.sam2_dynamic_masks = self.sam2_dynamic_masks[self.n_prev_frames:]
        
    def relative_pose_loss(self, RT1, RT2):
        relative_RT = torch.matmul(torch.inverse(RT1), RT2)
        rotation_diff = relative_RT[:, :3, :3]
        translation_diff = relative_RT[:, :3, 3]

        # Frobenius norm for rotation difference
        rotation_loss = torch.norm(rotation_diff - (torch.eye(3, device=RT1.device)), dim=(1, 2))

        # L2 norm for translation difference
        translation_loss = torch.norm(translation_diff, dim=1)

        # Combined loss (one can weigh these differently if needed)
        pose_loss = rotation_loss + translation_loss * self.translation_weight
        return pose_loss

def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img

def ordered_ratio(disp_a, disp_b, mask=None):
    ratio_a = torch.maximum(disp_a, disp_b) / \
        (torch.minimum(disp_a, disp_b)+1e-5)
    if mask is not None:
        ratio_a = ratio_a[mask]
    return ratio_a - 1
