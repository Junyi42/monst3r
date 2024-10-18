import torch
import numpy as np
import cv2
import glob
from pathlib import Path
from tqdm import tqdm
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from copy import deepcopy
from scipy.optimize import minimize
import os
from collections import defaultdict
import dust3r.eval_metadata
from dust3r.eval_metadata import dataset_metadata

def eval_mono_depth_estimation(args, model, device):
    metadata = dataset_metadata.get(args.eval_dataset)
    if metadata is None:
        raise ValueError(f"Unknown dataset: {args.eval_dataset}")
    
    img_path = metadata.get('img_path')
    if 'img_path_func' in metadata:
        img_path = metadata['img_path_func'](args)
    
    process_func = metadata.get('process_func')
    if process_func is None:
        raise ValueError(f"No processing function defined for dataset: {args.eval_dataset}")
    
    for filelist, save_dir in process_func(args, img_path):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        eval_mono_depth(args, model, device, filelist, save_dir=save_dir)


def eval_mono_depth(args, model, device, filelist, save_dir=None):
    model.eval()
    load_img_size = 512
    for file in tqdm(filelist):
        # construct the "image pair" for the single image
        file = [file]
        imgs = load_images(file, size=load_img_size, verbose=False, crop= not args.no_crop)
        imgs = [imgs[0], deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

        pairs = make_pairs(imgs, symmetrize=True, prefilter=None)
        output = inference(pairs, model, device, batch_size=1, verbose=False)
        depth_map = output['pred1']['pts3d'][...,-1].mean(dim=0)

        if save_dir is not None:
            #save the depth map to the save_dir as npy
            np.save(f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.npy')}", depth_map.cpu().numpy())
            # also save the png
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map = (depth_map * 255).cpu().numpy().astype(np.uint8)
            cv2.imwrite(f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.png')}", depth_map)



## used for calculating the depth evaluation metrics


def group_by_directory(pathes, idx=-1):
    """
    Groups the file paths based on the second-to-last directory in their paths.

    Parameters:
    - pathes (list): List of file paths.

    Returns:
    - dict: A dictionary where keys are the second-to-last directory names and values are lists of file paths.
    """
    grouped_pathes = defaultdict(list)

    for path in pathes:
        # Extract the second-to-last directory
        dir_name = os.path.dirname(path).split('/')[idx]
        grouped_pathes[dir_name].append(path)
    
    return grouped_pathes


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params

    predicted_aligned = s * predicted_depth + t

    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)

def absolute_value_scaling(predicted_depth, ground_truth_depth, s=1, t=0):
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)
    
    initial_params = [s, t]  # s = 1, t = 0
    
    result = minimize(absolute_error_loss, initial_params, args=(predicted_depth_np, ground_truth_depth_np))
    
    s, t = result.x  
    return s, t

def absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=1.0, t_init=0.0, lr=1e-4, max_iters=1000, tol=1e-6):
    # Initialize s and t as torch tensors with requires_grad=True
    s = torch.tensor([s_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
    t = torch.tensor([t_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)

    optimizer = torch.optim.Adam([s, t], lr=lr)
    
    prev_loss = None

    for i in range(max_iters):
        optimizer.zero_grad()

        # Compute predicted aligned depth
        predicted_aligned = s * predicted_depth + t

        # Compute absolute error
        abs_error = torch.abs(predicted_aligned - ground_truth_depth)

        # Compute loss
        loss = torch.sum(abs_error)

        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        # Check convergence
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss.item()

    return s.detach().item(), t.detach().item()

def depth_evaluation(predicted_depth_original, ground_truth_depth_original, max_depth=80, custom_mask=None, post_clip_min=None, post_clip_max=None, pre_clip_min=None, pre_clip_max=None,
                     align_with_lstsq=False, align_with_lad=False, align_with_lad2=False, lr=1e-4, max_iters=1000, use_gpu=False, align_with_scale=False,
                     disp_input=False):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.
    
    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map.
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map.
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.
    
    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    # if the dimension is 3, flatten to 2d along the batch dimension
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # put to device
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()
    
    # Filter out depths greater than max_depth
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original < max_depth)
    else:
        mask = (ground_truth_depth_original > 0)
    
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input: # align the pred to gt in the disparity space
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    # various alignment methods
    if align_with_lstsq:
        # Convert to numpy for lstsq
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)
        
        # Add a column of ones for the shift term
        A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])
        
        # Solve for scale (s) and shift (t) using least squares
        result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
        s, t = result[0][0], result[0][1]

        # convert to torch tensor
        s = torch.tensor(s, device=predicted_depth_original.device)
        t = torch.tensor(t, device=predicted_depth_original.device)
        
        # Apply scale and shift
        predicted_depth = s * predicted_depth + t
    elif align_with_lad:
        s, t = absolute_value_scaling(predicted_depth, ground_truth_depth, s=torch.median(ground_truth_depth) / torch.median(predicted_depth))
        predicted_depth = s * predicted_depth + t
    elif align_with_lad2:
        s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
        s, t = absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=s_init, lr=lr, max_iters=max_iters)
        predicted_depth = s * predicted_depth + t
    elif align_with_scale:
        # Compute initial scale factor 's' using the closed-form solution (L2 norm)
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred

        # Iterative reweighted least squares using the Weiszfeld method
        for _ in range(10):
            # Compute residuals between scaled predictions and ground truth
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = residuals.abs() + 1e-8  # Add small constant to avoid division by zero
            
            # Compute weights inversely proportional to the residuals
            weights = 1.0 / abs_residuals
            
            # Update 's' using weighted sums
            weighted_dot_pred_gt = torch.sum(weights * predicted_depth * ground_truth_depth)
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth ** 2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred

        # Optionally clip 's' to prevent extreme scaling
        s = s.clamp(min=1e-3)
        
        # Detach 's' if you want to stop gradients from flowing through it
        s = s.detach()
        
        # Apply the scale factor to the predicted depth
        predicted_depth = s * predicted_depth

    else:
        # Align the predicted depth with the ground truth using median scaling
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth *= scale_factor

    if disp_input:
        # convert back to depth
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask.cpu()[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    # Calculate the metrics
    abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth).item()
    sq_rel = torch.mean(((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth).item()
    
    # Correct RMSE calculation
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()
    
    # Clip the depth values to avoid log(0)
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)).item()
    
    # Calculate the accuracy thresholds
    max_ratio = torch.maximum(predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth)
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25 ** 2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25 ** 3).float()).item()

    # Compute the depth error parity map
    if align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original = predicted_depth_original * s + t
        if disp_input: predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    elif align_with_scale:
        predicted_depth_original = predicted_depth_original * s
        if disp_input: predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    else:
        predicted_depth_original = predicted_depth_original * scale_factor
        if disp_input: predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    
    # Reshape the depth_error_parity_map back to the original image size
    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(mask, depth_error_parity_map, depth_error_parity_map_full)

    predict_depth_map_full = predicted_depth_original

    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(mask, ground_truth_depth_original, gt_depth_map_full)

    num_valid_pixels = torch.sum(mask).item() if custom_mask is None else torch.sum(mask_within_mask).item()
    if num_valid_pixels == 0:
        abs_rel, sq_rel, rmse, log_rmse, threshold_1, threshold_2, threshold_3 = 0, 0, 0, 0, 0, 0, 0

    results = {
        'Abs Rel': abs_rel,
        'Sq Rel': sq_rel,
        'RMSE': rmse,
        'Log RMSE': log_rmse,
        'δ < 1.25': threshold_1,
        'δ < 1.25^2': threshold_2,
        'δ < 1.25^3': threshold_3,
        'valid_pixels': num_valid_pixels
    }

    return results, depth_error_parity_map_full, predict_depth_map_full, gt_depth_map_full
