# --------------------------------------------------------
# training executable for DUSt3R
# --------------------------------------------------------
from dust3r.training import get_args_parser, train, load_model
from dust3r.pose_eval import eval_pose_estimation
from dust3r.depth_eval import eval_mono_depth_estimation
import croco.utils.misc as misc  # noqa
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.mode.startswith('eval'):
        misc.init_distributed_mode(args)
        global_rank = misc.get_rank()
        world_size = misc.get_world_size()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # fix the seed
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = args.cudnn_benchmark
        model, _ = load_model(args, device)
        os.makedirs(args.output_dir, exist_ok=True)

        if args.mode == 'eval_pose':
            ate_mean, rpe_trans_mean, rpe_rot_mean, outfile_list, bug = eval_pose_estimation(args, model, device, save_dir=args.output_dir)
            print(f'ATE mean: {ate_mean}, RPE trans mean: {rpe_trans_mean}, RPE rot mean: {rpe_rot_mean}')
        if args.mode == 'eval_depth':
            eval_mono_depth_estimation(args, model, device)

        exit(0)
    train(args)
