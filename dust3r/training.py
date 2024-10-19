# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
import os
os.environ['OMP_NUM_THREADS'] = '4' # will affect the performance of pairwise prediction
import argparse
import datetime
import json
import numpy as np
import sys
import time
import math
import wandb
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch, visualize_results  # noqa

from dust3r.pose_eval import eval_pose_estimation
from dust3r.depth_eval import eval_mono_depth_estimation

# from demo import get_3D_model_from_scene
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa


def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', \
                        img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--train_criterion', default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")

    # dataset
    parser.add_argument('--train_dataset', default='[None]', type=str, help="training set")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    parser.add_argument("--eval_only", action='store_true', default=False)
    parser.add_argument("--fixed_eval_set", action='store_true', default=False)
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=5, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')
    parser.add_argument('--wandb', action='store_true', default=False, help='use wandb for logging')
    parser.add_argument('--num_save_visual', default=1, type=int, help='number of visualizations to save')
    
    # switch mode for train / eval pose / eval depth
    parser.add_argument('--mode', default='train', type=str, help='train / eval_pose / eval_depth')

    # for pose eval
    parser.add_argument('--pose_eval_freq', default=0, type=int, help='pose evaluation frequency')
    parser.add_argument('--pose_eval_stride', default=1, type=int, help='stride for pose evaluation')
    parser.add_argument('--scene_graph_type', default='swinstride-5-noncyclic', type=str, help='scene graph window size')
    parser.add_argument('--save_best_pose', action='store_true', default=False, help='save best pose')
    parser.add_argument('--n_iter', default=300, type=int, help='number of iterations for pose optimization')
    parser.add_argument('--save_pose_qualitative', action='store_true', default=False, help='save qualitative pose results')
    parser.add_argument('--temporal_smoothing_weight', default=0.01, type=float, help='temporal smoothing weight for pose optimization')
    parser.add_argument('--not_shared_focal', action='store_true', default=False, help='use shared focal length for pose optimization')
    parser.add_argument('--use_gt_focal', action='store_true', default=False, help='use ground truth focal length for pose optimization')
    parser.add_argument('--pose_schedule', default='linear', type=str, help='pose optimization schedule')
    
    parser.add_argument('--flow_loss_weight', default=0.01, type=float, help='flow loss weight for pose optimization')
    parser.add_argument('--flow_loss_fn', default='smooth_l1', type=str, help='flow loss type for pose optimization')
    parser.add_argument('--use_gt_mask', action='store_true', default=False, help='use gt mask for pose optimization, for sintel/davis')
    parser.add_argument('--sam2_mask_refine', action='store_true', default=False, help='use sam2 mask refine for the motion for pose optimization')
    parser.add_argument('--flow_loss_start_epoch', default=0.1, type=float, help='start epoch for flow loss')
    parser.add_argument('--flow_loss_thre', default=20, type=float, help='threshold for flow loss')
    parser.add_argument('--pxl_thresh', default=50.0, type=float, help='threshold for flow loss')
    parser.add_argument('--depth_regularize_weight', default=0.0, type=float, help='depth regularization weight for pose optimization')
    parser.add_argument('--translation_weight', default=1, type=float, help='translation weight for pose optimization')
    parser.add_argument('--silent', action='store_true', default=False, help='silent mode for pose evaluation')
    parser.add_argument('--full_seq', action='store_true', default=False, help='use full sequence for pose evaluation')
    parser.add_argument('--seq_list', nargs='+', default=None, help='list of sequences for pose evaluation')

    parser.add_argument('--eval_dataset', type=str, default='sintel', 
                    choices=['davis', 'kitti', 'kitti_new', 'shibuya', 'bonn', 'bonn_new', 'scannet', 'tum', 'tum_new', 'nyu'], 
                    help='choose dataset for pose evaluation')

    # for monocular depth eval
    parser.add_argument('--no_crop', action='store_true', default=False, help='do not crop the image for monocular depth evaluation')

    # output dir
    parser.add_argument('--output_dir', default='./results/tmp', type=str, help="path where to save the output")
    return parser

def load_model(args, device):
    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    model.to(device)
    model_without_ddp = model
    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    return model, model_without_ddp

def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()
    # if main process, init wandb
    if args.wandb and misc.is_main_process():
        wandb.init(name=args.output_dir.split('/')[-1], 
                   project='dust3r', 
                   config=args, 
                   sync_tensorboard=True,
                   dir=args.output_dir)

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume if not specified
    if args.resume is None:
        last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
        if os.path.isfile(last_ckpt_fname) and (not args.eval_only): args.resume = last_ckpt_fname

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = args.cudnn_benchmark

    model, model_without_ddp = load_model(args, device)

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    print('Building test dataset {:s}'.format(args.train_dataset))
    data_loader_test = {}
    for dataset in args.test_dataset.split('+'):
        testset = build_dataset(dataset, args.batch_size, args.num_workers, test=True)
        name_testset = dataset.split('(')[0]
        if getattr(testset.dataset.dataset, 'strides', None) is not None:
            name_testset += f'_stride{testset.dataset.dataset.strides}'
        data_loader_test[name_testset] = testset

    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            gathered_test_stats = {}
            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})

            for test_name, testset in data_loader_test.items():

                if test_name not in test_stats:
                    continue

                if getattr(testset.dataset.dataset, 'strides', None) is not None:
                    original_test_name = test_name.split('_stride')[0]
                    if original_test_name not in gathered_test_stats.keys():
                        gathered_test_stats[original_test_name] = []
                    gathered_test_stats[original_test_name].append(test_stats[test_name])

                log_stats.update({test_name + '_' + k: v for k, v in test_stats[test_name].items()})

            if len(gathered_test_stats) > 0:
                for original_test_name, stride_stats in gathered_test_stats.items():
                    if len(stride_stats) > 1:
                        stride_stats = {k: np.mean([x[k] for x in stride_stats]) for k in stride_stats[0]}
                        log_stats.update({original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()})
                        if args.wandb:
                            log_dict = {original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()}
                            log_dict.update({'epoch': epoch})
                            wandb.log(log_dict)

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far, best_pose_ate_sofar=None):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far, best_pose_ate_sofar=best_pose_ate_sofar)

    best_so_far, best_pose_ate_sofar = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
    if best_pose_ate_sofar is None:
        best_pose_ate_sofar = float('inf')
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):

        # Test on multiple datasets
        new_best = False
        new_pose_best = False
        already_saved = False
        if (epoch > args.start_epoch and args.eval_freq > 0 and epoch % args.eval_freq == 0) or args.eval_only:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(model, test_criterion, testset,
                                       device, epoch, log_writer=log_writer, args=args, prefix=test_name)
                test_stats[test_name] = stats

                # Save best of all
                if stats['loss_med'] < best_so_far:
                    best_so_far = stats['loss_med']
                    new_best = True

            # Ensure that eval_pose_estimation is only run on the main process
            if args.pose_eval_freq>0 and (epoch % args.pose_eval_freq==0 or args.eval_only):
                ate_mean, rpe_trans_mean, rpe_rot_mean, outfile_list, bug = eval_pose_estimation(args, model, device, save_dir=f'{args.output_dir}/{epoch}')
                print(f'ATE mean: {ate_mean}, RPE trans mean: {rpe_trans_mean}, RPE rot mean: {rpe_rot_mean}')
                
                # Optionally log the results to wandb
                if args.wandb and misc.is_main_process():
                    wandb_dict = {
                        'epoch': epoch,
                        'ATE mean': ate_mean,
                        'RPE trans mean': rpe_trans_mean,
                        'RPE rot mean': rpe_rot_mean,
                    }
                    if args.save_pose_qualitative:
                        for outfile in outfile_list:
                            wandb_dict[outfile.split('/')[-1]] = wandb.Object3D(open(outfile))
                    
                    wandb.log(wandb_dict)

                if ate_mean < best_pose_ate_sofar and not bug: # if the pose estimation is better, and w/o any error
                    best_pose_ate_sofar = ate_mean
                    new_pose_best = True

            # Synchronize all processes to ensure eval_pose_estimation is completed
            try:
                torch.distributed.barrier()
            except:
                pass

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if args.eval_only and args.epochs <= 1:
            exit(0)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far, best_pose_ate_sofar)
                already_saved = True
            if new_best:
                save_model(epoch - 1, 'best', best_so_far, best_pose_ate_sofar)
                already_saved = True
            if new_pose_best and args.save_best_pose:
                save_model(epoch - 1, 'best_pose', best_so_far, best_pose_ate_sofar)
                already_saved = True

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs and not already_saved:
                save_model(epoch - 1, 'last', best_so_far, best_pose_ate_sofar)

        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)
        
        batch_result = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp))
        loss, loss_details = batch_result['loss']  # criterion returns two values
        loss_value = float(loss)

        if (data_iter_step % max((len(data_loader) // args.num_save_visual), 1) == 0) and misc.is_main_process() :
            save_dir = f'{args.output_dir}/{epoch}'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            view1, view2, pred1, pred2 = batch_result['view1'], batch_result['view2'], batch_result['pred1'], batch_result['pred2']
            gt_visual = visualize_results(view1, view2, pred1, pred2, save_dir=save_dir, visualize_type='gt')
            pred_visual = visualize_results(view1, view2, pred1, pred2, save_dir=save_dir, visualize_type='pred')

            if args.wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_visual_gt': wandb.Object3D(open(gt_visual)),
                    'train_visual_pred': wandb.Object3D(open(pred_visual))
                })

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar('train_' + name, val, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch) if not args.fixed_eval_set else data_loader.dataset.set_epoch(0)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch) if not args.fixed_eval_set else data_loader.sampler.set_epoch(0)

    for idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        
        batch_result = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp))
        loss_tuple = batch_result['loss']
        loss_value, loss_details = loss_tuple  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

        if args.num_save_visual>0 and (idx % max((len(data_loader) // args.num_save_visual), 1) == 0) and misc.is_main_process() : # save visualizations
            save_dir = f'{args.output_dir}/{epoch}'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            view1, view2, pred1, pred2 = batch_result['view1'], batch_result['view2'], batch_result['pred1'], batch_result['pred2']
            gt_visual = visualize_results(view1, view2, pred1, pred2, save_dir=save_dir, visualize_type='gt')
            pred_visual = visualize_results(view1, view2, pred1, pred2, save_dir=save_dir, visualize_type='pred')
            
            if args.wandb:
                wandb.log({
                    'epoch': epoch,
                    'test_visual_gt': wandb.Object3D(open(gt_visual)),
                    'test_visual_pred': wandb.Object3D(open(pred_visual))
                })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    return results