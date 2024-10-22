# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import argparse
import math
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import copy

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb, get_dynamic_mask_from_pairviewer
import matplotlib.pyplot as pl

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--weights", type=str, help="path to the model weights", default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument("--model_name", type=str, default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth', help="model name")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp', help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--input_dir", type=str, help="Path to input images directory", default=None)
    parser.add_argument("--seq_name", type=str, help="Sequence name for evaluation", default='NULL')
    parser.add_argument('--use_gt_davis_masks', action='store_true', default=False, help='Use ground truth masks for DAVIS')
    parser.add_argument('--fps', type=int, default=0, help='FPS for video processing')
    parser.add_argument('--num_frames', type=int, default=200, help='Maximum number of frames for video processing')
    
    # Add "share" argument if you want to make the demo accessible on the public internet
    parser.add_argument("--share", action='store_true', default=False, help="Share the demo")
    return parser

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
    scene.min_conf_thr = min_conf_thr
    scene.thr_for_init_conf = thr_for_init_conf
    msk = to_numpy(scene.get_masks())
    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]
    return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
                                        cam_color=cam_color)


def get_reconstructed_scene(args, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask, fps, num_frames):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()
    if seq_name != "NULL":
        dynamic_mask_path = f'data/davis/DAVIS/masked_images/480p/{seq_name}'
    else:
        dynamic_mask_path = None
    imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path, fps=fps, num_frames=num_frames)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer  
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                               num_total_iter=niter, empty_cache= len(filelist) > 72)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)
    outfile = get_3D_model_from_scene(save_folder, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                            clean_depth, transparent_cams, cam_size, show_cam)

    poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    depth_maps = scene.save_depth_maps(save_folder)
    dynamic_masks = scene.save_dynamic_masks(save_folder)
    conf = scene.save_conf_maps(save_folder)
    init_conf = scene.save_init_conf_maps(save_folder)
    rgbs = scene.save_rgb_imgs(save_folder)
    enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    init_confs = to_numpy([c for c in scene.init_conf_maps])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [cmap(d/depths_max) for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]
    init_confs_max = max([d.max() for d in init_confs])
    init_confs = [cmap(d/init_confs_max) for d in init_confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))
        imgs.append(rgb(init_confs[i]))

    # if two images, and the shape is same, we can compute the dynamic mask
    if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
        error_map = get_dynamic_mask_from_pairviewer(scene, both_directions=True, output_dir=args.output_dir)
        # imgs.append(rgb(error_map))
        # apply threshold on the error map
        normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        error_map_max = normalized_error_map.max()
        error_map = cmap(normalized_error_map/error_map_max)
        imgs.append(rgb(error_map))
        binary_error_map = (normalized_error_map > 0.3).astype(np.uint8)
        imgs.append(rgb(binary_error_map*255))

    return scene, outfile, imgs


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    # if inputfiles[0] is a video, set the num_files to 200
    if inputfiles is not None and len(inputfiles) == 1 and inputfiles[0].name.endswith(('.mp4', '.avi', '.mov')):
        num_files = 200
    else:
        num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files-1)/2))
    if scenegraph_type == "swin" or scenegraph_type == "swin2stride" or scenegraph_type == "swinstride":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=min(max_winsize,5),
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    return winsize, refid


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False, args=None):
    recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="DUSt3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML(f'<h2 style="text-align: center;">DUSt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!")
                niter = gradio.Number(value=300, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                seq_name = gradio.Textbox(label="Sequence Name", placeholder="NULL", value=args.seq_name, info="For evaluation")
                scenegraph_type = gradio.Dropdown(["complete", "swin", "oneref", "swinstride", "swin2stride"],
                                                  value='swinstride', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=5,
                                        minimum=1, maximum=1, step=1, visible=False)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence thresholdx
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.1, minimum=0.0, maximum=20, step=0.01)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
                # adjust the temporal smoothing weight
                temporal_smoothing_weight = gradio.Slider(label="temporal_smoothing_weight", value=0.01, minimum=0.0, maximum=0.1, step=0.001)
                # add translation weight
                translation_weight = gradio.Textbox(label="translation_weight", placeholder="1.0", value="1.0", info="For evaluation")
                # change to another model
                new_model_weights = gradio.Textbox(label="New Model", placeholder=args.weights, value=args.weights, info="Path to updated model weights")
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                # not to show camera
                show_cam = gradio.Checkbox(value=True, label="Show Camera")
                shared_focal = gradio.Checkbox(value=True, label="Shared Focal Length")
                use_davis_gt_mask = gradio.Checkbox(value=False, label="Use GT Mask (DAVIS)")
            with gradio.Row():
                flow_loss_weight = gradio.Slider(label="Flow Loss Weight", value=0.01, minimum=0.0, maximum=1.0, step=0.001)
                flow_loss_start_iter = gradio.Slider(label="Flow Loss Start Iter", value=0.1, minimum=0, maximum=1, step=0.01)
                flow_loss_threshold = gradio.Slider(label="Flow Loss Threshold", value=25, minimum=0, maximum=100, step=1)
                # for video processing
                fps = gradio.Slider(label="FPS", value=0, minimum=0, maximum=60, step=1)
                num_frames = gradio.Slider(label="Num Frames", value=100, minimum=0, maximum=200, step=1)

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb,depth,confidence, init_conf', columns=4, height="100%")

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                                  mask_sky, clean_depth, transparent_cams, cam_size, show_cam,
                                  scenegraph_type, winsize, refid, seq_name, new_model_weights, 
                                  temporal_smoothing_weight, translation_weight, shared_focal, 
                                  flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_davis_gt_mask,
                                  fps, num_frames],
                          outputs=[scene, outmodel, outgallery])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, show_cam],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, show_cam],
                                    outputs=outmodel)
    demo.launch(share=args.share, server_name=server_name, server_port=server_port)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir is not None:
        tmp_path = args.output_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = args.model_name

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # Use the provided output_dir or create a temporary directory
    tmpdirname = args.output_dir if args.output_dir is not None else tempfile.mkdtemp(suffix='dust3r_gradio_demo')

    if not args.silent:
        print('Outputting stuff in', tmpdirname)

    if args.input_dir is not None:
        # Process images in the input directory with default parameters
        if os.path.isdir(args.input_dir):    # input_dir is a directory of images
            input_files = [os.path.join(args.input_dir, fname) for fname in sorted(os.listdir(args.input_dir))]
        else:   # input_dir is a video
            input_files = [args.input_dir]
        recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, args.device, args.silent, args.image_size)
        
        # Call the function with default parameters
        scene, outfile, imgs = recon_fun(
            filelist=input_files,
            schedule='linear',
            niter=300,
            min_conf_thr=1.1,
            as_pointcloud=True,
            mask_sky=False,
            clean_depth=True,
            transparent_cams=False,
            cam_size=0.05,
            show_cam=True,
            scenegraph_type='swinstride',
            winsize=5,
            refid=0,
            seq_name=args.seq_name,
            new_model_weights=args.weights,
            temporal_smoothing_weight=0.01,
            translation_weight='1.0',
            shared_focal=True,
            flow_loss_weight=0.01,
            flow_loss_start_iter=0.1,
            flow_loss_threshold=25,
            use_gt_mask=args.use_gt_davis_masks,
            fps=args.fps,
            num_frames=args.num_frames,
        )
        print(f"Processing completed. Output saved in {tmpdirname}/{args.seq_name}")
    else:
        # Launch Gradio demo
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent, args=args)
