# Dataset Preparation for Evaluation

We provide scripts to download and prepare the datasets for evaluation. The datasets include: **Sintel**, **Bonn**, **KITTI**, **NYU-v2**, **TUM-dynamics**, **ScanNetv2**, and **DAVIS**.

> [!NOTE]
> The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.


## Download Datasets

### Sintel
To download and prepare the **Sintel** dataset, execute:
```bash
cd data
bash download_sintel.sh
cd ..

# (optional) generate the GT dynamic mask
cd ..
python datasets_preprocess/sintel_get_dynamics.py --threshold 0.1 --save_dir dynamic_label_perfect 
```

### Bonn
To download and prepare the **Bonn** dataset, execute:
```bash
cd data
bash download_bonn.sh
cd ..

# create the subset for video depth evaluation, following depthcrafter
cd datasets_preprocess
python prepare_bonn.py
cd ..
```

### KITTI
To download and prepare the **KITTI** dataset, execute:
```bash
cd data
bash download_kitti.sh
cd ..

# create the subset for video depth evaluation, following depthcrafter
cd datasets_preprocess
python prepare_kitti.py
cd ..
```

### NYU-v2
To download and prepare the **NYU-v2** dataset, execute:
```bash
cd data
bash download_nyuv2.sh
cd ..

# prepare the dataset for depth evaluation
cd datasets_preprocess
python prepare_nyuv2.py
cd ..
```

### TUM-dynamics
To download and prepare the **TUM-dynamics** dataset, execute:
```bash
cd data
bash download_tum.sh
cd ..

# prepare the dataset for pose evaluation
cd datasets_preprocess
python prepare_tum.py
cd ..
```

### ScanNet
To download and prepare the **ScanNet** dataset, execute:
```bash
cd data
bash download_scannetv2.sh
cd ..

# prepare the dataset for pose evaluation
cd datasets_preprocess
python prepare_scannet.py
cd ..
```

### DAVIS
To download and prepare the **DAVIS** dataset, execute:
```bash
cd data
python download_davis.py
cd ..
```

## Evaluation Script (Video Depth)

### Sintel

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_pose  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=sintel --output_dir="results/sintel_video_depth" --full_seq
```

The results will be saved in the `results/sintel_video_depth` folder. You could then run the corresponding code block in [depth_metric.ipynb](../depth_metric.ipynb) to evaluate the results.

### Bonn

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_pose  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=bonn --output_dir="results/bonn_video_depth"
```

The results will be saved in the `results/bonn_video_depth` folder. You could then run the corresponding code block in [depth_metric.ipynb](../depth_metric.ipynb) to evaluate the results.

### KITTI

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_pose  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=kitti --output_dir="results/kitti_video_depth"
```

The results will be saved in the `results/kitti_video_depth` folder. You could then run the corresponding code block in [depth_metric.ipynb](../depth_metric.ipynb) to evaluate the results.

## Evaluation Script (Camera Pose)

### Sintel

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_pose  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=sintel --output_dir="results/sintel_pose"
    # To use the ground truth dynamic mask, add: --use_gt_mask
```

The evaluation results will be saved in `results/sintel_pose/_error_log.txt`. 

### TUM-dynamics

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_pose  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=tum --output_dir="results/tum_pose"
```

The evaluation results will be saved in `results/tum_pose/_error_log.txt`.

### ScanNet

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_pose  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=scannet --output_dir="results/scannet_pose"
```

The evaluation results will be saved in `results/scannet_pose/_error_log.txt`.

## Evaluation Script (Single-Frame Depth)

### NYU-v2

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_depth  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=nyu --output_dir="results/nyuv2_depth"
```

The results will be saved in the `results/nyuv2_depth` folder. You could then run the corresponding code block in [depth_metric.ipynb](../depth_metric.ipynb) to evaluate the results.