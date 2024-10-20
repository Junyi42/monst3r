
# Dataset Preparation for Training

We provide scripts to prepare datasets for training, including **PointOdyssey**, **TartanAir**, **Spring**, and **Waymo**. For evaluation, we also provide a script for preparing the **Sintel** dataset.  

> [!NOTE]
> The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

## Download Pre-Trained Models
To download the pre-trained models, run the following commands:
```bash
cd data
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P ../checkpoints/
cd ..
```

## Dataset Setup

### PointOdyssey
To download and prepare the **PointOdyssey** dataset, execute:
```bash
cd data
bash download_pointodyssey.sh
cd ..
```

### TartanAir
To download and prepare the **TartanAir** dataset, execute:
```bash
cd data
bash download_tartanair.sh
cd ..
```

### Spring
To download and prepare the **Spring** dataset, execute:
```bash
cd data
bash download_spring.sh
cd ..
```

### Waymo
To download and prepare the **Waymo** dataset, follow these steps:

1. Set up Google Cloud SDK (if you haven't done so already):
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
gcloud auth login
```

2. Download the Waymo dataset:
```bash
cd data
bash download_waymo.sh
cd ..
```

3. Preprocess the dataset and create training pairs:
```bash
python datasets_preprocess/preprocess_waymo.py
python datasets_preprocess/waymo_make_pairs.py
```

## Sintel (Evaluation)
To download and prepare the **Sintel** dataset for evaluation, execute:
```bash
cd data
bash download_sintel.sh
cd ..
```
