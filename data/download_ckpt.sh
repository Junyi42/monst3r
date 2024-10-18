mkdir -p ../checkpoints/
gdown --fuzzy https://drive.google.com/file/d/1Z1jO_JmfZj0z3bgMvCwqfUhyZ1bIbc9E/view?usp=sharing -O ../checkpoints/
# THE original dust3r ckpt
# wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P ../checkpoints/

# sea-raft ckpt
cd ../third_party/RAFT
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
rm models.zip
gdown --fuzzy https://drive.google.com/file/d/1a0C5FTdhjM4rKrfXiGhec7eq2YM141lu/view?usp=drive_link -O models/
cd ../../data

# sam2 ckpt
cd ../third_party/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/
cd ../../data