# download waymo dataset

mkdir -p waymo
cd waymo
gsutil -m cp -r gs://waymo_open_dataset_v_1_4_2/individual_files/training/ .
wget --no-proxy https://download.europe.naverlabs.com/ComputerVision/DUSt3R/waymo_pairs.npz
cd ..