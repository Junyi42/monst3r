# download waymo dataset
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
gcloud auth login
mkdir -p waymo
cd waymo
gsutil -m cp -r gs://waymo_open_dataset_v_1_4_2/individual_files/training/ .
wget --no-proxy https://download.europe.naverlabs.com/ComputerVision/DUSt3R/waymo_pairs.npz
cd ..

# cd ..
# python datasets_preprocess/preprocess_waymo.py
# python datasets_preprocess/waymo_make_pairs.py