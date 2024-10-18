# nyu-v2
mkdir -p nyu_v2
cd nyu_v2
wget https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/val-000000.tar -O val-000000.tar
wget https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/val-000001.tar -O val-000001.tar
# unzip all
find . -name "*.tar" -exec tar -xvf {} \;
