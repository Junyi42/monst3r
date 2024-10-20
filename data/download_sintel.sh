# Download Sintel
mkdir -p sintel
cd sintel
# images
wget --no-proxy http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip
# depth & cameras
wget --no-proxy http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip
# flow
wget --no-proxy http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip
# unzip all
find . -name "*.zip" -exec unzip -o -q {} \;
# remove all zip files
find . -name "*.zip" -exec rm {} \;
cd ..

# # preprocess the dynamic labels
# conda activate monst3r
# cd ..
# python datasets_preprocess/sintel_get_dynamics.py --threshold 0.1 --save_dir dynamic_label_perfect
