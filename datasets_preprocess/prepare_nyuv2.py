# %%
import h5py
import numpy as np
import os
from glob import glob
from PIL import Image

# Set the path to your dataset directory
dataset_dir = '../data/nyu-v2/val/official/'

# Get a list of all .h5 files in the dataset directory
file_paths = glob(os.path.join(dataset_dir, '*.h5'))

# Create output directories for images and depth data
output_image_dir = '../data/nyu-v2/val/nyu_images/'
output_depth_dir = '../data/nyu-v2/val/nyu_depths/'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_depth_dir, exist_ok=True)

for file_path in file_paths:
    with h5py.File(file_path, 'r') as h5file:
        # Read depth and rgb data
        depth_data = h5file['depth'][:]
        rgb_data = h5file['rgb'][:]
        
        # Convert rgb data from (3, H, W) to (H, W, 3)
        rgb_data = np.transpose(rgb_data, (1, 2, 0))
        
        # Ensure that rgb_data is of type uint8
        if rgb_data.dtype != np.uint8:
            rgb_data = rgb_data.astype(np.uint8)
        
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save the RGB image as PNG
        rgb_image = Image.fromarray(rgb_data)
        rgb_image.save(os.path.join(output_image_dir, f'{base_name}.png'))
        
        # Save the depth data as NPY file
        np.save(os.path.join(output_depth_dir, f'{base_name}.npy'), depth_data)
        
        print(f'Processed {base_name}')


# %%
import os
import numpy as np
from PIL import Image

# Paths
depth_npy_dir = '../data/nyu-v2/val/nyu_depths'
output_img_dir = '../data/nyu-v2/val/nyu_depth_imgs'

# Ensure the output directory exists
os.makedirs(output_img_dir, exist_ok=True)

# Iterate over all .npy files in the depth directory
for npy_file in os.listdir(depth_npy_dir):
    if npy_file.endswith('.npy'):
        # Load depth data from .npy file
        depth_path = os.path.join(depth_npy_dir, npy_file)
        depth_data = np.load(depth_path)
        
        # Normalize depth data to range [0, 255] for saving as an image
        depth_min = depth_data.min()
        depth_max = depth_data.max()
        depth_normalized = (depth_data - depth_min) / (depth_max - depth_min)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # Convert to an image
        depth_img = Image.fromarray(depth_uint8)
        
        # Save as PNG file
        img_name = os.path.splitext(npy_file)[0] + '.png'
        img_save_path = os.path.join(output_img_dir, img_name)
        depth_img.save(img_save_path)
        
        print(f'Saved {img_save_path}')

print("Conversion completed!")



