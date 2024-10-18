import zipfile
import requests
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm

def download_and_extract_davis(url, extract_to='.'):
    local_zip_file = os.path.join(extract_to, 'davis.zip')

    # Download the dataset
    print("Downloading DAVIS dataset...")
    response = requests.get(url, stream=True)
    with open(local_zip_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print("Download complete.")

    # Extract the dataset
    print("Extracting DAVIS dataset...")
    with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

    # Remove the zip file
    os.remove(local_zip_file)
    print("Removed the zip file.")

def create_videos_from_images(image_root, video_root):
    if not os.path.exists(video_root):
        os.makedirs(video_root)

    # Iterate over each image set
    for image_set in os.listdir(image_root):
        image_set_path = os.path.join(image_root, image_set)
        if os.path.isdir(image_set_path):
            images = sorted(glob.glob(os.path.join(image_set_path, '*.jpg')))

            # Read the first image to get dimensions
            frame = cv2.imread(images[0])
            height, width, layers = frame.shape

            # Define the codec and create VideoWriter object
            video_file = os.path.join(video_root, f"{image_set}.mp4")
            out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

            for image_file in images:
                frame = cv2.imread(image_file)
                out.write(frame)

            out.release()
            print(f"Video for {image_set} created at {video_file}")


davis_url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'  # Update with the correct URL if needed
extract_path = '../data/davis'
video_output_path = '../data/davis_videos'

# make directories if they don't exist
if not os.path.exists(extract_path):
    os.makedirs(extract_path)

# Download and extract the dataset
download_and_extract_davis(davis_url, extract_path)

# Create videos from the image sets
create_videos_from_images(os.path.join(extract_path, 'DAVIS/JPEGImages/480p'), video_output_path)

# get the mask where fg masked to white
image_paths = glob.glob(r"../data/davis/DAVIS/JPEGImages/480p/*/*.jpg")
image_paths.sort()
print(f"Found {len(image_paths)} images.")
mask_paths = glob.glob(r"../data/davis/DAVIS/Annotations/480p/*/*.png")
mask_paths.sort()
print(f"Found {len(mask_paths)} masks.")

masked_dir_path = r"../data/davis/DAVIS/masked_images"
os.makedirs(masked_dir_path, exist_ok=True)

for img_path, mask_path in tqdm(zip(image_paths, mask_paths)):
    assert img_path.replace(".jpg", "").replace("JPEGImages", "") == mask_path.replace(".png", "").replace("Annotations", "")
    masked_path = img_path.replace(".jpg", ".jpg").replace('JPEGImages', 'masked_images')

    os.makedirs(os.path.dirname(masked_path), exist_ok=True)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Failed to load image {img_path}")
        continue
    if mask is None:
        print(f"Failed to load mask {mask_path}")
        continue

    mask = cv2.threshold(mask, 2, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    masked = np.stack([mask, mask, mask], axis=-1)
    success = cv2.imwrite(masked_path, masked)
    if success:
        # print(f"Saved {masked_path}")
        pass
    else:
        print(f"Failed to save {masked_path}")
