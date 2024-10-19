import os
import random
import shutil

# Source directory containing multiple subdirectories with images
source_dir = "/home/jovyan/work/dataset/shift/discrete/images/train/front/img"
source_dir_left = "/home/jovyan/work/dataset/shift/discrete/images/train/left_45/img"
source_dir_right = "/home/jovyan/work/dataset/shift/discrete/images/train/right_45/img"


# Target directory to copy the sampled images
target_dir = "T2I-Adriver/dataset/val"

NUM_IMGS = 3

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Clear the target directory
for filename in os.listdir(target_dir):
    file_path = os.path.join(target_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

# List all subdirectories in the source directory
subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

# Randomly sample 20 subdirectories (or as many as available if less than 20)
sampled_subdirs = random.sample(subdirs, min(NUM_IMGS, len(subdirs)))

# List to keep track of selected images
selected_images = []

# Iterate over the sampled subdirectories and select one random image from each
for subdir in sampled_subdirs:
    subdir_path = os.path.join(source_dir, subdir)
    images = [f for f in os.listdir(subdir_path) if f.endswith((".png", ".jpg", ".jpeg"))]
    
    if images:
        selected_image = random.choice(images)
        selected_images.append((subdir, selected_image))

# Copy the selected images to the target directory
for subdir, image in selected_images:
    source_image_path = os.path.join(source_dir, subdir, image)
    target_image_path = os.path.join(target_dir, image)
    
    # If a file with the same name exists, add a suffix
    if os.path.exists(target_image_path):
        base, ext = os.path.splitext(image)
        new_image_name = f"{base}_2{ext}"
        target_image_path = os.path.join(target_dir, new_image_name)
        
    shutil.copy2(source_image_path, target_image_path)
    print(f"Copied {source_image_path} to {target_image_path}")

print("Sampling and copying of images completed.")
