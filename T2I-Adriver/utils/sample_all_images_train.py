import os
import csv
from tqdm import tqdm

# Define the directory to search for images
image_dir = '/home/jovyan/work/dataset/shift/discrete/images/train/'
image_dir_val = '/home/jovyan/work/dataset/shift/discrete/images/val/'

# Define the output CSV file path
output_csv = '/home/jovyan/work/code/jaime/training/T2I-Adriver/dataset/train_all_no_prompts.csv'
output_csv_val = '/home/jovyan/work/code/jaime/training/T2I-Adriver/dataset/val_all_no_prompts.csv'

# Collect all image paths
image_paths = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')) and "semseg" not in file.lower() and "depth" not in file.lower():
            image_paths.append(os.path.join(root, file))

image_paths_val = []
for root, dirs, files in os.walk(image_dir_val):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')) and "semseg" not in file.lower() and "depth" not in file.lower():
            image_paths_val.append(os.path.join(root, file))

# Write the image paths and empty labels to the CSV
with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Image', 'Labels'])  # Write the header
    for image_path in tqdm(image_paths, desc="Processing images"):
        writer.writerow([image_path, ""])  # Write the image path and empty label

with open(output_csv_val, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Image', 'Labels'])  # Write the header
    for image_path in tqdm(image_paths_val, desc="Processing images"):
        writer.writerow([image_path, ""])  # Write the image path and empty label

        
print(f"CSV file created: {output_csv}, {output_csv_val}")