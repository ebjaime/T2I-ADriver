import os
import csv
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, BlipProcessor, BlipForConditionalGeneration, LlavaForConditionalGeneration

# Variable to choose the model
MODEL = "LLAVA"  # Change to "BLIP" to use the BLIP model
MAX_TOKENS = 75
USE_GPU = True
LONG_PROMPT = True
length_prompt = "in 75 or less words" if LONG_PROMPT else "in 25 or less words"

# Define the image directory and output CSV file
#image_dir = "T2I-Adriver/dataset/val"
image_dir = '/home/jovyan/work/dataset/shift/discrete/images/train/front/img/0b47-c059' # 0b47-c059, d8ac-a9fd
output_csv = "llava_prompts.csv" if LONG_PROMPT else "llava_prompts.csv"

#image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
image_files = []

# Define the directory to search for images

# Collect all image paths
for root, dirs, files in os.walk(image_dir):
    if "ipynb_" in root:
        continue
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')) and "semseg" not in file.lower() and "depth" not in file.lower():
            image_files.append(os.path.join(root, file))


# Load the appropriate model and processor based on the MODEL variable
if MODEL == "BLIP":
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
elif MODEL == "LLAVA":
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model.half()
else:
    raise ValueError("MODEL variable must be either 'BLIP' or 'LLAVA'")

# Move model to GPU if available
if USE_GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\t-> Running in ", device)
    model.to(device)

# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()

# Function to predict labels for an image using BLIP model
def predict_labels_blip(image_path):
    image = Image.open(image_path).convert("RGB")
    if USE_GPU:
        inputs = processor(images=image, return_tensors="pt").to(device)
    else:
        inputs = processor(images=image, return_tensors="pt")
    # Generate predictions with mixed precision
    with torch.cuda.amp.autocast():
        outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

# Function to predict labels for an image using LLAVA model
def predict_labels_llava(image_path):
    image = Image.open(image_path).convert("RGB")
    prompt = "USER: <image>\nThis image is taken from the first-person perspective of a car. Describe what you see in the scene "+length_prompt+" Take into account Environment Type, Weather Conditions, Pedestrian Presence, Traffic Lights, Traffic Signs, Time of the Day, Road Surface Condition, Road Markings, Vehicle Presence. ASSISTANT:"
    if USE_GPU:
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    else:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    # Generate predictions with mixed precision
    with torch.cuda.amp.autocast():
        generate_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
    full_caption = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Extract only the assistant's response
    assistant_response = full_caption.split("ASSISTANT:")[-1].strip()
    
    return assistant_response

# Select the appropriate function based on the model
if MODEL == "BLIP":
    predict_labels = predict_labels_blip
elif MODEL == "LLAVA":
    predict_labels = predict_labels_llava

# Iterate through images in the directory and label them
results = []

for filename in tqdm(sorted(image_files)[6:7], desc="Processing images"):
    #image_path = os.path.join(image_dir, filename)
    image_path = filename
    try:
        labels = predict_labels(image_path)
        results.append({"image": filename, "labels": labels})
    except torch.cuda.OutOfMemoryError:
        print(f"Skipping {filename} due to CUDA out of memory error.")
        torch.cuda.empty_cache()
        continue

# Save results to a CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Labels"])  # Write the header
    for result in results:
        writer.writerow([result["image"], result["labels"]])

print(f"Results saved to {output_csv}")
