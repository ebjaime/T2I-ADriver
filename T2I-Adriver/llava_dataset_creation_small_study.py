
import os
import csv
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Global Parameters
MODEL = "LLAVA"  # Change to "BLIP" to use the BLIP model
USE_GPU = True

# Parameters to test
MAX_TOKENS_LIST = [50, 100, 150]
LONG_PROMPT_LIST = [False, True]
NUM_BEAMS_LIST = [1, 3, 5]
TEMPERATURE_LIST = [1.0, 0.7, 0.5, 0.1]
TOP_K_LIST = [0, 50]  # 0 means no top-k sampling
TOP_P_LIST = [1.0, 0.9]  # 1.0 means no nucleus sampling

# Define the image directory
image_dir = "T2I-Adriver/dataset/train_20"
image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

# Loop through each combination of parameters
for MAX_TOKENS in MAX_TOKENS_LIST:
    for LONG_PROMPT in LONG_PROMPT_LIST:
        for NUM_BEAMS in NUM_BEAMS_LIST:
            for TEMPERATURE in TEMPERATURE_LIST:
                for TOP_K in TOP_K_LIST:
                    for TOP_P in TOP_P_LIST:
                        length_prompt = "in 100 or less words" if LONG_PROMPT else "in 25 or less words"
                        output_csv = f"T2I-Adriver/small_study_llava/train_{MAX_TOKENS}_tokens_{LONG_PROMPT}_long_prompt_{NUM_BEAMS}_beams_{TEMPERATURE}_temp_{TOP_K}_top_k_{TOP_P}_top_p.csv"

                        # Load the appropriate model and processor based on the MODEL variable
                        if MODEL == "LLAVA":
                            model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
                            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
                            model.half()
                        else:
                            raise ValueError("MODEL variable must be 'LLAVA'")

                        # Move model to GPU if available
                        if USE_GPU:
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            print("\t-> Running in ", device)
                            model.to(device)

                        # Enable mixed precision
                        scaler = torch.cuda.amp.GradScaler()

                        # Function to predict labels for an image using LLAVA model
                        def predict_labels_llava(image_path):
                            image = Image.open(image_path).convert("RGB")
                            prompt = f"USER: <image>\nThis image is taken from the first-person perspective of a car. Describe what you see in the scene {length_prompt}. ASSISTANT:"
                            if USE_GPU:
                                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
                            else:
                                inputs = processor(text=prompt, images=image, return_tensors="pt")
                            # Generate predictions with mixed precision
                            with torch.cuda.amp.autocast():
                                generate_ids = model.generate(**inputs,
                                                              max_new_tokens=MAX_TOKENS,
                                                              num_beams=NUM_BEAMS,
                                                              temperature=TEMPERATURE,
                                                              top_k=TOP_K,
                                                              top_p=TOP_P)
                            full_caption = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                            # Extract only the assistant's response
                            assistant_response = full_caption.split("ASSISTANT:")[-1].strip()

                            return assistant_response

                        # Iterate through images in the directory and label them
                        results = []
                        for filename in tqdm(image_files, desc=f"Processing images with {MAX_TOKENS} tokens, LONG_PROMPT={LONG_PROMPT}, {NUM_BEAMS} beams, {TEMPERATURE} temp, {TOP_K} top_k, {TOP_P} top_p"):
                            image_path = os.path.join(image_dir, filename)
                            try:
                                labels = predict_labels_llava(image_path)
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
