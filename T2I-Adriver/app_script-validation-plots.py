import argparse
import copy
import os
import shutil
import torch
from torch import autocast
from pytorch_lightning import seed_everything
from basicsr.utils import tensor2img
from ldm.inference_base import diffusion_inference, get_adapters, get_sd_models, get_sd_models_coadapter
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_cond_model, get_adapter_feature
import pyiqa
from pyiqa.utils import img2tensor
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from itertools import combinations, product
import csv
import pandas as pd
from ldm.modules.encoders.adapter import CoAdapterFuser
from ldm.util import instantiate_from_config, read_state_dict

def compute_metrics(reference, generated):
    mse_metric = torch.nn.functional.mse_loss
    perceptual_metric = pyiqa.create_metric('lpips')

    to_tensor = transforms.ToTensor()

    if not isinstance(generated, torch.Tensor):
        generated = to_tensor(generated).unsqueeze(0)#img2tensor(generated).unsqueeze(0)
    if not isinstance(reference, torch.Tensor):
        reference = to_tensor(reference).unsqueeze(0)#img2tensor(generated).unsqueeze(0)

    reference_size = reference.shape[2:]
    resize_transform = transforms.Resize(reference_size)
    generated = resize_transform(generated)

    mse = mse_metric(reference, generated).item()
    perceptual_loss = perceptual_metric(reference, generated).item()

    return mse, perceptual_loss


# Define the main function to run the model
def run(opt, IMAGE, PROMPT, CONDITIONINGS, CONDITIONING_WEIGHTS, name_ckpt="last.ckpt", diff_color_img=None, diff_depth_img=None, diff_seg_img=None, scale=7.5):
    torch.set_grad_enabled(False)

    global_opt = opt
    adapters = {}
    cond_models = {}
    torch.cuda.empty_cache()

    with torch.inference_mode(), autocast('cuda'):

        opt = copy.deepcopy(global_opt)
        opt.prompt = PROMPT if type(PROMPT) != float else ""
        opt.scale = scale
        opt.n_samples = 1
        opt.seed = 42
        opt.steps = 50
        opt.resize_short_edge = 512

        im = IMAGE
        conds = {}
        activated_conds = []
        for cond_name, cond_weight in zip(CONDITIONINGS, CONDITIONING_WEIGHTS):
            activated_conds.append(cond_name)
            if cond_name in adapters:
                adapters[cond_name]['model'] = adapters[cond_name]['model'].to(opt.device)
            else:
                adapters[cond_name] = get_adapters(opt, getattr(ExtraCondition, cond_name))
            adapters[cond_name]['cond_weight'] = cond_weight
            process_cond_module = getattr(api, f'get_cond_{cond_name}')
            if cond_name not in cond_models:   
                cond_models[cond_name] = get_cond_model(opt, getattr(ExtraCondition, cond_name))
            
            if cond_name == "depth" and diff_depth_img is not None:
                conds[cond_name] = (diff_depth_img.unsqueeze(0)).to(global_opt.device)
                
            elif cond_name == "seg" and diff_seg_img is not None:
                conds[cond_name] = (diff_seg_img.unsqueeze(0)).to(global_opt.device)
                
            elif cond_name == "color" and diff_color_img is not None: 
                conds[cond_name] = (img2tensor(diff_color_img).unsqueeze(0) / 255.).to(global_opt.device) #(process_cond_module(opt, diff_color_img, 'image', cond_models[cond_name]))
            else:
                conds[cond_name] = (process_cond_module(opt, im, 'image', cond_models[cond_name]))
            
        if len(activated_conds) != 3:
            print("-> Less or more than 3 conditionings. To use our trained models one must choose 3 between sketch, seg, depth and color... ")
            print("\t -> Using vanilla stable diffusion with coadapter")
            sd_model, sampler = get_sd_models(global_opt)
        else:
            print("-> Load model ", f'T2I-Adriver/logs/coadapter-{activated_conds[0]}-{activated_conds[1]}-{activated_conds[2]}-big/checkpoints/{name_ckpt}')
            activated_conds.sort()
            global_opt.coadapter_ckpt = f'T2I-Adriver/logs/coadapter-{activated_conds[0]}-{activated_conds[1]}-{activated_conds[2]}-big/checkpoints/{name_ckpt}'
            sd_model, sampler = get_sd_models_coadapter(global_opt)

        # Option 1
        batch = {
            "jpg": (img2tensor(im).unsqueeze(0) / 255.).to(global_opt.device),
            "txt": [opt.prompt],
            "style": transforms.Resize((224, 224))(img2tensor(im)).unsqueeze(0) / 255. # NOT USED
        }
        print("jpg RANGE", torch.min(batch["jpg"]), torch.max(batch["jpg"]))
        for idx, cond_name in enumerate(activated_conds):
            batch[cond_name] = conds[cond_name]
            
        x, c,  = sd_model.get_input(batch, sd_model.first_stage_key)
        sd_model.data_preparation_on_gpu(batch)
        
        features = sd_model.get_adapter_features(batch, activated_conds)
        
        coadapter_fuser = sd_model.coadapter_fuser.to(global_opt.device)
        adapter_features, append_to_context = coadapter_fuser(features)
        
        # Option 2
        # features = dict()
        # for idx, cond_name in enumerate(activated_conds):
        #     cur_feats = adapters[cond_name]['model'](conds[idx])
        #     if isinstance(cur_feats, list):
        #         for i in range(len(cur_feats)):
        #             cur_feats[i] *= adapters[cond_name]['cond_weight']
        #     else:
        #         cur_feats *= adapters[cond_name]['cond_weight']
                
        #     features[cond_name] = cur_feats
                        
        # coadapter_fuser = sd_model.coadapter_fuser.to(global_opt.device)
        # adapter_features, append_to_context = coadapter_fuser(features)

        # Option 3 (default)
        # adapter_features, append_to_context = get_adapter_feature(
        #     conds, [adapters[cond_name] for cond_name in activated_conds], [cond_name for cond_name in activated_conds])
        
        print("-> Using sd_model: ", sd_model.__class__.__name__) 
        print("-> Using sampler: ", sampler.__class__.__name__) 
        output_conds = []
        for cond in list(conds.values()):
            output_conds.append(tensor2img(cond, rgb2bgr=False))

        ims = []
        metrics = []
        seed_everything(opt.seed)
        
        for _ in range(opt.n_samples):
            #with sd_model.ema_scope(), autocast('cuda'):
            result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
            generated_img = tensor2img(result, rgb2bgr=False)
            ims.append(generated_img)

            reference_img = im
            mse, perceptual_loss = compute_metrics(reference_img, generated_img)
            metrics.append((mse, perceptual_loss))

        torch.cuda.empty_cache()
        return metrics, ims, output_conds

# Configurations
parser = argparse.ArgumentParser()
parser.add_argument('--sd_ckpt', type=str, default='T2I-Adriver/models/sd-v1-4.ckpt', help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported')
parser.add_argument('--vae_ckpt', type=str, default=None, help='vae checkpoint, anime SD models usually have separate vae ckpt that need to be loaded')
opt = parser.parse_args([])
opt.config = 'T2I-Adriver/configs/stable-diffusion/sd-v1-inference.yaml'
supported_cond = ["depth", "seg", "sketch"]
for cond_name in supported_cond:
    setattr(opt, f'{cond_name}_adapter_ckpt', f'T2I-Adriver/models/t2iadapter_{cond_name}_sd14v1.pth')
opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
opt.max_resolution = 512 * 512
opt.sampler = 'ddim'
opt.cond_weight = 1.0
opt.C = 4
opt.f = 8
opt.style_cond_tau = 1.0
opt.cond_tau = 1.0
#opt.scale = 7.5
opt.neg_prompt = "blurry, distorted face, extra limbs, low resolution, bad anatomy, grainy, oversaturated colors, pixelated, out of focus, wrong proportions, cropped, text, watermark"
        

# Input variables
train_dir = 'T2I-Adriver/dataset/other_models/templates'
IMAGE_PATHS = [os.path.join(train_dir, img) for img in os.listdir(train_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
MAX_NUM_GENERATED = 20
CONDITIONINGS_LIST = list(combinations(supported_cond, 3))
CONDITIONING_WEIGHTS_LIST = [0.5, 1.0]

# Load the prompts from train.csv
train_csv_path = 'T2I-Adriver/dataset/other_models/llava_long_prompts.csv'
prompts_df = pd.read_csv(train_csv_path)
prompts_dict = dict(zip(prompts_df['Image'], prompts_df['Labels']))

# Create output directory
output_dir = 'T2I-Adriver/dataset/other_models_new'
#if os.path.exists(output_dir):
#    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# CSV file setup
csv_file_path = os.path.join(output_dir, 'metrics.csv')
csv_columns = ["Checkpoint", 'Image', 'Conditionings', 'Weights', 'MSE', 'Perceptual_Loss']

# Initialize the CSV file with headers
#with open(csv_file_path, mode='w', newline='') as csv_file:
#    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
#    writer.writeheader()

# Define the function to save images and metrics
def save_images(image_path, output_dir, conditionings, weights, name_ckpt="last.ckpt", diff_color_img_path=None, diff_depth_img_path=None, diff_seg_img_path=None, scale=7.5):
    # Get the prompt for the image
    image_name = os.path.basename(image_path)
    prompt = "" #prompts_dict.get(image_name, "")  # Fallback to a default prompt if not found
    print("PROMPT", prompt)
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    
    # Run the model with the specific prompt
    if diff_color_img_path is not None:
        diff_color_img = cv2.imread(diff_color_img_path)
        #diff_color_img = cv2.cvtColor(diff_color_img, cv2.COLOR_BGR2RGB)
        #diff_color_img = cv2.resize(diff_color_img, (512, 512))
        metrics, ims, output_conds = run(opt, image, prompt, conditionings, weights, name_ckpt, diff_color_img=diff_color_img, scale=scale)
    elif diff_depth_img_path is not None and diff_seg_img_path is not None :
        diff_depth_img = diff_depth_img_path #cv2.imread(diff_depth_img_path)
        #diff_depth_img = cv2.resize(diff_depth_img, (512, 512))
        diff_seg_img = diff_seg_img_path #cv2.imread(diff_seg_img_path)
        #diff_seg_img = cv2.resize(diff_seg_img, (512, 512))
        metrics, ims, output_conds = run(opt, image, prompt, conditionings, weights, name_ckpt, diff_depth_img=diff_depth_img, diff_seg_img=diff_seg_img, scale=scale)
    else:
        metrics, ims, output_conds = run(opt, image, prompt, conditionings, weights, name_ckpt, diff_color_img=None, scale=scale)

    # Create subdirectory based on conditionings and weights
    name_ckpt = name_ckpt[:-5]
    conditioning_str = '_'.join(conditionings)
    weights_str = '_'.join(map(str, weights))
    if diff_depth_img_path is not None and diff_seg_img_path is not None:
        #sub_dir_name = f"{conditionings[0]}_{conditionings[1]}_{conditionings[2]}_{name_ckpt}"
        sub_dir_name = f"{conditionings[0]}_{conditionings[1]}_{conditionings[2]}_ground_truth_{name_ckpt}_{scale}"
    else:
        sub_dir_name = f"{conditionings[0]}_{conditionings[1]}_{conditionings[2]}_{name_ckpt}_{scale}"
    sub_dir_path = os.path.join(output_dir, sub_dir_name)
    os.makedirs(sub_dir_path, exist_ok=True)

    # Save the original image
    original_img_pil = Image.fromarray(image)
    #original_img_pil.save(os.path.join(sub_dir_path, f'{image_name}_original.png'))

    # Save the generated images and conditioning images
    for idx, img in enumerate(ims):
        img_pil = Image.fromarray(img)

        # Define the base file name
        if diff_color_img_path is not None:
            base_name = f'{conditionings[0]}_{conditionings[1]}_{conditionings[2]}_{name_ckpt}_{diff_color_img_path.split("/")[-1][:-4]}_{scale}_1.png'
        elif diff_depth_img_path is not None and diff_seg_img_path is not None:
            #base_name = f'{conditionings[0]}_{conditionings[1]}_{conditionings[2]}_{name_ckpt}_1.png'
            base_name = f'{conditionings[0]}_{conditionings[1]}_{conditionings[2]}_ground_truth_{name_ckpt}_{scale}_7.png' 
        else:
            base_name = f'{conditionings[0]}_{conditionings[1]}_{conditionings[2]}_{name_ckpt}_{scale}_7.png'
        file_path = os.path.join(sub_dir_path, base_name)

        # Check if the file exists, and if so, append "_1", "_2", etc.
        #if os.path.exists(file_path):
        #    file_name, file_ext = os.path.splitext(base_name)
        #    counter = 2
        #    file_name = file_name[:-2]
        #    while os.path.exists(os.path.join(sub_dir_path, f'{file_name}_{counter}{file_ext}')):
        #        counter += 1
        #    file_path = os.path.join(sub_dir_path, f'{file_name}_{counter}{file_ext}')

        # Save the image with the final unique name
        #img_pil.save(file_path)

    #for idx, cond_name in enumerate(conditionings):
    #    cond_pil = Image.fromarray(output_conds[idx])
    #    cond_pil.save(os.path.join(sub_dir_path, f'{image_name}_{cond_name}_{name_ckpt}.png'))

    # Save metrics to CSV
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        for metric in metrics:
            row = {
                'Checkpoint': name_ckpt,
                'Image': image_name,
                'Conditionings': conditioning_str,
                'Weights': weights_str,
                'MSE': metric[0],
                'Perceptual_Loss': metric[1]
            }
            writer.writerow(row)

    print(f"Images and metrics for {image_name} with {conditionings} and weights {weights} saved to {sub_dir_path}")

# When using ground truth for depth and semseg
def load_diff_depth_semseg(filepath_semseg, filepath_depth):
    import torchvision.transforms as transforms
    from PIL import Image
    from shift_dev.utils.backend import FileBackend
    from shift_dev.utils.load import im_decode
    
    backend=FileBackend()

    # SEMSEG
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    im_bytes = backend.get(filepath_semseg)
    image = im_decode(im_bytes)[..., 0]
    diff_seg_img = torch.as_tensor(image, dtype=torch.int64).unsqueeze(0)
    seg = diff_seg_img.squeeze(0)
    seg_np = (seg.detach().cpu().numpy()).astype('uint8')
    seg_np_rgb = np.repeat(seg_np[:, :, np.newaxis], 3, axis=2)  # Replicate across RGB channels
    seg_pil = Image.fromarray(seg_np_rgb)
    seg_tensor = transform(seg_pil)
    
    #DEPTH
    im_bytes = backend.get(filepath_depth)
    max_depth = 1000.0
    image = im_decode(im_bytes)
    if image.shape[2] > 3:  # pragma: no cover
        image = image[:, :, :3]
    image = image.astype(np.float32)
    # Convert to depth
    depth = image[:, :, 2] * 256 * 256 + image[:, :, 1] * 256 + image[:, :, 0]
    depth /= 16777216.0  # 256 ** 3
    depth = torch.as_tensor(
        np.ascontiguousarray(depth * max_depth),
        dtype=torch.float32,
    ).unsqueeze(0)
    depth = depth.squeeze(0)
    depth_np = (depth.detach().cpu().numpy()).astype('uint8')
    depth_np_rgb = np.repeat(depth_np[:, :, np.newaxis], 3, axis=2)  # Replicate across RGB channels
    depth_pil = Image.fromarray(depth_np_rgb)
    depth_tensor = transform(depth_pil)
    
    return seg_tensor, depth_tensor
    
    
checkpoint_dir = f'T2I-Adriver/logs/coadapter-{supported_cond[0]}-{supported_cond[1]}-{supported_cond[2]}-big/checkpoints/'

# List all checkpoints in the directory
checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')])
# List all color templates in the directory
color_templates = sorted([f for f in os.listdir("T2I-Adriver/dataset/other_models/color_templates") if f.endswith('.png')])


# Generate and save images for each combination of conditionings and weights
first = checkpoints[0]
middle = checkpoints[len(checkpoints)//2]
last = checkpoints[-1]
print(checkpoints)
for checkpoint in checkpoints[14:]:
    for conditionings in CONDITIONINGS_LIST:
        for weights in [[1,1,1]]:#product(CONDITIONING_WEIGHTS_LIST, repeat=3):
            for image_path in IMAGE_PATHS:
                for scale in [7.5]:
                    save_images(image_path, output_dir, conditionings, weights, checkpoint, diff_color_img_path=None, scale=scale)
                    #diff_depth_img_path = "/".join(image_path.split("/")[:-2]+["ground_truth_templates", image_path.split("/")[-1].replace("img", "depth").replace("jpg", "png")])
                    #diff_seg_img_path = "/".join(image_path.split("/")[:-2]+["ground_truth_templates", image_path.split("/")[-1].replace("img", "semseg").replace("jpg", "png")])
                    #diff_seg_img, diff_depth_img = load_diff_depth_semseg(diff_seg_img_path, diff_depth_img_path)

                    #save_images(image_path, output_dir, conditionings, weights, checkpoint, diff_depth_img_path=diff_depth_img, diff_seg_img_path=diff_seg_img, scale=scale)
                
                    #for color_template in color_templates: 
                    #    save_images(image_path, output_dir, conditionings, weights, checkpoint, diff_color_img_path="T2I-Adriver/dataset/other_models/color_templates/"+color_template, scale=scale)
                
# for conditionings in CONDITIONINGS_LIST:
#     for weights in [[1,1,1]]:#product(CONDITIONING_WEIGHTS_LIST, repeat=3):
#         for image_path in IMAGE_PATHS[:min(MAX_NUM_GENERATED, len(IMAGE_PATHS))]:
#             save_images(image_path, output_dir, conditionings, weights)

    # Example to generate seg images
    # seg_model = api.get_cond_model(opt, ExtraCondition.seg)
    # opt.prompt = PROMPT
    # opt.neg_prompt = ""
    # opt.scale = 7.5
    # opt.n_samples = 1
    # opt.seed = 42
    # opt.steps = 50
    # opt.resize_short_edge = 512
    # opt.cond_tau = 1.0
    # img = api.get_cond_seg(opt, image_path, "image", seg_model)
    # img_pil = tensor2img(img, rgb2bgr=False)
    # img_pil = Image.fromarray(img_pil)
    # img_pil.save(os.path.join(output_dir, f'test_generated_{i}.png'))
    # i+=1