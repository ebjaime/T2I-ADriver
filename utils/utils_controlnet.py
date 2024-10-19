from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from os.path import exists
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np

from controlnet_aux import OpenposeDetector, HEDdetector


from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, pipeline


width = 512
height = width
dim = (width, height)

def reshape(image, dim=dim):
    return image.resize(dim)


def create_image_two_conditionings(prompt,
                                        checkpoint_1, 
                                        checkpoint_2,
                                        image_1,
                                        image_2,
                                        pretrained_dm = "runwayml/stable-diffusion-v1-5",
                                        num_inference_steps=20,
                                        controlnet_conditioning_scale=[1.0,1.0],
                                        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                                        save_img=True,
                                        seed=42):
    
    if exists("img2/"+"_".join([checkpoint_1,checkpoint_2,image_1,image_2,prompt,str(controlnet_conditioning_scale[0]),str(controlnet_conditioning_scale[1])])+".jpg"):
        print("\t-> Already generated")
        return load_image("img2/2_conditionings/"+"_".join([checkpoint_1,checkpoint_2,image_1,image_2,prompt,str(controlnet_conditioning_scale[0]),str(controlnet_conditioning_scale[1])])+".jpg")
    # Conditionings
    controlnet = [
        ControlNetModel.from_pretrained(checkpoints[checkpoint_1][0], torch_dtype=torch.float16),
        ControlNetModel.from_pretrained(checkpoints[checkpoint_2][0], torch_dtype=torch.float16),
    ]

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_dm, controlnet=controlnet, torch_dtype=torch.float16, 
        # safety_checker = None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    url_1 = images_url[image_1]
    url_2 = images_url[image_2]
    images = [checkpoints[checkpoint_1][1](url_1), checkpoints[checkpoint_2][1](url_2)]

    image = pipe(
        prompt,
        images,
        num_inference_steps=num_inference_steps,
        generator=generator,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images[0]
    if save_img:
        image.save("img2/2_conditionings/"+"_".join([checkpoint_1,checkpoint_2,image_1,image_2,prompt,str(controlnet_conditioning_scale[0]),str(controlnet_conditioning_scale[1])])+".jpg")
    
    return image


def create_image_multiple_conditionings(prompt,
                                        checkpoints,
                                        _checkpoints,
                                        images,
                                        controlnet_conditioning_scale,
                                        pretrained_dm = "runwayml/stable-diffusion-v1-5",
                                        num_inference_steps=20,
                                        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                                        save_img=True,
                                        seed=42):

    assert len(images) == len(_checkpoints)
    assert len(controlnet_conditioning_scale) == len(images)
    
    title = str(len(images))+"_conditionings/"+"_".join([*_checkpoints,*images,prompt,str(controlnet_conditioning_scale)])
    
    if exists("img2/"+title+".jpg"):
        print("\t-> Already generated")
        return load_image("img2/"+title+".jpg")
    # Conditionings
    controlnet = [ControlNetModel.from_pretrained(checkpoints[checkpoint][0], torch_dtype=torch.float16) for checkpoint in _checkpoints]

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_dm, controlnet=controlnet, torch_dtype=torch.float16, 
        # safety_checker = None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    urls = images
    images = [checkpoints[checkpoint][1](url) for checkpoint,url in zip(_checkpoints,urls)]

    image = pipe(
        prompt,
        images,
        num_inference_steps=num_inference_steps,
        generator=generator,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images[0]
    if save_img:
        image.save("img2/"+title+".jpg")
    
    return image

# Preprocessing functions

def to_canny(image_url, threshold=[100,200]):

    canny_image = load_image(
        image_url
    )
    canny_image=reshape(canny_image)
    canny_image = np.array(canny_image)

    low_threshold = threshold[0]
    high_threshold = threshold[1]

    canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)

    canny_image = Image.fromarray(canny_image)
    return canny_image


def to_openpose(image_url):
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    openpose_image = load_image(
       image_url 
    )
    openpose_image = reshape(openpose_image)
    openpose_image = openpose(openpose_image)
    return openpose_image


def to_ip2p(image_url):
    ip2p_image = load_image(
        image_url
    )

    return reshape(ip2p_image)


def to_depth(image_url):
    depth_image = load_image(
        image_url
    )
    depth_image = reshape(depth_image)
    depth_estimator = pipeline('depth-estimation')
    depth_image = depth_estimator(depth_image)['depth']
    depth_image = np.array(depth_image)
    depth_image = depth_image[:, :, None]
    depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
    control_image = Image.fromarray(depth_image).resize(dim)
    return control_image

ada_palette = np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])

def to_segmentation(image_url):
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

    image = load_image(
      image_url
    )
    image = reshape(image)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
        seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        for label, color in enumerate(ada_palette):
            color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    control_image = Image.fromarray(color_seg)
    segmentation_image = control_image

    return segmentation_image


def to_hededge(image_url):
    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    hed_image = load_image(image_url)
    hed_image = reshape(hed_image)
    hed_image = hed(hed_image)
    return hed_image