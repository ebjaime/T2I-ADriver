# demo inspired by https://huggingface.co/spaces/lambdalabs/image-mixer-demo
import argparse
import copy
import gradio as gr
import torch
from functools import partial
from itertools import chain
from torch import autocast
from pytorch_lightning import seed_everything

from basicsr.utils import tensor2img
from ldm.inference_base import DEFAULT_NEGATIVE_PROMPT, diffusion_inference, get_adapters, get_sd_models
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_adapter_feature, get_cond_model

import pyiqa
from pyiqa import create_metric
from pyiqa.utils import img2tensor
from torchvision import transforms

def compute_metrics(reference, generated):
    # Create metric objects
    mse_metric = torch.nn.functional.mse_loss
    perceptual_metric = pyiqa.create_metric('lpips')  # You can use other perceptual metrics if you prefer

    # Convert the generated image to a tensor if it's not already
    if not isinstance(generated, torch.Tensor):
        generated = img2tensor(generated).unsqueeze(0)  # Add batch dimension

    # Ensure the reference is also a tensor and has the correct shape
    if not isinstance(reference, torch.Tensor):
        reference = img2tensor(reference).unsqueeze(0)  # Add batch dimension

    # Resize the reference image to match the size of the generated image
    reference_size = reference.shape[2:]  # Get the height and width of the generated image
    resize_transform = transforms.Resize(reference_size)
    generated = resize_transform(generated)

    # Compute MSE
    mse = mse_metric(reference, generated).item()

    # Compute Perceptual Loss
    perceptual_loss = perceptual_metric(reference, generated).item()

    # TODO: Compute FID with batches of images

    return mse, perceptual_loss

torch.set_grad_enabled(False)

supported_cond = ["color", "depth", "seg", "sketch"] #[e.name for e in ExtraCondition]

# config
parser = argparse.ArgumentParser()
parser.add_argument(
    '--sd_ckpt',
    type=str,
    default='T2I-Adriver/models/sd-v1-4.ckpt',
    help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported',
)
parser.add_argument(
    '--vae_ckpt',
    type=str,
    default=None,
    help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
)
global_opt = parser.parse_args()
global_opt.config =  'T2I-Adriver/configs/stable-diffusion/sd-v1-inference.yaml' #'T2I-Adriver/configs/pl_train/coadapter-v1-train.yaml'
for cond_name in supported_cond:
    setattr(global_opt, f'{cond_name}_adapter_ckpt', f'T2I-Adriver/models/t2iadapter_{cond_name}_sd14v1.pth')
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
global_opt.max_resolution = 512 * 512
global_opt.sampler = 'ddim'
global_opt.cond_weight = 1.0
global_opt.C = 4
global_opt.f = 8
global_opt.style_cond_tau = 0.5

# stable-diffusion model
# Will be called later
# sd_model, sampler = get_sd_models(global_opt)
# adapters and models to processing condition inputs
adapters = {}
cond_models = {}

torch.cuda.empty_cache()


def run(*args):
    with torch.inference_mode(), \
            autocast('cuda'):

        inps = []
        for i in range(0, len(args) - 8, len(supported_cond)):
            inps.append(args[i:i + len(supported_cond)])

        opt = copy.deepcopy(global_opt)
        opt.prompt, opt.neg_prompt, opt.scale, opt.n_samples, opt.seed, opt.steps, opt.resize_short_edge, opt.cond_tau \
            = args[-8:]

        im = None
        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            if b == "Nothing":
                continue
            if im is None:
                im = im1 
            else:
                assert im.shape == im1.shape
                assert (im - im1).sum()==0
        
        assert im is not None
        
        conds = []
        activated_conds = []
        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            cond_name = supported_cond[idx]
            if b == 'Nothing':
                if cond_name in adapters:
                    adapters[cond_name]['model'] = adapters[cond_name]['model'].cpu()
            else:
                activated_conds.append(cond_name)
                if cond_name in adapters:
                    adapters[cond_name]['model'] = adapters[cond_name]['model'].to(opt.device)
                else:
                    adapters[cond_name] = get_adapters(opt, getattr(ExtraCondition, cond_name))
                adapters[cond_name]['cond_weight'] = cond_weight

                process_cond_module = getattr(api, f'get_cond_{cond_name}')

                if b == 'Image':
                    if cond_name not in cond_models:
                        cond_models[cond_name] = get_cond_model(opt, getattr(ExtraCondition, cond_name))
                    conds.append(process_cond_module(opt, im1, 'image', cond_models[cond_name]))
                else:
                    conds.append(process_cond_module(opt, im2, cond_name, None))
        
        # Our model is currently only trained for 3 conditions
        if len(activated_conds) != 3:
            print("-> Less than 3 conditionings. To use our trained models one must choose 3 between sketch, seg, canny, depth and color... ")
            print("\t -> Using vainilla stable diffusion with coadapter")
        else:
            activated_conds.sort()
            global_opt.sd_ckpt = f'T2I-Adriver/logs/coadapter-{activated_conds[0]}-{activated_conds[1]}-{activated_conds[2]}/checkpoints/last.ckpt'

        sd_model, sampler = get_sd_models(global_opt)

        adapter_features, append_to_context = get_adapter_feature(
            conds, [adapters[cond_name] for cond_name in activated_conds])

        output_conds = []
        for cond in conds:
            output_conds.append(tensor2img(cond, rgb2bgr=False))

        ims = []
        metrics = []
        seed_everything(opt.seed)
        for _ in range(opt.n_samples):
            with sd_model.ema_scope(), autocast('cuda'):
                result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
            generated_img = tensor2img(result, rgb2bgr=False)
            ims.append(tensor2img(result, rgb2bgr=False))
            
            reference_img = im  # Load or specify your reference image here
            mse, perceptual_loss = compute_metrics(reference_img, generated_img)
            metrics.append((mse, perceptual_loss))

        # Clear GPU memory cache so less likely to OOM
        torch.cuda.empty_cache()
        return metrics, ims, output_conds


def change_visible(im1, im2, val):
    outputs = {}
    if val == "Image":
        outputs[im1] = gr.update(visible=True)
        outputs[im2] = gr.update(visible=False)
    elif val == "Nothing":
        outputs[im1] = gr.update(visible=False)
        outputs[im2] = gr.update(visible=False)
    else:
        outputs[im1] = gr.update(visible=False)
        outputs[im2] = gr.update(visible=True)
    return outputs

def warning_generic_model():
    gr.Warning('Less than 3 conditionings. To use our trained models one must choose 3 between sketch, seg, canny, depth and color... \n Using vainilla stable diffusion with coadapter')
    return "Less than 3 conditionings. To use our trained models one must choose 3 between sketch, seg, canny, depth and color... \n Using vainilla stable diffusion with coadapter"


DESCRIPTION = '''# Composable T2I-Adriver

This gradio demo is for a simple experience of composable T2I-Adriver:
'''
with gr.Blocks(title="T2I-Adriver", css=".gr-box {border-color: #8136e2}") as demo:
    gr.Markdown(DESCRIPTION)

    btns = []
    ims1 = []
    ims2 = []
    cond_weights = []

    with gr.Row():
        for cond_name in supported_cond:
            with gr.Box():
                with gr.Column():
                    btn1 = gr.Radio(
                        choices=["Image", cond_name, "Nothing"],
                        label=f"Input type for {cond_name}",
                        interactive=True,
                        value="Nothing",
                    )
                    im1 = gr.Image(source='upload', label="Image", interactive=True, visible=False, type="numpy")
                    im2 = gr.Image(source='upload', label=cond_name, interactive=True, visible=False, type="numpy")
                    cond_weight = gr.Slider(
                        label="Condition weight", minimum=0, maximum=5, step=0.05, value=1, interactive=True)

                    fn = partial(change_visible, im1, im2)
                    btn1.change(fn=fn, inputs=[btn1], outputs=[im1, im2], queue=False)

                    btns.append(btn1)
                    ims1.append(im1)
                    ims2.append(im2)
                    cond_weights.append(cond_weight)

    with gr.Column():
        prompt = gr.Textbox(label="Prompt")
        with gr.Accordion('Advanced options', open=False):
            neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
            scale = gr.Slider(label="Guidance Scale (Classifier free guidance)", value=7.5, minimum=1, maximum=20, step=0.1)
            n_samples = gr.Slider(label="Num samples", value=1, minimum=1, maximum=8, step=1)
            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1)
            steps = gr.Slider(label="Steps", value=50, minimum=10, maximum=100, step=1)
            resize_short_edge = gr.Slider(label="Image resolution", value=512, minimum=320, maximum=1024, step=1)
            cond_tau = gr.Slider(
                label="timestamp parameter that determines until which step the adapter is applied",
                value=1.0,
                minimum=0.1,
                maximum=1.0,
                step=0.05)

    with gr.Row():
        submit = gr.Button("Generate")
    metrics_output = gr.Dataframe(headers=["MSE", "LPIPS"])
    output = gr.Gallery().style(grid=2, height='auto')
    cond = gr.Gallery().style(grid=2, height='auto')
    
    inps = list(chain(btns, ims1, ims2, cond_weights))
    inps.extend([prompt, neg_prompt, scale, n_samples, seed, steps, resize_short_edge, cond_tau])
    submit.click(fn=run, inputs=inps, outputs=[metrics_output, output, cond])
demo.launch(share=True)
