import argparse
import logging
import os
import os.path as osp
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from basicsr.utils import (
    get_env_info,
    get_root_logger,
    get_time_str,
    img2tensor,
    scandir,
    tensor2img,
)
from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf
from PIL import Image

# from ldm.data.dataset_coco import dataset_coco_mask_color
from ldm.data.dataset_fill import dataset_fill_mask

from dist_util import get_bare_model, get_dist_info, init_dist, master_only
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter
from ldm.util import instantiate_from_config
from ldm.modules.extra_condition.model_edge import pidinet

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.set_per_process_memory_fraction(0.5, device=0)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--bsize", type=int, default=8, help="the prompt to render")
parser.add_argument("--epochs", type=int, default=10000, help="the prompt to render")
parser.add_argument("--num_workers", type=int, default=8, help="the prompt to render")
parser.add_argument(
    "--use_shuffle", type=bool, default=True, help="the prompt to render"
)
parser.add_argument(
    "--dpm_solver",
    action="store_true",
    help="use dpm_solver sampling",
)
parser.add_argument(
    "--plms",
    action="store_true",
    help="use plms sampling",
)
parser.add_argument(
    "--auto_resume",
    action="store_true",
    help="use plms sampling",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="T2I-Adriver/models/sd-v1-4.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/stable-diffusion/train_sketch.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--print_fq",
    type=int,
    default=100,
    help="", # TODO:
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--gpus",
    default=[0, 1, 2, 3],
    help="gpu idx",
)
parser.add_argument(
    "--local_rank", default=0, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--launcher", default="pytorch", type=str, help="node rank for distributed training"
)
parser.add_argument("--l_cond", default=4, type=int, help="number of scales")
opt = parser.parse_args()

if __name__ == "__main__":
    os.chdir("T2I-Adriver")
    print("-> Configuration from", opt.config)
    config = OmegaConf.load(f"{opt.config}")
    opt.name = config["name"]

    # distributed setting
    print("-> Distributed setting")
    init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    torch.cuda.set_device(opt.local_rank)

    os.chdir("..")

    # stable diffusion
    print("-> Stable diffusion")
    model = load_model_from_config(config, f"{opt.ckpt}").to(device)

    # to gpus
    print("-> to gpus")
    model = torch.nn.parallel.DistributedDataParallel(
       model, device_ids=[opt.local_rank], output_device=opt.local_rank
    )
    


    print(model.get_learned_conditioning(""))
    print(model.get_learned_conditioning("A road with a traffic sign"))
    
    # training