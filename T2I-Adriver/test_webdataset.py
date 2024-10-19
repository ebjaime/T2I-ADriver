from omegaconf import OmegaConf
from ldm.data.dataset_laion import WebDataModuleFromConfig
import os
import numpy as np
import os
import pytorch_lightning as pl
import torch
import webdataset as wds
from torchvision.transforms import transforms
'''
config = OmegaConf.load("T2I-Adriver/configs/pl_train/coadapter-v1-train.yaml")
datamod = WebDataModuleFromConfig(**config["data"]["params"])
dataloader = datamod.train_dataloader()

from basicsr.utils import tensor2img
import cv2
save_root = 'tmp/coadapter'
os.makedirs(save_root, exist_ok=True)

for idx, batch in enumerate(dataloader):
    print(batch.keys())
    print(batch['jpg'].shape, torch.min(batch['jpg']), torch.max(batch['jpg']))
    img = tensor2img(batch['jpg'])
    cv2.imwrite(f'{save_root}/{batch["txt"]}_{idx:03d}.png', img)
    if idx > 20:
        break
'''
from omegaconf import OmegaConf

config = OmegaConf.load("T2I-Adriver/configs/pl_train/coadapter-v1-train.yaml")
datamod = WebDataModuleFromConfig(**config["data"]["params"])
dataloader = datamod.train_dataloader()

from basicsr.utils import tensor2img
import cv2
save_root = 'tmp/coadapter'
os.makedirs(save_root, exist_ok=True)

for idx, batch in enumerate(dataloader):
    print(batch.keys())
    print(batch["style"])
    print(batch['jpg'].shape, torch.min(batch['jpg']), torch.max(batch['jpg']))
    img = tensor2img(batch['jpg'])
    cv2.imwrite(f'{save_root}/{idx:03d}.png', img)
    if idx > 20:
        break
