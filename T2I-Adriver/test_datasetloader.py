# Usage example
from omegaconf import OmegaConf
from ldm.data.dataset_shift_llava import ImageDataModule
import torch
import cv2
from basicsr.utils import tensor2img

config = OmegaConf.load("T2I-Adriver/configs/pl_train/coadapter-v1-train_color_depth_seg.yaml")
data_module = ImageDataModule(**config["data"]["params"])

# Prepare data loaders
train_loader = data_module.train_dataloader()

for idx, batch in enumerate(train_loader):
    print(batch.keys())
    #print(batch["style"])
    print(batch['jpg'].shape, torch.min(batch['jpg']), torch.max(batch['jpg']))
    if idx > 20:
        break
