import os
import csv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision import transforms
import torch
import numpy as np

from ldm.util import instantiate_from_config
from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import ZipBackend, FileBackend, HDF5Backend

#https://stackoverflow.com/questions/42462431/oserror-broken-data-stream-when-reading-image-file
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    # (existing code for dict_collation_fn)
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, tar_base, batch_size, csv_file=None, train=None, validation=None, val=None, num_workers=4, multinode=True, min_size=None, max_pwatermark=1.0, **kwargs):
        super().__init__()
        self.img_dir = tar_base
        self.csv_file = csv_file 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark
        self.filter_size_func = None
        self.filter_keys_func = None
        self.multinode = multinode
        self.train = train
        self.val = val
        self.validation = validation

    def make_loader(self, dataset_config):
        dataset = SHIFTDataset(
            data_root=self.img_dir,
            split="train",
            keys_to_load=[
                Keys.images,
                Keys.intrinsics,
                Keys.boxes2d,
                Keys.boxes2d_classes,
                Keys.boxes2d_track_ids,
                Keys.depth_maps,
                Keys.segmentation_masks,
            ],
            views_to_load=["front"],
            framerate="images",
            shift_type="discrete",
            backend=FileBackend(),  # also supports HDF5Backend(), FileBackend()
            verbose=True,
        )
        
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=dict_collation_fn)

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.val)

if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/pl_train/coadapter-v1-train.yaml")
    data_module = ImageDataModule(**config["data"]["params"])

    # Prepare data loaders
    train_loader = data_module.train_dataloader()

    for batch in train_loader:
        print(batch)
