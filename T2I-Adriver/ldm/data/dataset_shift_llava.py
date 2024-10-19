import os
import csv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision import transforms
import torch
import numpy as np

from ldm.util import instantiate_from_config

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
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


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, filter_size_func=None, filter_keys_func=None, process_list=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        self.process_list = process_list
        with open(csv_file, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.img_dir != "None":
            img_path = os.path.join(self.img_dir, self.data[idx]["Image"])
        else:
            img_path = self.data[idx]["Image"] 
        image = Image.open(img_path).convert("RGB")
        label = self.data[idx]["Labels"]

        if self.transform:
            image = self.transform(image)

        sample = {"jpg": image, "txt":label}
        for process in self.process_list:
            sample = process(sample)
        return sample
        


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

    def filter_size(self, x):
        if self.min_size is None:
            return True
        try:
            return x['json']['original_width'] >= self.min_size and x['json']['original_height'] >= self.min_size and x[
                'json']['pwatermark'] <= self.max_pwatermark
        except Exception:
            return False

    def filter_keys(self, x):
        try:
            return ("jpg" in x) and ("txt" in x)
        except Exception:
            return False

    def make_loader(self, dataset_config):
        print(dataset_config)
        image_transforms = [instantiate_from_config(tt) for tt in dataset_config.image_transforms]
        image_transforms = transforms.Compose(image_transforms)

        process_list = []
        for process_config in dataset_config['process']:
            process_list.append(instantiate_from_config(process_config))

        if "csv_file" in dataset_config.keys():
            dataset = CustomDataset(dataset_config.csv_file, dataset_config.tar_base, transform=image_transforms, process_list=process_list)
        else:
            dataset = CustomDataset(self.csv_file, self.img_dir, transform=image_transforms, process_list=process_list)
        # dataset = (dataset.batched(self.batch_size, partial=False, collation_fn=dict_collation_fn))
            
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.val)


if __name__ == "__main__":
    # Usage example
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/pl_train/coadapter-v1-train.yaml")
    data_module = ImageDataModule(**config["data"]["params"])

    # Prepare data loaders
    train_loader = data_module.train_dataloader()

    data_module.setup('predict')
    predict_loader = data_module.test_dataloader()

    for images, labels in train_loader:
        print(images.size(), labels)
