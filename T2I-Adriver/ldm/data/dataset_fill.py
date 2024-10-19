from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models
import numpy as np
import torch

from PIL import Image,ImageFile
from transformers import CLIPProcessor, CLIPModel
import itertools


class dataset_fill_mask(Dataset):
    def __init__(self, split='train', empty_prompts=False, image_prompt=False):
        self.dataset = load_dataset('BertramRay/fill1k')[split]
        self.dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
        self.empty_prompts=empty_prompts
        self.image_prompt = image_prompt
        if(image_prompt):
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            type_of_road = ["paved road", "highway", "dirt road"]
            dirs_of_the_road = ["going straight", "going left", "going right"]
            type_of_lines = ["with double center lines", "with a single center line", "with a broken center line", "with mixed center lines"]
            environments = ["in the open road", "in nature", "on a city street"]
            objects = ["with an object to the right", "with a car to the right",
                    "with an object to the left", "with a car to the left",
                    "with an object in front", "with a car in front",
                    ]
                    
            self.prompts= []
            for type_road, direction, line, environment, object in itertools.product(type_of_road, dirs_of_the_road, type_of_lines, environments, objects):
                self.prompts.append(" ".join([type_road, direction, line, environment, object]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        newsize = (256, 256) 
        sample["image"] = sample["image"].resize(newsize) 
        #mask =  transform(self.modify(sample["image"] )) / 255.
        mask = transform(sample["image"]) / 255.
        sample["image"] = transform(sample["image"]) / 255.

        if self.empty_prompts:
            sentence = "" 
        elif self.image_prompt:
            inputs = self.processor(text=self.prompts, images=sample["image"], return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # probabilities 
            probs = probs.detach().numpy()
            sentence = self.prompts[np.argmax(probs)]
        else:
            sentence = sample['text']
        return {'im': sample['image'], 'mask':mask, 'sentence': sentence}

    def decode_segmap(image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def modify(self, img):
        trf = transforms.Compose([transforms.Resize(640),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                        std = [0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0)
        out = self.dlab(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        result = self.decode_segmap(om) # use result = crop(om,source) while using crop function discussed later
        return result
    
