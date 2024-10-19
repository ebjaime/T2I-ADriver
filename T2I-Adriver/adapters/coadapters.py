import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPVisionModel

from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.util import instantiate_from_config
from ldm.modules.extra_condition.midas.api import MiDaSInference
from ldm.modules.extra_condition.model_edge import pidinet
from ldm.inference_base import read_state_dict

from basicsr.utils import img2tensor

class CoAdapter(LatentDiffusion):

    def __init__(self, adapter_configs, coadapter_fuser_config, noise_schedule, *args, **kwargs):
        super(CoAdapter, self).__init__(*args, **kwargs)
        self.adapters = nn.ModuleDict()
        for adapter_config in adapter_configs:
            cond_name = adapter_config['cond_name']
            self.adapters[cond_name] = instantiate_from_config(adapter_config)
            if 'pretrained' in adapter_config:
                self.load_pretrained_adapter(cond_name, adapter_config['pretrained'])
        self.coadapter_fuser = instantiate_from_config(coadapter_fuser_config)
        self.training_adapters = list(self.adapters.keys())
        self.noise_schedule = noise_schedule

        # clip vision model as style model backbone
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            'openai/clip-vit-large-patch14'
        )
        self.clip_vision_model = self.clip_vision_model.eval()
        self.clip_vision_model.train = disabled_train
        for param in self.clip_vision_model.parameters():
            param.requires_grad = False

        # depth model
        self.midas_model = MiDaSInference(model_type='dpt_hybrid')
        self.midas_model = self.midas_model.eval()
        self.midas_model.train = disabled_train
        for param in self.midas_model.parameters():
            param.requires_grad = False

        # sketch model
        self.sketch_model = pidinet()
        ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
        self.sketch_model.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()}, strict=True)
        self.sketch_model = self.sketch_model.eval()
        self.sketch_model.train = disabled_train
        for param in self.sketch_model.parameters():
            param.requires_grad = False
            
        # seg model
        from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512').to(self.device)
        self.seg_model_feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')

    def load_pretrained_adapter(self, cond_name, pretrained_path):
        print(f'loading adapter {cond_name} from {pretrained_path}')
        state_dict = read_state_dict(pretrained_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('adapter.'):
                new_state_dict[k[len('adapter.'):]] = v
            else:
                new_state_dict[k] = v
        self.adapters[cond_name].load_state_dict(new_state_dict)

    # FIXME: for batches of size > 1
    @torch.inference_mode()
    def data_preparation_on_gpu(self, batch, keep_conds=["style", "depth", "sketch", "seg"]):
        print("Data preparation on GPU")
        # style
        if "style" in keep_conds and "style" not in batch.keys():
            style = batch['style'].to(self.device)
            style = self.clip_vision_model(style)['last_hidden_state']
            batch['style'] = style

        # depth
        if "depth" in keep_conds and "depth" not in batch.keys():
            # depth = self.midas_model(batch['jpg']).repeat(1, 3, 1, 1)  # jpg range [0, 1]
            # depth = (depth - depth.min(dim=(1, 2, 3), keepdim=True)[0]) / depth.max(dim=(1, 2, 3), keepdim=True)[0]
            # print("Depth Range: ", torch.min(depth), torch.max(depth))
            # batch['depth'] = depth
            depth = self.midas_model(batch['jpg']).repeat(1, 3, 1, 1)  # jpg range [0, 1]
            depth_min = depth.view(depth.size(0), -1).min(dim=1, keepdim=True)[0].view(depth.size(0), 1, 1, 1)
            depth_max = depth.view(depth.size(0), -1).max(dim=1, keepdim=True)[0].view(depth.size(0), 1, 1, 1)
            depth = (depth - depth_min) / (depth_max - depth_min)
            batch['depth'] = depth

        # sketch
        if "sketch" in keep_conds and "sketch" not in batch.keys():
            edge = batch['jpg']  # [0, 1]
            edge = self.sketch_model(edge)[-1]
            edge = (edge > 0.5).float()
            batch['sketch'] = edge
        
        # seg
        if "seg" in keep_conds and "seg" not in batch.keys():
            # Get the batch size and image dimensions
            batch_size, _, seg_H, seg_W = batch["jpg"].shape

            # Convert the image tensor to the appropriate format
            image_tensor = batch["jpg"] * 255.0  # Convert the values from [0, 1] to [0, 255]
            image_array = image_tensor.cpu().numpy().astype(np.uint8)  # Convert to NumPy array and ensure uint8 format

            # Initialize a list to store the segmentation results
            seg_list = []
            for i in range(batch_size):
                # Process each image in the batch individually
                single_image = image_array[i]  # Shape: (3, 512, 512)
                single_image = np.transpose(single_image, (1, 2, 0))  # Change shape to (512, 512, 3)
                single_image = self.seg_model_feature_extractor(single_image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.seg_model(**single_image)
                    seg = outputs.logits

                # Apply thresholding
                seg = torch.nn.functional.softmax(seg, dim=1)  # Convert logits to probabilities
                seg[seg < 0.5] = 0

                # Resize the segmentation map to the original image size
                seg = F.interpolate(seg, size=(seg_H, seg_W), mode="bilinear", align_corners=False)
                seg = torch.argmax(seg, dim=1).unsqueeze(1)  # Get the segmentation mask

                # Define a color palette (example with 150 colors, modify as needed)
                palette = np.array([
                    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], 
                    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                    # Add more colors as needed...
                ])

                seg = seg.squeeze(0).cpu().numpy()  # Remove the batch and channel dimensions

                # Create the colored segmentation map
                colored_seg = np.zeros((seg.shape[1], seg.shape[2], 3), dtype=np.uint8)  # Shape: (512, 512, 3)
                for class_idx in range(palette.shape[0]):
                    colored_seg[seg[0] == class_idx] = palette[class_idx]

                colored_seg_tensor = torch.from_numpy(colored_seg).permute(2, 0, 1).float() / 255.  # Shape: (3, 512, 512)
                seg_list.append(colored_seg_tensor.unsqueeze(0))  # Add batch dimension

            # Stack all segmentation results into a batch
            batch['seg'] = torch.cat(seg_list, dim=0).to(self.device)  # Shape: [n, 3, H, W]

        print("COADAPTER DATA PREPARATION ON GPU")
        if not hasattr(self, 'num_steps_to_save_imgs'):
            return
        
        if self.global_step % self.num_steps_to_save_imgs == 0: 
            print(batch.keys(), keep_conds)
            from torchvision import transforms
            from PIL import Image
            import os
            save_dir = f"T2I-Adapter/dataset/while_training/test_img_{int(self.global_step/10)}"
            if not os.path.exists(save_dir):
                print(f"{save_dir} does not exist")
                os.makedirs(save_dir, exist_ok=True)
            for cond in ["depth", "sketch", "color", "seg", "jpg"]:
                if cond not in batch.keys() or (cond not in keep_conds and cond != "jpg"): 
                    continue
                for idx, tensor_image in enumerate(batch[cond]):
                    to_pil_image = transforms.ToPILImage()
                    pil_image = to_pil_image(tensor_image)
                    pil_image.save(f"{save_dir}/{cond}_{idx}.png")

    def get_adapter_features(self, batch, keep_conds):
        features = dict()
        for cond_name in keep_conds:
            if cond_name in batch:
                features[cond_name] = self.adapters[cond_name](batch[cond_name])
        return features

    def shared_step(self, batch,batch_idx=0,**kwargs):
        # FIXME: removes text labels sometimes
        # for k in self.ucg_training:
        #     p = self.ucg_training[k]
        #     for i in range(len(batch[k])):
        #         if self.ucg_prng.choice(2, p=[1 - p, p]):
        #             if isinstance(batch[k], list):
        #                 batch[k][i] = ""
        #FIXME: batch['jpg'] = batch['jpg'] * 2 - 1
        print("COADAPTER SHARED STEP")
        p = np.random.rand()
        if p < 0.1:
            keep_conds = self.training_adapters
        elif p < 0.2:
            keep_conds = []
        else:
            keep = np.random.choice(2, len(self.training_adapters), p=[0.5, 0.5])
            keep_conds = [cond_name for k, cond_name in zip(keep, self.training_adapters) if k == 1]
        
        keep_conds = self.training_adapters #FIXME: 

        self.data_preparation_on_gpu(batch, keep_conds)
        print("\n Batch KEYS",batch.keys())
        x, c = self.get_input(batch, self.first_stage_key)
        features = self.get_adapter_features(batch, keep_conds)
        features_adapter, append_to_context = self.coadapter_fuser(features)
        
        t = self.get_time_with_schedule(self.noise_schedule, x.size(0))
        loss, loss_dict = self(x, c, t=t, features_adapter=features_adapter, append_to_context=append_to_context)
        # FIXME: needed for DDIMSampler to have this info
        self.features_adapter = features_adapter
        self.append_to_context = append_to_context
        return loss, loss_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.adapters.parameters()) + list(self.coadapter_fuser.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def on_save_checkpoint(self, checkpoint):
        keys = list(checkpoint['state_dict'].keys())
        for key in keys:
            if 'adapter' not in key:
                del checkpoint['state_dict'][key]

    def on_load_checkpoint(self, checkpoint):
        for name in self.state_dict():
            if 'adapter' not in name:
                checkpoint['state_dict'][name] = self.state_dict()[name]
