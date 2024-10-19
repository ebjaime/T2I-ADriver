from enum import Enum, unique

import cv2
import torch
from basicsr.utils import img2tensor
from ldm.util import resize_numpy_image
from PIL import Image
from torch import autocast
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

@unique
class ExtraCondition(Enum):
    sketch = 0
    keypose = 1
    seg = 2
    depth = 3
    canny = 4
    style = 5
    color = 6
    openpose = 7


def get_cond_model(opt, cond_type: ExtraCondition):
    if cond_type == ExtraCondition.sketch:
        from ldm.modules.extra_condition.model_edge import pidinet
        model = pidinet()
        ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()}, strict=True)
        model.to(opt.device)
        return model
    elif cond_type == ExtraCondition.seg:
        # raise NotImplementedError
        from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
        model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512').to(opt.device)
        feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
        return {'model': model, 'feature_extractor': feature_extractor}
    elif cond_type == ExtraCondition.keypose:
        import mmcv
        from mmdet.apis import init_detector
        from mmpose.apis import init_pose_model
        det_config = 'configs/mm/faster_rcnn_r50_fpn_coco.py'
        det_checkpoint = 'models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        pose_config = 'configs/mm/hrnet_w48_coco_256x192.py'
        pose_checkpoint = 'models/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        det_config_mmcv = mmcv.Config.fromfile(det_config)
        det_model = init_detector(det_config_mmcv, det_checkpoint, device=opt.device)
        pose_config_mmcv = mmcv.Config.fromfile(pose_config)
        pose_model = init_pose_model(pose_config_mmcv, pose_checkpoint, device=opt.device)
        return {'pose_model': pose_model, 'det_model': det_model}
    elif cond_type == ExtraCondition.depth:
        from ldm.modules.extra_condition.midas.api import MiDaSInference
        model = MiDaSInference(model_type='dpt_hybrid').to(opt.device)
        return model
    elif cond_type == ExtraCondition.canny:
        return None
    elif cond_type == ExtraCondition.style:
        from transformers import CLIPProcessor, CLIPVisionModel
        version = 'openai/clip-vit-large-patch14'
        processor = CLIPProcessor.from_pretrained(version)
        clip_vision_model = CLIPVisionModel.from_pretrained(version).to(opt.device)
        return {'processor': processor, 'clip_vision_model': clip_vision_model}
    elif cond_type == ExtraCondition.color:
        return None
    elif cond_type == ExtraCondition.openpose:
        from ldm.modules.extra_condition.openpose.api import OpenposeInference
        model = OpenposeInference().to(opt.device)
        return model
    else:
        raise NotImplementedError


def get_cond_sketch(opt, cond_image, cond_inp_type, cond_model=None):
    if isinstance(cond_image, str):
        edge = cv2.imread(cond_image)
    else:
        # for gradio input, pay attention, it's rgb numpy
        edge = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
    edge = resize_numpy_image(edge, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = edge.shape[:2]
    if cond_inp_type == 'sketch':
        edge = img2tensor(edge)[0].unsqueeze(0).unsqueeze(0) / 255.
        edge = edge.to(opt.device)
    elif cond_inp_type == 'image':
        edge = img2tensor(edge).unsqueeze(0) / 255.
        edge = cond_model(edge.to(opt.device))[-1]
    else:
        raise NotImplementedError

    # edge = 1-edge # for white background
    edge = edge > 0.5
    edge = edge.float()

    return edge


# def get_cond_seg(opt, cond_image, cond_inp_type='image', cond_model=None):
#     if isinstance(cond_image, str):
#         seg = cv2.imread(cond_image)
#     else:
#         seg = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
#     seg = resize_numpy_image(seg, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
#     opt.H, opt.W = seg.shape[:2]
#     if cond_inp_type == 'seg':
#         seg = img2tensor(seg).unsqueeze(0) / 255.
#         seg = seg.to(opt.device)
#     else:
#          raise NotImplementedError # TODO:
#     return seg


def get_cond_seg(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        seg = cv2.imread(cond_image)
    else:
        seg = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
    seg = resize_numpy_image(seg, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = seg.shape[:2]
    if cond_inp_type == 'seg':
        seg = img2tensor(seg).unsqueeze(0) / 255.
        seg = seg.to(opt.device)
        return seg
    else:
        seg = cond_model['feature_extractor'](seg, return_tensors="pt").to(opt.device)
        # opt.H, opt.W = seg['pixel_values'].shape[-2], seg['pixel_values'].shape[-1]
        with torch.no_grad():
            outputs = cond_model['model'](**seg)
            seg = outputs.logits

        # Apply thresholding
        threshold = 0.5
        seg = torch.nn.functional.softmax(seg, dim=1)  # Convert logits to probabilities
        seg[seg < threshold] = 0

        seg = F.interpolate(seg, size=(opt.H, opt.W), mode="bilinear", align_corners=False)
        seg = torch.argmax(seg, dim=1).unsqueeze(1)  # Get the segmentation mask

        # FIXME: Define a color palette (example with 150 colors, modify as needed)
        palette = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], 
                            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],]) 
                            # [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], 
                            # [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], 
                            # [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0], 
                            # [192, 64, 0], [64, 192, 0], [192, 192, 0], [64, 64, 128], [192, 64, 128], 
                            # [64, 192, 128], [192, 192, 128], [0, 0, 64], [128, 0, 64], [0, 128, 64], 
                            # [128, 128, 64], [0, 0, 192], [128, 0, 192], [0, 128, 192], [128, 128, 192], 


        # Remove batch and channel dimensions, should be of shape [H, W]
        seg = seg.squeeze(0).squeeze(0).cpu().numpy()

        colored_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for class_idx in range(palette.shape[0]):
            colored_seg[seg == class_idx] = palette[class_idx]

        # colored_seg_image = Image.fromarray(colored_seg)

        # return colored_seg_image
        # Convert the colored segmentation to a tensor
        colored_seg_tensor = torch.from_numpy(colored_seg).permute(2, 0, 1).float().unsqueeze(0) / 255.
        colored_seg_tensor=colored_seg_tensor.to(opt.device)  # Change shape to [1, 3, H, W]

        return colored_seg_tensor


def get_cond_keypose(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        pose = cv2.imread(cond_image)
    else:
        pose = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
    pose = resize_numpy_image(pose, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = pose.shape[:2]
    if cond_inp_type == 'keypose':
        pose = img2tensor(pose).unsqueeze(0) / 255.
        pose = pose.to(opt.device)
    elif cond_inp_type == 'image':
        from ldm.modules.extra_condition.utils import imshow_keypoints
        from mmdet.apis import inference_detector
        from mmpose.apis import (inference_top_down_pose_model, process_mmdet_results)

        # mmpose seems not compatible with autocast fp16
        with autocast("cuda", dtype=torch.float32):
            mmdet_results = inference_detector(cond_model['det_model'], pose)
            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, 1)

            # optional
            return_heatmap = False
            dataset = cond_model['pose_model'].cfg.data['test']['type']

            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None
            pose_results, returned_outputs = inference_top_down_pose_model(
                cond_model['pose_model'],
                pose,
                person_results,
                bbox_thr=0.2,
                format='xyxy',
                dataset=dataset,
                dataset_info=None,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

        # show the results
        pose = imshow_keypoints(pose, pose_results, radius=2, thickness=2)
        pose = img2tensor(pose).unsqueeze(0) / 255.
        pose = pose.to(opt.device)
    else:
        raise NotImplementedError

    return pose


def get_cond_depth(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        depth = cv2.imread(cond_image)
    else:
        depth = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
    depth = resize_numpy_image(depth, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = depth.shape[:2]
    if cond_inp_type == 'depth':
        depth = img2tensor(depth).unsqueeze(0) / 255.
        depth = depth.to(opt.device)
    elif cond_inp_type == 'image':
        depth = img2tensor(depth).unsqueeze(0) / 127.5 - 1.0
        depth = cond_model(depth.to(opt.device)).repeat(1, 3, 1, 1)
        depth -= torch.min(depth)
        depth /= torch.max(depth)
    else:
        raise NotImplementedError

    return depth


def get_cond_canny(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        canny = cv2.imread(cond_image)
    else:
        canny = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
    canny = resize_numpy_image(canny, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = canny.shape[:2]
    if cond_inp_type == 'canny':
        canny = img2tensor(canny)[0:1].unsqueeze(0) / 255.
        canny = canny.to(opt.device)
    elif cond_inp_type == 'image':
        canny = cv2.Canny(canny, 100, 200)[..., None]
        canny = img2tensor(canny).unsqueeze(0) / 255.
        canny = canny.to(opt.device)
    else:
        raise NotImplementedError

    return canny


def get_cond_style(opt, cond_image, cond_inp_type='image', cond_model=None):
    assert cond_inp_type == 'image'
    if isinstance(cond_image, str):
        style = Image.open(cond_image)
    else:
        # numpy image to PIL image
        style = Image.fromarray(cond_image)

    style_for_clip = cond_model['processor'](images=style, return_tensors="pt")['pixel_values']
    style_feat = cond_model['clip_vision_model'](style_for_clip.to(opt.device))['last_hidden_state']

    return style_feat


def get_cond_color(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        color = cv2.imread(cond_image)
    else:
        color = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
    color = resize_numpy_image(color, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = color.shape[:2]
    if cond_inp_type == 'image':
        color = cv2.resize(color, (opt.W//64, opt.H//64), interpolation=cv2.INTER_CUBIC)
        color = cv2.resize(color, (opt.W, opt.H), interpolation=cv2.INTER_NEAREST)
    color = img2tensor(color).unsqueeze(0) / 255.
    color = color.to(opt.device)
    return color


def get_cond_openpose(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        openpose_keypose = cv2.imread(cond_image)
    else:
        openpose_keypose = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
    openpose_keypose = resize_numpy_image(
        openpose_keypose, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = openpose_keypose.shape[:2]
    if cond_inp_type == 'openpose':
        openpose_keypose = img2tensor(openpose_keypose).unsqueeze(0) / 255.
        openpose_keypose = openpose_keypose.to(opt.device)
    elif cond_inp_type == 'image':
        with autocast('cuda', dtype=torch.float32):
            openpose_keypose = cond_model(openpose_keypose)
        openpose_keypose = img2tensor(openpose_keypose).unsqueeze(0) / 255.
        openpose_keypose = openpose_keypose.to(opt.device)

    else:
        raise NotImplementedError

    return openpose_keypose


def get_adapter_feature(inputs, adapters, names):
    ret_feat_map = None
    ret_feat_seq = None
    if not isinstance(inputs, list):
        inputs = [inputs]
        adapters = [adapters]

    for input, adapter, name in zip(inputs, adapters, names):
        print(name)
        cur_feature = adapter['model'](input)
        if isinstance(cur_feature, list):
            if ret_feat_map is None:
                ret_feat_map = list(map(lambda x: x * adapter['cond_weight'], cur_feature))
            else:
                ret_feat_map = list(map(lambda x, y: x + y * adapter['cond_weight'], ret_feat_map, cur_feature))
        else:
            if ret_feat_seq is None:
                ret_feat_seq = cur_feature * adapter['cond_weight']
            else:
                ret_feat_seq = torch.cat([ret_feat_seq, cur_feature * adapter['cond_weight']], dim=1)

    return ret_feat_map, ret_feat_seq



