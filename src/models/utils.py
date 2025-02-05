import os
import numpy as np
from PIL import Image
from safetensors import safe_open

import torch
import torch.nn.functional as F


def get_generator(seed, device):
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator


def load_unet(unet, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    unet.load_state_dict(state_dict["unet_trained"], strict=False)


def load_ip_adapter(image_proj_model, unet, ckpt_path):
    if os.path.splitext(ckpt_path)[-1] == ".safetensors":
        state_dict = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    ori_param_weights_sum = torch.sum(torch.stack([torch.sum(p) for p in image_proj_model.parameters()]))
    image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
    new_param_weights_sum = torch.sum(torch.stack([torch.sum(p) for p in image_proj_model.parameters()]))

    if ori_param_weights_sum == new_param_weights_sum:
        print(f"Weights of image_proj_model did not change!")
    
    if unet is not None:
        ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    del state_dict


def load_ckpt(image_proj_model, unet, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    image_proj_ckpt = {}

    for key in checkpoint['image_proj'].keys():
        if key.startswith("module."):
            image_proj_ckpt[key.replace("module.", "")] = checkpoint["image_proj"][key]
        else:
            image_proj_ckpt[key] = checkpoint["image_proj"][key]
    del checkpoint['image_proj']

    image_proj_model.load_state_dict(image_proj_ckpt, strict=True)
    unet.load_state_dict(checkpoint['unet_trained'], strict=False)


def load_ckpt_mllm(unet, agent_model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    unet.load_state_dict(checkpoint["unet_trained"], strict=False)
    agent_model.load_state_dict(checkpoint["agent_model"], strict=False)
