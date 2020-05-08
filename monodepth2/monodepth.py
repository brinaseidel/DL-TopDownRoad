import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import monodepth2.networks as networks

import PIL.Image as pil

class MonoDepthEstimator(object):
    """Transform RGB image to RGB-D using monodepth2 to estimate depth

    Args:
        model_path (string): Path to folder with saved monodepth2 model
    """

    def __init__(self, model_path):
        assert isinstance(model_path, (str))
        self.model_path = model_path
        
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING MODEL
        encoder = networks.ResnetEncoder(18, False)
        depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)

        self.encoder = encoder
        self.depth_decoder = depth_decoder
        
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        
    def __call__(self, input_image):
        original_width, original_height = input_image.size
        original_input_image_pytorch = torchvision.transforms.ToTensor()(input_image).unsqueeze(0)
        
        input_image_resized = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
        input_image_depth = torchvision.transforms.ToTensor()(input_image_resized).unsqueeze(0)
        
        self.encoder.eval();
        self.depth_decoder.eval();

        with torch.no_grad():
            features = self.encoder(input_image_depth)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]

        disp_resized = torch.nn.functional.interpolate(disp,
                                                       (original_height, original_width), mode="bilinear", align_corners=False)

        combined = torch.cat((original_input_image_pytorch, disp_resized),1).squeeze()
        return combined