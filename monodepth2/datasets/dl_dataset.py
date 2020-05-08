# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2
import random
import torch
import torch.utils.data as data
from torchvision import transforms

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

intrinsics = {'CAM_FRONT_LEFT': [[879.03824732/4, 0.0, 613.17597314/4, 0],
[0.0, 879.03824732/4, 524.14407205/4, 0],
[0.0, 0.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 1.0]],
'CAM_FRONT': [[882.61644117/4, 0.0, 621.63358525/4, 0.0],
[0.0, 882.61644117/4, 524.38397862/4, 0.0],
[0.0, 0.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 1.0]],
'CAM_FRONT_RIGHT': [[880.41134027/4, 0.0, 618.9494972/4, 0.0],
[0.0, 880.41134027/4, 521.38918482/4, 0.0],
[0.0, 0.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 1.0]],
'CAM_BACK_LEFT': [[881.28264688/4, 0.0, 612.29732111/4, 0.0],
[0.0, 881.28264688/4, 521.77447199/4, 0.0],
[0.0, 0.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 1.0]],
'CAM_BACK': [[882.93018422/4, 0.0, 616.45479905/4, 0.0],
[0.0, 882.93018422/4, 528.27123027/4, 0.0],
[0.0, 0.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 1.0]],
'CAM_BACK_RIGHT': [[881.63835671/4, 0.0, 607.66308183/4, 0.0],
[0.0, 881.63835671/4, 525.6185326/4, 0.0],
[0.0, 0.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 1.0]]}

NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]


class DLDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self,*args, **kwargs):
        super(DLDataset, self).__init__(*args, **kwargs)

        self.intrinsics = intrinsics
        self.image_folder = self.data_path
        self.full_res_shape = (256, 306)
        self.width = 288
        self.height = 256
        # self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folders = self.filenames[index].split('/')
        scene_id = folders[0]
        sample_id = folders[1]


        camera = index % NUM_IMAGE_PER_SAMPLE

        for i in self.frame_idxs:
            current_sample = sample_id.split('_')[1]
            next_sample_id = 'sample_{}'.format(int(current_sample)+i)
            inputs[("color", i, -1)] = self.get_color(index, do_flip, scene_id, next_sample_id)

        self.K = np.array(self.intrinsics[image_names[camera][:-5]], np.float32)
        self.K[0,:] /= self.full_res_shape[1]
        self.K[1,:] /= self.full_res_shape[0]

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K).float()
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K).float()

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # if self.load_depth:
        #     depth_gt = self.get_depth(folder, frame_index, side, do_flip)
        #     inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        #     inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        # if "s" in self.frame_idxs:
        #     stereo_T = np.eye(4, dtype=np.float32)
        #     baseline_sign = -1 if do_flip else 1
        #     side_sign = -1 if side == "l" else 1
        #     stereo_T[0, 3] = side_sign * baseline_sign * 0.1
        #
        #     inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def check_depth(self):
        return False

    def get_image_path(self, index, scene_id, sample_id):
        image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

        image_path = os.path.join(self.image_folder, f'{scene_id}', f'{sample_id}', image_name)
        return image_path

    def get_color(self, index, do_flip, scene_id, sample_id):
        color = self.loader(self.get_image_path(index, scene_id, sample_id))
        color = color.resize((288,256), pil.BICUBIC)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        pass
