# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import numpy as np
from torchvision import transforms
from PIL import Image

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        def color_jittering(im, strength):
            # Modify this function to handle 4-channel images
            if im.shape[2] == 4:  # Check if the image has 4 channels
                rgb_im = im[:, :, :3]
                nir_im = im[:, :, 3]
                
                # Apply color jittering to RGB channels
                rgb_im = transforms.ColorJitter(
                    brightness=0.4 * strength,
                    contrast=0.4 * strength,
                    saturation=0.4 * strength,
                    hue=0.1 * strength,
                )(rgb_im)
                
                # Combine RGB and NIR channels
                return np.dstack((rgb_im, nir_im))
            else:
                # Original implementation for 3-channel images
                return transforms.ColorJitter(
                    brightness=0.4 * strength,
                    contrast=0.4 * strength,
                    saturation=0.4 * strength,
                    hue=0.1 * strength,
                )(im)

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        # Modify this method to handle 4-channel images
        if isinstance(image, np.ndarray) and image.shape[2] == 4:
            # Convert numpy array to PIL Image for RGB channels
            rgb_image = Image.fromarray(image[:, :, :3].astype(np.uint8))
            nir_channel = image[:, :, 3]

            crops = []
            for i, (scale, n_crop) in enumerate(zip(self.scales, self.n_crops)):
                for _ in range(n_crop):
                    rgb_crop = self.global_transfo(rgb_image.copy())
                    nir_crop = self.global_transfo(Image.fromarray(nir_channel))
                    
                    # Combine RGB and NIR channels
                    combined_crop = np.dstack((np.array(rgb_crop), np.array(nir_crop)))
                    crops.append(combined_crop)
            return crops
        else:
            # Original implementation for 3-channel images
            return super().__call__(image)
