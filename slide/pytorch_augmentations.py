#augmentation and preprocessing and stained patches
import albumentations as album
from stainlib.augmentation.augmenter import HedLighterColorAugmenter
import numpy as np
import random


#colour augmentations 
def get_stained_patch(patch, patch_size, num_patches): 
    options = ["stain", "no stain"]
    decision = random.choices(options, weights=(0.3, 0.7))[0]
    if decision =="stain": 
        patch = np.array(patch)[0:patch_size,0:patch_size,0:3]
        hed_lighter_aug = HedLighterColorAugmenter()
        hed_lighter_aug.randomize()
        return hed_lighter_aug.transform(patch)
    else: 
        return patch


def get_validation_augmentation():   
    test_transform = [
        album.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)
