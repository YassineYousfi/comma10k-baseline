import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import pickle
import cv2
import albumentations as A
from albumentations.core.composition import Compose
from typing import Callable, List
from pathlib import Path
import os
from torch.utils.data import Dataset
import torch
import sys

def pad_to_multiple(x, k=32):
    return int(k*(np.ceil(x/k)))

def get_train_transforms(height: int = 437, 
                         width: int = 582, 
                         level: str = 'hard'): 
    if level == 'light':
        return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.IAAAdditiveGaussianNoise(p=0.2),
                A.OneOf(
                    [A.CLAHE(p=1.0),
                    A.RandomBrightness(p=1.0),
                    A.RandomGamma(p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.IAASharpen(p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.RandomContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    ],p=0.5),
                A.Resize(height=height, width=width, p=1.0),
                A.PadIfNeeded(pad_to_multiple(height), 
                              pad_to_multiple(width), 
                              border_mode=cv2.BORDER_CONSTANT, 
                              value=0, 
                              mask_value=0)
            ], p=1.0)

    elif level == 'hard':
        return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.IAAAdditiveGaussianNoise(p=0.2),
                A.OneOf(
                    [A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                     A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                     A.ShiftScaleRotate(
                         shift_limit=0,
                         scale_limit=0,
                         rotate_limit=10,
                         border_mode=cv2.BORDER_CONSTANT,
                         value=0,
                         mask_value=0,
                         p=1.0
                     ),
                     A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.CLAHE(p=1.0),
                    A.RandomBrightness(p=1.0),
                    A.RandomGamma(p=1.0),
                    A.ISONoise(p=1.0)
                    ],p=0.5),
                A.OneOf(
                    [A.IAASharpen(p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.RandomContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    ],p=0.5),
                A.Resize(height=height, width=width, p=1.0),
                A.Cutout(p=0.3),
                A.PadIfNeeded(pad_to_multiple(height), 
                              pad_to_multiple(width), 
                              border_mode=cv2.BORDER_CONSTANT, 
                              value=0, 
                              mask_value=0) 
            ], p=1.0)
    elif level == 'hard_weather':
        raise NotImplementedError("WIP")

def get_valid_transforms(height: int = 437, 
                         width: int = 582): 
    return A.Compose([
            A.Resize(height=height, width=width, p=1.0),
            A.PadIfNeeded(pad_to_multiple(height), 
                          pad_to_multiple(width), 
                          border_mode=cv2.BORDER_CONSTANT, 
                          value=0, 
                          mask_value=0)
        ], p=1.0)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn: Callable):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

class TrainRetriever(Dataset):

    def __init__(self, 
                 data_path: Path, 
                 image_names: List[str], 
                 preprocess_fn: Callable, 
                 transforms: Compose,
                 class_values: List[int]):
        super().__init__()
        
        self.data_path = data_path
        self.image_names = image_names
        self.transforms = transforms
        self.preprocess = get_preprocessing(preprocess_fn)
        self.class_values = class_values
        self.images_folder = 'imgs'
        self.masks_folder = 'masks'

    def __getitem__(self, index: int):
        
        image_name = self.image_names[index]
        
        image = cv2.imread(str(self.data_path/self.images_folder/image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.data_path/self.masks_folder/image_name), 0).astype('uint8')

        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

        mask = np.stack([(mask == v) for v in self.class_values], axis=-1).astype('uint8')

        if self.preprocess:
            sample = self.preprocess(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

        return image, mask

    def __len__(self) -> int:
        return len(self.image_names)
    
    
