import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import torch


class DisasterSegDataset(Dataset):
    """Semantic Image Segmentation Dataset."""

    def __init__(self, root_dir, preprocessor, train=False, augment_data=False):
        self.root_dir = root_dir
        self.preprocessor = preprocessor
        self.train = train
        self.augment_data = augment_data
        self.transform = self.augment()

        sub_path = "train" if self.train else "valid"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "annotations", sub_path)

        # read images
        image_file_names = []

        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)

        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []

        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)

        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        if self.augment_data:
            augmented = self.transform(image=np.array(image), mask=np.array(segmentation_map))
            data = self.preprocessor(augmented['image'], augmented['mask'], return_tensors="pt")
        else:
            data = self.preprocessor(image, segmentation_map, return_tensors="pt")

        for k in data.keys():
            data[k] = torch.squeeze(data[k], dim=0) # remove batch dimension

        return data
    
    def augment(self):
        """
        On-the-Fly augmentations to increase training dataset size artificially. 
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.4,
                rotate_limit=(-25, 25), border_mode=0,
                shift_limit_x=0.2,
                shift_limit_y=0.2),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.5), contrast_limit=0.7, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-180, 50),
                val_shift_limit=0,
                p=0.5),
            A.RandomToneCurve(p=0.2),
            A.ISONoise(p=0.1),
            A.OneOf([
                A.MedianBlur(blur_limit=5, p=0.5),
                A.GaussianBlur(blur_limit=5, p=0.5)
            ], p=0.1),
            A.ImageCompression(quality_lower=35, p=0.3),
            A.Perspective(scale=(0.05, 0.13), p=0.1),
            A.LongestMaxSize(max_size=512, p=1),
            A.PadIfNeeded(512, 512, border_mode=0, p=1),
        ],
            p=1.0)
