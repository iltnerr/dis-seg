import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DisasterSegDataset(Dataset):
    """Semantic Image Segmentation Dataset."""

    def __init__(self, root_dir, feature_extractor, train=True, transforms=None):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train
        self.transforms = transforms

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

        if self.transforms:
            data = self.transforms(image=np.array(image), mask=np.array(segmentation_map))
            # randomly crop + pad both image and segmentation map to same size
            encoded_inputs = self.feature_extractor(data['image'], data['mask'], return_tensors="pt")

        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs