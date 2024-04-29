from dataset.disaster_dataset import DisasterSegDataset
from utils.common import common_paths
from transformers import SegformerImageProcessor
from torch.utils.data import DataLoader
from configs.train_cfg import default_cfg
from dataset.dataset_utils import cls_dict
from dataset.dataset_utils import pixel_values_to_pil_image
from utils.visualization import visualize_augmentations


cfg = default_cfg

preprocessor = SegformerImageProcessor()

train_dataset = DisasterSegDataset(
    root_dir=common_paths["dataset_root"],
    preprocessor=preprocessor,
    train=True,
    augment_data=False
)

train_dataset_augmented = DisasterSegDataset(
    root_dir=common_paths["dataset_root"],
    preprocessor=preprocessor,
    train=True,
    augment_data=True
)

valid_dataset = DisasterSegDataset(
    root_dir=common_paths["dataset_root"],
    preprocessor=preprocessor,
    train=False,
    augment_data=False
)

train_dataloader = DataLoader(train_dataset, batch_size=1)
train_aug_dataloader = DataLoader(train_dataset_augmented, batch_size=1)
valid_dataloader = DataLoader(valid_dataset, batch_size=1)

for batch, batch_aug in zip(train_dataloader, train_aug_dataloader):
    pixel_values = batch['pixel_values'][0]
    pixel_values_aug= batch_aug['pixel_values'][0]
    labels = batch['labels'][0]
    labels_augmented = batch_aug['labels'][0]

    image = pixel_values_to_pil_image(pixel_values)
    image_augmented = pixel_values_to_pil_image(pixel_values_aug
                                          )
    visualize_augmentations(img=image, masks=labels, 
                            aug_image=image_augmented, aug_masks=labels_augmented, 
                            label2id={v: k for k, v in cls_dict.items()})