import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
from models.load_models import load_segformer
from utils.common import common_paths
from configs.infer_cfg import default_cfg
from dataset.dataset_utils import cls_dict
from dataset.disaster_dataset import DisasterSegDataset
from torchmetrics import JaccardIndex, Accuracy


cfg = default_cfg

# Dataset
preprocessor = SegformerImageProcessor()

valid_dataset = DisasterSegDataset(
    root_dir=common_paths["dataset_root"],
    preprocessor=preprocessor,
    train=False,
    transforms=None,
)

valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'])
pbar = tqdm(valid_dataloader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

# Model
model = load_segformer(config_path=common_paths['segformer_config_path'], id2label=cls_dict, label2id={v: k for k, v in cls_dict.items()})

checkpoint = torch.load(common_paths['checkpoint_load_path'], map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() and not cfg['is_office'] else 'cpu')
model.to(device)
model.eval()

# Metrics
val_losses =  []
jaccard = JaccardIndex(task='multiclass', num_classes=len(cls_dict), ignore_index=0, average='none')
accuracy = Accuracy(task="multiclass", num_classes=len(cls_dict), ignore_index=0)

with torch.no_grad():
    for idx, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        
        # Get val metrics for one batch
        val_losses.append(outputs.loss.item())
        jaccard(predicted, labels)
        accuracy(predicted, labels)

val_loss = sum(val_losses)/len(val_losses)
val_iou = jaccard.compute()
val_accuracy = accuracy.compute()

# Class-wise IOU, mean for relevant classes 
class_iou = {class_name: iou.item() for class_name, iou in zip(cls_dict.values(), list(val_iou))}
mean_iou = val_iou.sum()/len(torch.where(val_iou > 0.0)[0])

print(f"\n\nPerformance Metrics:\n \
Val Loss: {val_loss:.4f}\n \
Class-wise IOU: {class_iou}\n \
Val P.wise Acc: {val_accuracy:.4f}\n \
Val MIoU: {mean_iou}") 