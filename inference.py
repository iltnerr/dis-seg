import os
import random
import cv2
import torch
from PIL import Image

from dataset.dataset_utils import cls_dict
from configs.infer_cfg import default_cfg
from utils.common import common_paths
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from utils.visualization import plot_compare_predictions


cfg = default_cfg
img_dir = common_paths['dataset_root'] + 'images/valid/'

# Model
id2label = dict(cls_dict)
label2id = {v: k for k, v in cls_dict.items()}

preprocessor = SegformerImageProcessor()
model = SegformerForSemanticSegmentation.from_pretrained(
    cfg['model_type'],
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

checkpoint = torch.load(cfg['checkpoint_load_path'], map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() and not cfg['IS_OFFICE'] else 'cpu')
model.to(device) 

# Plot results for random images
val_images = os.listdir(img_dir)
random.shuffle(val_images)

for image_file in val_images:
    fp = img_dir+image_file
    print(fp)
    
    # Get GT
    gt_path = fp.replace('images', 'annotations').replace('.jpg', '_mask.png')
    gt_map = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    # Predict
    img = Image.open(fp)
    encoding = preprocessor(img, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(device)
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(logits, # TODO: Review upsamling: Could have an impact on sharpness of predictions
                    size=img.size[::-1],  # (H,W)
                    mode='bilinear',
                    align_corners=False)
    preds = torch.sigmoid(upsampled_logits).argmax(dim=1)[0]

    plot_compare_predictions(img=img, preds=preds, label2id=label2id, gt_map=gt_map,
                             alpha_img=0.7)
    

exit()