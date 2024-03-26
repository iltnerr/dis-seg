import os
import numpy as np
from PIL import Image
import torch
from torch import nn

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from utils.visualization import plot_result, plot_result_gt
from dataset.dataset_utils import cls_dict
from configs.infer_cfg import infer_cfg


# Define the directory where the checkpoint file is located
checkpoint_dir = './checkpoints'

# List files in the directory
checkpoint_files = os.listdir(checkpoint_dir)

# Filter for files that end with ".pt"
filtered_checkpoint_files = [file for file in checkpoint_files if file.endswith(".pt")]

if len(filtered_checkpoint_files) == 1:
    # If exactly one .pt file is found, update the checkpoint_load_path value
    infer_cfg['checkpoint_load_path'] = os.path.join(checkpoint_dir, filtered_checkpoint_files[0])
elif len(filtered_checkpoint_files) > 1:
    # Handle the case when multiple .pt files are found
    print("Error: Multiple .pt files found in the directory. Update the code to handle this case.")

# Labels and classes names correspondance
id2label = dict(cls_dict)
label2id = {v: k for k, v in cls_dict.items()}


feature_extractor = SegformerImageProcessor(do_reduce_labels=False)
model = SegformerForSemanticSegmentation.from_pretrained(
    infer_cfg['model_type'],
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

checkpoint = torch.load(infer_cfg['checkpoint_load_path'])
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # initialize available device
model.to(device)  # send model to device

img = Image.fromarray(np.array(Image.open(infer_cfg['input_path'])))

encoding = feature_extractor(img, return_tensors="pt")
pixel_values = encoding.pixel_values.to(device)
outputs = model(pixel_values=pixel_values)
pred_masks = outputs.logits
logits = outputs.logits.cpu()
upsampled_logits = nn.functional.interpolate(logits,
                size=img.size[::-1],  # (height, width)
                mode='bilinear',
                align_corners=False)
upsampled_logits = torch.sigmoid(upsampled_logits)
seg = upsampled_logits.argmax(dim=1)[0]


# Plot result based on the condition
if 'disaster_ds_raw' in infer_cfg['input_path']:
    plot_result_gt(img, seg, cls_dict, infer_cfg, model)
else:
    plot_result(img, seg, cls_dict, model)
