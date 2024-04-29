import os
import cv2
import torch
from PIL import Image

from dataset.dataset_utils import cls_dict
from configs.infer_cfg import default_cfg
from utils.common import common_paths
from utils.visualization import plot_compare_annotations


cfg = default_cfg
ds_split = 'train'
img_dir = common_paths['dataset_root'] + f'images/{ds_split}/'

# Plot Annotations
image_l = os.listdir(img_dir)
with torch.no_grad():
    for image_file in image_l:
        fp = img_dir+image_file
        print(fp)
        
        man_annot_path = fp.replace('images', 'annotations').replace('.jpg', '_mask.png')
        gsam_annot_path = man_annot_path.replace('annotations', 'annotations-gsam')
        
        plot_compare_annotations(img=Image.open(fp), 
                                 masks_manual=cv2.imread(man_annot_path, cv2.IMREAD_GRAYSCALE), 
                                 masks_gsam=cv2.imread(gsam_annot_path, cv2.IMREAD_GRAYSCALE), 
                                 label2id={v: k for k, v in cls_dict.items()}, 
                                 alpha_img=0.7,
                                 save_path=gsam_annot_path.replace(ds_split, 'comparison')) # path or None