import cv2
import os

from PIL import Image

from configs.infer_cfg import default_cfg
from dataset.dataset_utils import cls_dict
from utils.common import common_paths
from utils.visualization import plot_compare_annotations


cfg = default_cfg
ds_split = 'val'
img_dir = common_paths['dataset_root'] + f'images/{ds_split}/'
anns1 = 'annotations'
anns2 = 'annotations'

# Compare Annotations
image_l = os.listdir(img_dir)

for image_file in image_l:
    fp = img_dir+image_file
    print(fp)
    
    man_annot_path = fp.replace('images', anns1).replace('.jpg', '_mask.png')
    gsam_annot_path = man_annot_path.replace(anns1, anns2)

    img = Image.open(fp)
    masks_manual = cv2.imread(man_annot_path, cv2.IMREAD_GRAYSCALE)
    masks_gsam = cv2.imread(gsam_annot_path, cv2.IMREAD_GRAYSCALE)
    #save_path=gsam_annot_path.replace(ds_split, 'comparison')) # path or None
    save_path=None
    
    plot_compare_annotations(img=img, 
                             masks_manual=masks_manual, 
                             masks_gsam=masks_gsam, 
                             label2id={v: k for k, v in cls_dict.items()}, 
                             alpha_img=0.7,
                             save_path=save_path)