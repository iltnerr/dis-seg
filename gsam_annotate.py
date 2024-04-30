import os
import cv2
import numpy as np
from tqdm import tqdm

from utils.common import gsam_paths
from utils.gsam_utils import initialize, run_gdino, run_sam


# Settings
images_dir = gsam_paths['input_dir']
out_dir = gsam_paths['output_dir']

BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2
NMS_THRESHOLD = 0.8
SAVE_OUTPUTS_BOXES = False
SAVE_OUTPUTS_MASKS = True
SAVE_OUTPUTS_MASKS_RGB = False

grounding_dino_model, sam_predictor, box_annotator, mask_annotator, label_annotator = initialize(use_sam_hq=False)

image_list = os.listdir(images_dir)
pbar = tqdm(image_list)
for img_file in pbar:
    SOURCE_IMAGE_PATH = f"{images_dir}{img_file}"

    print(f"\n\n{SOURCE_IMAGE_PATH}")
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # predictions
    detections, frame_boxes = run_gdino(grounding_dino_model, image, BOX_THRESHOLD, TEXT_THRESHOLD, box_annotator)
    class_ids, masks, frame_masks = run_sam(sam_predictor, detections, NMS_THRESHOLD, image, box_annotator, mask_annotator, label_annotator)

    masks_by_id = np.array([class_id*mask for class_id, mask in zip(class_ids, masks)]).max(axis=0)

    if SAVE_OUTPUTS_MASKS:
        cv2.imwrite(f"{out_dir}masks/{img_file[:-4]}_mask.png", masks_by_id)

    if SAVE_OUTPUTS_MASKS_RGB:
        cv2.imwrite(f"{out_dir}masksRGB/{img_file[:-4]}_mask.png", frame_masks)
    
    if SAVE_OUTPUTS_BOXES:
        cv2.imwrite(f"{out_dir}boxes/{img_file[:-4]}_boxes.jpg", frame_boxes)
