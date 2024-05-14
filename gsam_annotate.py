import cv2
import numpy as np
import os
import warnings

from tqdm import tqdm

from utils.common import gsam_paths
from utils.gsam_utils import initialize, run_gdino, run_sam


# Settings
warnings.filterwarnings("ignore")
input_dir = gsam_paths['input_dir']
out_dir = gsam_paths['output_dir']

BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2
NMS_THRESHOLD = 0.8
SAVE_OUTPUTS_BOXES = False
SAVE_OUTPUTS_MASKS = True
SAVE_OUTPUTS_MASKS_RGB = False

logfile = "zero_detections.txt"
errorfile = "errors.txt"

# Model
grounding_dino_model, sam_predictor, box_annotator, mask_annotator, label_annotator = initialize(input_dir=input_dir, out_dir=out_dir, use_sam_hq=False)

for root, dirs, files in os.walk(input_dir):
    for images_dir in dirs:

        # Skip existing annotations
        image_list = os.listdir(root+images_dir)
        annotated_list = os.listdir(out_dir+'masks/'+images_dir)
        to_remove = [fname.replace('_mask.png','.jpg') for fname in annotated_list]
        to_annotate = list(set(image_list) - set(to_remove)) # order of list may have changed
        [print(f"{root}{images_dir}/{img_file}") for img_file in to_annotate]

        # Annotate
        zero_acc = 0
        pbar = tqdm(to_annotate)
        for img_file in pbar:
            try:
                SOURCE_IMAGE_PATH = f"{root}{images_dir}/{img_file}"

                #print(f"\n\n{SOURCE_IMAGE_PATH}")
                image = cv2.imread(SOURCE_IMAGE_PATH)

                # predictions
                detections, frame_boxes = run_gdino(grounding_dino_model, image, BOX_THRESHOLD, TEXT_THRESHOLD, box_annotator)
                class_ids, masks, frame_masks = run_sam(sam_predictor, detections, NMS_THRESHOLD, image, box_annotator, mask_annotator, label_annotator)

                # if no objects detected, skip and log file name 
                if len(class_ids)==0:
                    with open(out_dir+logfile, 'a') as f:
                        f.write(f"{SOURCE_IMAGE_PATH}\n")
                    zero_acc += 1
                    print(f"Number of images with no detections in this session: {zero_acc}. Logging file name. Skipping...")
                    continue

                # postproc masks and save for training
                masks_by_id = np.array([class_id*mask for class_id, mask in zip(class_ids, masks)]).max(axis=0)

                if SAVE_OUTPUTS_MASKS:
                    cv2.imwrite(f"{out_dir}masks/{images_dir}/{img_file[:-4]}_mask.png", masks_by_id)

                if SAVE_OUTPUTS_MASKS_RGB:
                    cv2.imwrite(f"{out_dir}masksRGB/{img_file[:-4]}_mask.png", frame_masks)
                
                if SAVE_OUTPUTS_BOXES:
                    cv2.imwrite(f"{out_dir}boxes/{img_file[:-4]}_boxes.jpg", frame_boxes)

            except Exception as e:
                with open(out_dir+errorfile, 'a') as f:
                    f.write(f"{SOURCE_IMAGE_PATH}\n")
                
                print(f'Error encountered for file {SOURCE_IMAGE_PATH}.\nLogging file name. Skipping...\nError:\n {e}')
                continue
