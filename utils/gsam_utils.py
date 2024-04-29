import numpy as np
import torch
import torchvision
import seaborn as sns
import supervision as sv
import cv2

from utils.common import gsam_paths

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor


cls_dict = {
    0: 'Background',
    1: 'Building',
    2: 'Road',
    3: 'Vehicle',
    4: 'Debris',
    5: 'Fire',
    6: 'Water',
    7: 'Animal',
    8: 'Injured_Person', # Keeping this class for convenience, since it is part of the annotations. With a small dataset it does not make sense to distinguish healthy and injured persons, however.
    9: 'Sky',
    10: 'Smoke',
    11: 'Tree',
    12: 'Person'
}

CLASSES = ["Background", "Building", "Road", "Vehicle", "Debris", "Fire", "Water", "Animal", "Snow", # snow==injured_person
           "Sky", "Smoke", "Tree", "Person"]
 
palette = sns.color_palette("husl", len(CLASSES))
colors_rgb =[(r, g, b) for r, g, b in palette]
CLASS_COLORS = dict(zip(CLASSES, colors_rgb)) # unique, distinct class colors (r,g,b,a)
CLASS_COLORS = [
        [0, 0, 0],     
        [255, 0, 0],    
        [0, 255, 0],    
        [0, 0, 255],   
        [255, 255, 0],  
        [255, 0, 255],  
        [0, 255, 255],  
        [128, 0, 0],    
        [0, 128, 0],    
        [128, 128, 128],
        [128, 0, 128],  
        [0, 128, 128], 
        [0, 0, 128]     
    ]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize(use_sam_hq=True):
    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=gsam_paths['gdino_config'], model_checkpoint_path=gsam_paths['gdino_ckpt'])

    # Building SAM Model and SAM Predictor
    if use_sam_hq:
        sam = sam_hq_model_registry["vit_h"](checkpoint=gsam_paths['samhq_ckpt'])
    else:
        sam = sam_model_registry["vit_h"](checkpoint=gsam_paths['sam_ckpt'])
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette(colors=[sv.Color(r=b, g=g, b=r) for r, g, b in CLASS_COLORS]))
    mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette(colors=[sv.Color(r=b, g=g, b=r) for r, g, b in CLASS_COLORS])) 
    return grounding_dino_model, sam_predictor, box_annotator, mask_annotator

def run_gdino(grounding_dino_model, image, BOX_THRESHOLD, TEXT_THRESHOLD, box_annotator, verbose=False):
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # annotate image with detections
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _ 
        in detections]
    if verbose:
        print(f"box labels: {labels}")
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    return detections, annotated_frame

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    # Prompting SAM with detected boxes
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def run_sam(sam_predictor, detections, NMS_THRESHOLD, image, box_annotator, mask_annotator, show_boxes=False, verbose=False):
    # NMS post process
    if verbose:
        print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    if verbose:
        print(f"After NMS: {len(detections.xyxy)} boxes")

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections    
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _
        in detections]
    
    if verbose:
        print(f"mask labels: {labels}")
    
    image = np.zeros(image.shape, np.uint8)
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    
    if show_boxes:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER, color=sv.ColorPalette(colors=[sv.Color(r=b, g=g, b=r) for r, g, b in CLASS_COLORS]))
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return detections.class_id, detections.mask, annotated_image
