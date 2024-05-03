import numpy as np
import cv2
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def color_palette():
    """
    Color palette that maps each class to BGR values.
    
    TODO: uniform distribution in color space depending on cls_dict:
    CLASSES = list(cls_dict.values())
    palette = sns.color_palette("husl", len(CLASSES))
    colors_rgb = [(r, g, b) for r, g, b in palette]
    CLASS_COLORS = dict(zip(CLASSES, colors_rgb)) # unique, distinct (r,g,b) class colors 
    """

    return [
        [0, 0, 0],      # Background (BLACK)
        [255, 0, 0],    # Building (BLUE)
        [0, 255, 0],    # Road (GREEN)
        [0, 0, 255],    # Vehicle (RED)
        [255, 255, 0],  # Debris (YELLOW)
        [255, 0, 255],  # Fire (MAGENTA)
        [0, 255, 255],  # Water (CYAN)
        [128, 0, 0],    # Animal (MAROON)
        [128, 128, 128],# Sky (GRAY)
        [128, 0, 128],  # Smoke (PURPLE)
        [0, 128, 128],  # Tree (TEAL)
        [0, 0, 128]     # Person (NAVY)
    ]

def plot_compare_predictions(img, preds, label2id, gt_map=None, alpha_img=0.5):

    # Colorize segmentations   
    colored_preds = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.uint8)
    colored_gt = np.zeros((img.height, img.width, 3), dtype=np.uint8)

    palette = np.array(color_palette())
    for label, color in enumerate(palette):
        colored_preds[preds == label] = color 
        if gt_map is not None:
            colored_gt[gt_map == label] = color

    colored_preds = cv2.cvtColor(colored_preds, cv2.COLOR_BGR2RGB)
    colored_gt = cv2.cvtColor(colored_gt, cv2.COLOR_BGR2RGB)

    # Blend prediction with image
    blend_img = np.array(img) * alpha_img + colored_preds * (1-alpha_img)
    blend_img = blend_img.astype(np.uint8)

    # Figure
    fig = plt.figure(constrained_layout=True, figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[4,1,4])

    axes = [fig.add_subplot(gs[0, 0]), # Image
            fig.add_subplot(gs[0, 1]), # Overlay
            fig.add_subplot(gs[2, 0]), # GT
            fig.add_subplot(gs[2, 1]), # Pred
            fig.add_subplot(gs[1, :])  # Legend
    ]
    
    plt_cfg = {0: {'img': img, 
                   'title': 'Image'},
               1: {'img': blend_img,
                   'title': 'Image + Predictions'},
               2: {'img': colored_gt,
                   'title': 'GT Annotation'},
               3: {'img': colored_preds,
                   'title': 'Predictions'}
                   } 
    
    # 4 Images
    for idx in range(len(axes)-1):
        cfg = plt_cfg[idx]
        axes[idx].imshow(cfg['img'])
        axes[idx].set_title(cfg['title'], fontsize=16)
        axes[idx].axis('off')

    # Legend
    legend_elements = create_legend(label2id=label2id, palette=palette)
    axes[4].legend(handles=legend_elements, ncol=4, loc='center', bbox_to_anchor=(0.5, 0.5))
    axes[4].axis('off')
    plt.show()    

def plot_curves(log_dir, log_file):
    colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])

    # Settings
    SMALL_SIZE = 8
    BIGGER_SIZE = 10
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE, direction='out')
    plt.rc('ytick', labelsize=BIGGER_SIZE, direction='out')
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('lines', linewidth=2)
    plt.rcParams['axes.linewidth'] = 0.1    
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='#000000',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('patch', edgecolor='#E6E6E6')

    # Get data
    df = pd.read_csv(log_dir+log_file, index_col="Epoch")
    df_loss = df[[col for col in df.columns if "Loss" in col]]
    df_acc = df[[col for col in df.columns if "Acc" in col]]
    df_miou = df[[col for col in df.columns if "IoU" in col]]
    df_list = [df_loss, df_acc, df_miou]

    cfg_axes = {0: {"ylabel": "Loss", "ylimits": None},
                1: {"ylabel": "Pixelwise Acc.", "ylimits": [0, 1]},
                2: {"ylabel": "MIoU", "ylimits": [0, 1]}
    }

    # Plots
    fig, axes = plt.subplots(3)
    fig.suptitle("Learning Curves")

    for idx in range(0, 3):
        df_list[idx].plot(ax=axes[idx])
        axes[idx].legend()
        axes[idx].set_ylabel(cfg_axes[idx]["ylabel"])
        axes[idx].set_ylim(cfg_axes[idx]["ylimits"])
        axes[idx].set_xlabel("")

    plt.xlabel("Epoch")
    plt.subplots_adjust(hspace=0.5)   
    plt.savefig(log_dir + "/learning_curves.pdf")

def plot_compare_annotations(img, masks_manual, masks_gsam, label2id, alpha_img=0.5, save_path=None):
    """
    Compare annotations of grounded-sam with manual annotations.
    """

    # Colorize segmentations   
    colored_gsam = np.zeros((img.height, img.width, 3), dtype=np.uint8)
    colored_manual = np.zeros((img.height, img.width, 3), dtype=np.uint8)

    palette = np.array(color_palette())
    for label, color in enumerate(palette):
        colored_gsam[masks_gsam == label] = color 
        colored_manual[masks_manual == label] = color

    colored_gsam = cv2.cvtColor(colored_gsam, cv2.COLOR_BGR2RGB)
    colored_manual = cv2.cvtColor(colored_manual, cv2.COLOR_BGR2RGB)

    # Blend image with annotations
    blend_gsam = np.array(img) * alpha_img + colored_gsam * (1-alpha_img)
    blend_gsam = blend_gsam.astype(np.uint8)

    blend_manual = np.array(img) * alpha_img + colored_manual * (1-alpha_img)
    blend_manual = blend_manual.astype(np.uint8) 

    # Figure
    fig = plt.figure(constrained_layout=True, figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[4,1,4])

    axes = [fig.add_subplot(gs[0, 0]), # Blend Manual
            fig.add_subplot(gs[0, 1]), # Blend GSAM
            fig.add_subplot(gs[2, 0]), # Manual Annot.
            fig.add_subplot(gs[2, 1]), # GSAM Annot. 
            fig.add_subplot(gs[1, :])  # Legend
    ]
    
    plt_cfg = {0: {'img': blend_manual, 
                   'title': 'Image + Manual Annotation'},
               1: {'img': blend_gsam,
                   'title': 'Image + GSAM Annotation'},
               2: {'img': colored_manual,
                   'title': 'Manual Masks'},
               3: {'img': colored_gsam,
                   'title': 'GSAM Masks'}
                   } 
    
    # 4 Images
    for idx in range(len(axes)-1):
        cfg = plt_cfg[idx]
        axes[idx].imshow(cfg['img'])
        axes[idx].set_title(cfg['title'], fontsize=16)
        axes[idx].axis('off')

    # Legend
    legend_elements = create_legend(label2id=label2id, palette=palette)
    axes[4].legend(handles=legend_elements, ncol=4, loc='center', bbox_to_anchor=(0.5, 0.5))
    axes[4].axis('off')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()  

    plt.close()

def visualize_augmentations(img, masks, aug_image, aug_masks, label2id, alpha_img=0.5):

    # Colorize segmentations   
    colored_labels = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
    colored_aug_labels = np.zeros((aug_masks.shape[0], aug_masks.shape[1], 3), dtype=np.uint8)

    palette = np.array(color_palette())
    for label, color in enumerate(palette):
        colored_labels[masks == label] = color 
        colored_aug_labels[aug_masks == label] = color 

    colored_labels = cv2.cvtColor(colored_labels, cv2.COLOR_BGR2RGB)
    colored_aug_labels = cv2.cvtColor(colored_aug_labels, cv2.COLOR_BGR2RGB)

    # Blend image with annotations
    blend = np.array(img) * alpha_img + colored_labels * (1-alpha_img)
    blend = blend.astype(np.uint8)

    blend_aug = np.array(aug_image) * alpha_img + colored_aug_labels * (1-alpha_img)
    blend_aug = blend_aug.astype(np.uint8) 

    # Figure
    fig = plt.figure(constrained_layout=True, figsize=(12, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[4,1,4])

    axes = [fig.add_subplot(gs[0, 0]), # Image
            fig.add_subplot(gs[0, 1]), # Masks
            fig.add_subplot(gs[0, 2]), # Blend
            fig.add_subplot(gs[2, 0]), # Augmented Image
            fig.add_subplot(gs[2, 1]), # Augmented Masks
            fig.add_subplot(gs[2, 2]), # Aug Blend 
            fig.add_subplot(gs[1, :])  # Legend
    ]
    
    plt_cfg = {0: {'img': img, 
                   'title': 'Image'},
               1: {'img': colored_labels,
                   'title': 'Masks'},
               2: {'img': blend,
                   'title': 'Blend Masks'},
               3: {'img': aug_image,
                   'title': 'Augmented Image'},
               4: {'img': colored_aug_labels,
                   'title': 'Augmented Masks'},
               5: {'img': blend_aug,
                   'title': 'Blend Augmented Masks'}
                   } 
    
    # 4 Images
    for idx in range(len(axes)-1):
        cfg = plt_cfg[idx]
        axes[idx].imshow(cfg['img'])
        axes[idx].set_title(cfg['title'], fontsize=16)
        axes[idx].axis('off')

    # Legend
    legend_elements = create_legend(label2id=label2id, palette=palette)
    axes[6].legend(handles=legend_elements, ncol=4, loc='center', bbox_to_anchor=(0.5, 0.5))
    axes[6].axis('off')
    plt.show()

def create_legend(label2id, palette):
    return [Patch(facecolor=(color[::-1] / 255), label=cls_name) for [cls_name, color] in zip(label2id.keys(), palette)]

