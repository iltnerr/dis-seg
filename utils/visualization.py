import numpy as np
import cv2
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def ade_palette():
    """
    Color palette that maps each class to BGR values.

    Note on 'Injured_Person' and 'Person' classes:
    Originally, injured and healthy people were distinguished and annotated as different classes. 
    However, with a too small dataset, this does not work out. Therefore, these are considered as same classes for now (same color in visualizations).
    
    TODO: color mapping should be integrated into the cls_dict. Also, uniform distribution in color space depending on the number of classes. 
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
        #[0, 128, 0],    # Injured_Person (OLIVE)
        [0, 0, 128],    # Injured_Person (Same color as 'Person', see above)
        [128, 128, 128],# Sky (GRAY)
        [128, 0, 128],  # Smoke (PURPLE)
        [0, 128, 128],  # Tree (TEAL)
        [0, 0, 128]     # Person (NAVY)
    ]

def plot_compare_predictions(img, preds, label2id, gt_map=None, alpha_img=0.5):

    # Colorize segmentations   
    colored_preds = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.uint8)
    colored_gt = np.zeros((img.height, img.width, 3), dtype=np.uint8)

    palette = np.array(ade_palette())
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
    legend_elements = [Patch(facecolor=(color[::-1] / 255), label=cls_name) for [cls_name, color] in zip(label2id.keys(), palette)]
    del legend_elements[8] # remove entry for 'Injured_Person' class
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