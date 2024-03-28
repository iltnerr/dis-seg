import numpy as np
import cv2
import csv
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def ade_palette():
    """Custom palette that maps each class to BGR values.
    TODO: uniform distribution in color space depending on the number of classes. 
    """
    return [
        [0, 0, 0],      # background (BLAK)
        [255, 0, 0],    # Damaged_buildings (BLUE)
        [0, 255, 0],    # Damaged_road (GREEN)
        [0, 0, 255],    # Damaged_vehicle (RED)
        [255, 255, 0],  # Debris (YELLOW)
        [255, 0, 255],  # Fire (MAGENTA)
        [0, 255, 255],  # Flood (CYAN)
        [128, 0, 0],    # Injured_animal (MAROON)
        [0, 128, 0],    # Injured_person (OLIVE)
        [128, 128, 128],# Sky_background (GRAY)
        [128, 0, 128],  # Smoke (PURPLE)
        [0, 128, 128],  # Trees (TEAL)
        [0, 0, 128]     # Uninjured_person (NAVY)
    ]

def normalize_color(color):
    # Convert BGR to RGB and normalize to the range 0-1
    return tuple(component / 255.0 for component in reversed(color))

def plot_result_gt(img, seg, cls_dict, infer_cfg, model):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    
    map = seg
    classes_map = np.unique(map).tolist()
    unique_classes = [model.config.id2label[idx] if idx != 255 else None for idx in classes_map]

    id2label = dict(cls_dict)
    label2id = {v: k for k, v in cls_dict.items()}

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
    palette = np.array(ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    img = np.array(img) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    ax1.imshow(img)
    ax1.set_title('Original image an model\'s prediction Overlay', fontsize=13)  # Add title to the first subplot

    legend_elements = [Patch(facecolor=(clr[::-1] / 255), label=cls_name) for [cls_name, clr] in zip(label2id.keys(), palette)]
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    # Create a text box with class information
    class_info = "Detected classes in this image:\n" + "\n".join([f"{cls}: {cls_id}" for cls, cls_id in zip(unique_classes, classes_map)])
    ax1.text(-0.9, 0.0, class_info, fontsize=9, color='white', transform=ax1.transAxes, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.7))

    # Retrieve input_path from infer_cfg
    input_path = infer_cfg['input_path']

    # Generate seg_map_path from input_path
    seg_map_path = input_path.replace("images", "annotations").replace(".jpg", "_mask.png")

    # Load the grayscale segmentation map
    seg_map = cv2.imread(seg_map_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply color mapping to the segmentation map
    colored_seg_map = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
   
    for class_idx, color in enumerate(palette):
        colored_seg_map[seg_map == class_idx] = color
        
    # Display the colored segmentation map using Matplotlib
    ax2.imshow(cv2.cvtColor(colored_seg_map, cv2.COLOR_BGR2RGB))
    ax2.axis('off')  # Turn off axis labels and ticks
    ax2.set_title('Ground truth annotation', fontsize=13)  # Add title to the second subplot

    plt.show()    

def plot_result(img, seg, cls_dict, model):
    
    map = seg
    classes_map = np.unique(map).tolist()
    unique_classes = [model.config.id2label[idx] if idx!=255 else None for idx in classes_map]
    print("Classes in this image:", unique_classes, classes_map)
    id2label = dict(cls_dict)
    label2id = {v: k for k, v in cls_dict.items()}

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
    palette = np.array(ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    img = np.array(img) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)

    legend_elements = [Patch(facecolor=(clr[::-1] / 255), label=cls_name) for [cls_name, clr] in zip(label2id.keys(), palette)]
    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.30, 0.9))

    #Create a text box with class information
    class_info = "Detected classes in this image:\n" + "\n".join([f"{cls}: {cls_id}" for cls, cls_id in zip(unique_classes, classes_map)])
    plt.text(-0.5, 0.0, class_info, fontsize=12, color='white', transform=plt.gca().transAxes, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.7))

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