import numpy as np
from tabulate import tabulate


class IoUTable:

    def __init__(self, cfg, cls_dict):
        self.cfg = cfg

        self.id2label = dict(cls_dict)
        self.label2id = {v: k for k, v in cls_dict.items()}

        self.iou_per_class_epoch = {class_name: [] for class_name in self.id2label.values()}
        self.mean_iou = None

    def get_iou_per_class_epoch(self, predicted, labels):
        for class_id, class_name in enumerate(self.id2label.values()):
            class_iou = calculate_iou(predicted.cpu(), labels.cpu(), class_id)
            self.iou_per_class_epoch[class_name].append(class_iou)

    def update_data(self):
        self.iou_table_data = []

        for class_name in self.id2label.values():
            # Calculate the average IoU for the current epoch
            average_iou = np.mean(self.iou_per_class_epoch[class_name])
            self.iou_table_data.append([class_name, average_iou])

        # Calculate MIoU
        miou_values = [iou[1] for iou in self.iou_table_data]
        self.mean_iou = np.mean(miou_values)

        #print(tabulate(self.iou_table_data, headers=["Class", "IoU per class"], tablefmt="grid"))

        # Reset per class IoU values for the next epoch
        self.iou_per_class_epoch = {class_name: [] for class_name in self.id2label.values()}


def calculate_iou(pred, target, class_id):
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    iou = 0.0 if union == 0 else intersection / union

    return iou  