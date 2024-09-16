import torch
import os
import csv
import numpy as np


def initialize_session(local_dir, session_id):
    log_dir = local_dir + session_id
    ckpt_dir = log_dir + "/checkpoints/"

    print("\n\n=================")
    print("Initializing directories:\n")
    for directory in [local_dir, log_dir, ckpt_dir]:
        print(directory)
        os.makedirs(directory, exist_ok=True)
    print("=================\n\n")
    
    init_stats = {k: 0 for k in ['epoch', 'lr', 'train_loss', 'train_miou', 'val_loss', 'val_miou']}
    log_metrics(init_stats, logs_path=log_dir + "/logs.csv")

def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, checkpoint_path):   
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        }, 
        checkpoint_path)

def log_metrics(stats, logs_path):
    header = stats.keys()

    # Append metrics for the current epoch
    with open(logs_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if stats['epoch'] == 0:
            writer.writerow(header)
        else:
            writer.writerow(stats.values())

def log_train_config(log_dir, cfg):
    content = [f"{k}: {v}" for k, v in cfg.items()]
        
    print("\n\n=================")
    with open(log_dir+"/train_config.txt", mode='a') as file:
        for row in content:
            file.write(row+"\n")
            print(row)
    print("=================\n\n")

def delete_old_checkpoint(type, checkpoint_dir):
    allfiles = os.listdir(checkpoint_dir)
    relevant = [file for file in allfiles if file.startswith(type)]
    
    # Reduce filenames to number of epoch so that the old checkpoint can be identified
    epochs = [file.split("epoch-")[1] for file in relevant]
    epochs = [int(file.split("__miou")[0]) for file in epochs]

    if len(epochs) > 1:
        # Remove old checkpoint
        to_delete = relevant[np.argmin(epochs)]
        os.remove(checkpoint_dir + to_delete)