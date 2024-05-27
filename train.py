import datetime
import multiprocessing
import os
import torch
import uuid
import warnings

from torch.utils.data import DataLoader, Subset
from torchmetrics import JaccardIndex, Accuracy
from tqdm import tqdm
from transformers import SegformerImageProcessor

from configs.train_cfg import default_cfg
from dataset.dataset_utils import cls_dict
from dataset.disaster_dataset import DisasterSegDataset
from models.load_models import load_segformer
from utils.common import common_paths
from utils.early_stopper import EarlyStopper
from utils.misc import initialize_session, save_checkpoint, log_metrics, log_train_config, delete_old_checkpoint
from utils.visualization import plot_curves


warnings.filterwarnings("ignore", message='Palette images with Transparency expressed in bytes should be converted to RGBA images')


def main():
    cfg = default_cfg
    if cfg['expand_pytorch_alloc_mem']:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    num_cpus = min(4, multiprocessing.cpu_count())
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg['is_office'] else "cpu")

    # Create directories for this session
    session_id = '{date:%Y-%m-%d}__'.format(date=datetime.datetime.now()) + uuid.uuid4().hex
    log_dir = common_paths['train_runs'] + session_id
    initialize_session(local_dir=common_paths["train_runs"], session_id=session_id)

    # Datasets
    preprocessor = SegformerImageProcessor()

    train_dataset = DisasterSegDataset(
        root_dir=common_paths["dataset_root"],
        preprocessor=preprocessor,
        train=True,
        augment_data=cfg['use_augmentation'],
    )
    valid_dataset = DisasterSegDataset(
        root_dir=common_paths["dataset_root"],
        preprocessor=preprocessor,
        train=False,
        augment_data=False
    )

    # Use subsets for testing
    if cfg['is_office']:
        train_dataset = Subset(train_dataset, torch.arange(0,6))
        valid_dataset = Subset(train_dataset, torch.arange(0,2))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=num_cpus, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], num_workers=num_cpus)
   
    # Model
    model = load_segformer(config_path=common_paths['segformer_config_path'], 
                           id2label=cls_dict, 
                           label2id={v: k for k, v in cls_dict.items()}, 
                           checkpoint_file=common_paths['segformer_pretrained_weights_path'])
    model.to(device) 

    # Initialize metrics
    train_jaccard = JaccardIndex(task='multiclass', num_classes=len(cls_dict), ignore_index=0, average='macro').to(device)
    train_pw_acc = Accuracy(task="multiclass", num_classes=len(cls_dict), ignore_index=0).to(device)
    val_jaccard = JaccardIndex(task='multiclass', num_classes=len(cls_dict), ignore_index=0, average='macro').to(device) 
    val_pw_acc = Accuracy(task="multiclass", num_classes=len(cls_dict), ignore_index=0).to(device)
    best_val_metric = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    early_stopper = EarlyStopper(patience=cfg['early_stop_patience'], min_delta=cfg['early_stop_min_delta'])

    # Continue training or start from scratch
    """ may be necessary once training is conducted with more data
    if cfg['continue']:
        start_epoch = ...
        ...
    else:
        start_epoch = 1
    """
    start_epoch = 1

    log_train_config(log_dir=log_dir,
                     cfg={"Session ID": session_id,
                          "Dataset": common_paths['dataset_root'],
                          "Train Dataset Size": len(train_dataset),
                          "Val Dataset Size": len(valid_dataset),
                          "Augment Training Data": cfg['use_augmentation'],
                          "Batch Size": cfg['batch_size'],
                          "Num Workers": num_cpus,
                          "Using Device": str(device),
                          "Max Epochs": cfg['max_epochs'],
                          "Learning Rate": cfg['lr'],
                          "Early Stop Delta": cfg['early_stop_min_delta'], 
                          "Early Stop Patience": cfg['early_stop_patience'],
                          })

    # Iterate through epochs
    epoch_counter = 1
    for epoch in range(start_epoch, cfg['max_epochs'] + 1):
        print("Epoch:", epoch, "/", cfg['max_epochs'])
        pbar = tqdm(
            train_dataloader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        # Loss for this epoch
        train_losses, val_losses =  [], []

        model.train()

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Zero Parameter Gradients + Forward
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)

            # Get Predictions TODO: Review upsamling: Could have an impact on sharpness of predictions
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)

            # Get train metrics
            train_loss_batch = outputs.loss
            train_losses.append(train_loss_batch.item())
            train_jaccard(predicted, labels)
            train_pw_acc(predicted, labels)

            # Backward + Update Params
            train_loss_batch.backward()
            optimizer.step()

        else:
            model.eval()

            pbar = tqdm(valid_dataloader,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                        )

            with torch.no_grad():

                for batch in pbar:
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(pixel_values=pixel_values, labels=labels)
                    upsampled_logits = torch.nn.functional.interpolate(
                        outputs.logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    predicted = upsampled_logits.argmax(dim=1)

                    # Get val metrics for one batch
                    val_loss_batch = outputs.loss
                    val_losses.append(val_loss_batch.item()) 
                    val_jaccard(predicted, labels)
                    val_pw_acc(predicted, labels)

        # Mean performance for this epoch   
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        train_accuracy = train_pw_acc.compute().item()
        val_accuracy = val_pw_acc.compute().item()
        train_miou = train_jaccard.compute().item()
        val_miou = val_jaccard.compute().item()

        print(
            f"Train Loss: {train_loss:.4f} \
            Train Pixelwise Accuracy: {train_accuracy:.4f} \
            Train MIoU: {train_miou:.4f} \
            Val Loss: {val_loss:.4f} \
            Val Pixelwise Accuracy: {val_accuracy:.4f} \
            Val MIoU: {val_miou:.4f}"
        )
    
        log_metrics(epoch, train_accuracy, train_loss, train_miou, val_accuracy, val_loss, val_miou,
                            logs_path=log_dir + "/logs.csv")
        
        # Update checkpoints
        if (epoch_counter == cfg["update_checkpoint_frequency"]):
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(),
                            checkpoint_path=log_dir + f"/checkpoints/last__epoch-{epoch}__acc-{val_accuracy:.3f}__miou-{val_miou:.3f}.pt")
            delete_old_checkpoint(type="last", checkpoint_dir=log_dir + "/checkpoints/")
            epoch_counter = 0

        if val_miou > best_val_metric:
            best_val_metric = val_miou
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(),
                            checkpoint_path=log_dir + f"/checkpoints/best__epoch-{epoch}__acc-{val_accuracy:.3f}__miou-{val_miou:.3f}.pt")
            delete_old_checkpoint(type="best", checkpoint_dir=log_dir + "/checkpoints/")

        # Reset metrics for next epoch    
        train_jaccard.reset()
        val_jaccard.reset()
        train_pw_acc.reset()
        val_pw_acc.reset()

        epoch_counter += 1

        if early_stopper.early_stop(val_miou):
            break

    # Finished training
    plot_curves(log_dir, log_file="/logs.csv")

if __name__ == "__main__":
    t_start = datetime.datetime.now()
    main()
    t_train = datetime.datetime.now() - t_start
    print(f"Training took {(t_train.total_seconds() / 3600):.2f} hours.")