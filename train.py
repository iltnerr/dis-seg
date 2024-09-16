import datetime
import multiprocessing
import os
import torch
import uuid
import warnings

from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader, Subset
from torchmetrics import JaccardIndex
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

    num_workers = min(4, multiprocessing.cpu_count())
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg['is_office'] else "cpu")
    session_id = '{date:%Y-%m-%d}__'.format(date=datetime.datetime.now()) + uuid.uuid4().hex
    log_dir = common_paths['train_runs'] + session_id
    initialize_session(local_dir=common_paths["train_runs"], session_id=session_id)

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
        valid_dataset = Subset(valid_dataset, torch.arange(0,2))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=num_workers, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], num_workers=num_workers)
   
    model = load_segformer(
        id2label=cls_dict, 
        label2id={v: k for k, v in cls_dict.items()},
        checkpoint_file=None
    )

    total_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Learnable Params: {total_learnable_params}")

    model.to(device) 

    train_jaccard = JaccardIndex(task='multiclass', num_classes=len(cls_dict), ignore_index=0, average='macro').to(device)
    val_jaccard = JaccardIndex(task='multiclass', num_classes=len(cls_dict), ignore_index=0, average='macro').to(device) 
    best_val_metric = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    early_stopper = EarlyStopper(patience=cfg['early_stop_patience'], min_delta=cfg['early_stop_min_delta'])
    
    # lr schedule: linear warmup, then decay
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=cfg['warmup_interval']),
            CosineAnnealingLR(optimizer, T_max=cfg['max_epochs'])
        ],
        milestones=[cfg['warmup_interval']]
    )
    
    log_train_config(
        log_dir=log_dir,
        cfg={
            "Session ID": session_id,
            "Dataset": common_paths['dataset_root'],
            "Train Dataset Size": len(train_dataset),
            "Val Dataset Size": len(valid_dataset),
            "Augment Training Data": cfg['use_augmentation'],
            "Batch Size": cfg['batch_size'],
            "Num Workers": num_workers,
            "Using Device": str(device),
            "Max Epochs": cfg['max_epochs'],
            "Learning Rate": cfg['lr'],
            "Early Stop Delta": cfg['early_stop_min_delta'], 
            "Early Stop Patience": cfg['early_stop_patience'],
        }
    )

    stats = {k: [] for k in ['epoch', 'lr', 'train_loss', 'train_miou', 'val_loss', 'val_miou']}
    
    # Train loop
    for epoch in range(1, cfg['max_epochs'] + 1):
        pbar = tqdm(
            train_dataloader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        train_losses, val_losses = [], []

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

            # Batch stats
            train_loss_batch = outputs.loss
            train_losses.append(train_loss_batch.item())
            train_jaccard(predicted, labels)

            # Backward + Update
            train_loss_batch.backward()
            optimizer.step()

        else:
            model.eval()

            pbar = tqdm(
                valid_dataloader,
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

        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Collect stats for this epoch   
        stats['epoch'] = epoch
        stats['lr'] = lr
        stats['train_miou'] = train_jaccard.compute().item()
        stats['val_miou'] = val_jaccard.compute().item()
        stats['train_loss'] = sum(train_losses) / len(train_losses)
        stats['val_loss'] = sum(val_losses) / len(val_losses)

        print(stats)
        log_metrics(stats, logs_path=log_dir + "/logs.csv")
        
        # Update checkpoints
        if epoch % cfg["update_checkpoint_frequency"] == 0:
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(),
                            checkpoint_path=log_dir + f"/checkpoints/last__epoch-{epoch}__miou-{stats['val_miou']:.3f}.pt")
            delete_old_checkpoint(type="last", checkpoint_dir=log_dir + "/checkpoints/")

        if stats['val_miou'] > best_val_metric:
            best_val_metric = stats['val_miou']
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(),
                            checkpoint_path=log_dir + f"/checkpoints/best__epoch-{epoch}__miou-{stats['val_miou']:.3f}.pt")
            delete_old_checkpoint(type="best", checkpoint_dir=log_dir + "/checkpoints/")

        # Reset metrics for next epoch    
        train_jaccard.reset()
        val_jaccard.reset()

        if early_stopper.early_stop(stats['val_miou']):
            break

    # Finished training
    plot_curves(log_dir, log_file="logs.csv")

if __name__ == "__main__":
    t_start = datetime.datetime.now()
    main()
    t_train = datetime.datetime.now() - t_start
    print(f"Training took {(t_train.total_seconds() / 3600):.2f} hours.")