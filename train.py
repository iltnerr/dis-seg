import torch
import datetime
import uuid
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from transformers import SegformerImageProcessor
from models.load_models import load_segformer

from utils.common import common_paths
from utils.misc import initialize_session, save_checkpoint, log_metrics, log_train_config, delete_old_checkpoint
from utils.visualization import plot_curves
from utils.early_stopper import EarlyStopper

from dataset.dataset_utils import cls_dict, get_augmentations
from dataset.disaster_dataset import DisasterSegDataset

from sklearn.metrics import accuracy_score
from torchmetrics import JaccardIndex

from configs.train_cfg import default_cfg


def main():
    cfg = default_cfg
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg['is_office'] else "cpu")

    # Create directories for this session
    session_id = '{date:%Y-%m-%d}__'.format(date=datetime.datetime.now()) + uuid.uuid4().hex
    log_dir = common_paths['train_runs'] + session_id
    initialize_session(local_dir=common_paths["train_runs"], session_id=session_id)

    # Augmentations
    train_transforms = get_augmentations(mode='train')
    val_transforms = None

    # Datasets
    preprocessor = SegformerImageProcessor()

    train_dataset = DisasterSegDataset(
        root_dir=common_paths["dataset_root"],
        preprocessor=preprocessor,
        train=True,
        transforms=train_transforms,
    )
    valid_dataset = DisasterSegDataset(
        root_dir=common_paths["dataset_root"],
        preprocessor=preprocessor,
        train=False,
        transforms=val_transforms,
    )

    # Use subsets for testing
    if cfg['is_office']:
        train_dataset = Subset(train_dataset, torch.arange(0,6))
        valid_dataset = Subset(train_dataset, torch.arange(0,2))

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'])

    # Model
    id2label = dict(cls_dict)
    label2id = {v: k for k, v in cls_dict.items()}

    # Use pretrained segformer model 
    model = load_segformer(config_path=cfg['segformer_config_path'], id2label=id2label, label2id=label2id, checkpoint_file=cfg['segformer_pretrained_weights_path'])
    model.to(device) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    early_stopper = EarlyStopper(patience=cfg['early_stop_patience'], min_delta=cfg['early_stop_min_delta'])

    train_jaccard = JaccardIndex(task='multiclass', num_classes=len(cls_dict), ignore_index=0, average='macro').to(device) 
    val_jaccard = JaccardIndex(task='multiclass', num_classes=len(cls_dict), ignore_index=0, average='macro').to(device) 

    # Best metric variable for checkpointing
    best_val_metric = 0.0

    # Continue training or start from scratch
    """ TODO: this may be necessary once training is conducted with more data
    if cfg['continue']:
        ...
    else:
        start_epoch = 1
    """
    start_epoch = 1

    log_train_config(log_dir=log_dir,
                     cfg={"Session ID": session_id,
                          "Train Dataset Size": len(train_dataset),
                          "Val Dataset Size": len(valid_dataset),
                          "Batch Size": cfg['batch_size'],
                          "Using Device": str(device),
                          "Max Epochs": cfg['max_epochs'],
                          "Learning Rate": cfg['lr'],
                          "Early Stop Patience": cfg['early_stop_patience'],
                          "Early Stop Delta": cfg['early_stop_min_delta'] 
                          })

    epoch_counter = 1
    # Iterate through epochs
    for epoch in range(start_epoch, cfg['max_epochs'] + 1):
        print("Epoch:", epoch, "/", cfg['max_epochs'])
        pbar = tqdm(
            train_dataloader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        # Metrics for this epoch. 
        accuracies, val_accuracies, losses, val_losses =  [], [], [], []

        model.train()

        for idx, batch in enumerate(pbar):
            # Get inputs
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Zero Parameter Gradients + Forward
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)

            # Get Predictions
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)

            # Exclude background class for accuracy
            mask = (labels != 0)  
            pred_labels = predicted[mask].detach().cpu().numpy()
            true_labels = labels[mask].detach().cpu().numpy()

            # Get train metrics
            accuracy = accuracy_score(pred_labels, true_labels)
            loss = outputs.loss
            accuracies.append(accuracy)
            losses.append(loss.item())
            train_jaccard(predicted, labels)

            # Backward + Update Params
            loss.backward()
            optimizer.step()

        else: # TODO: is there a better way?
            model.eval()

            with torch.no_grad():

                for idx, batch in enumerate(valid_dataloader):
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
                    
                    # Exclude background class in the accuracy calculation
                    mask = (labels != 0)  
                    pred_labels = predicted[mask].cpu().detach().numpy()
                    true_labels = labels[mask].cpu().detach().numpy()

                    # Get val metrics for one batch
                    accuracy = accuracy_score(pred_labels, true_labels)
                    val_loss = outputs.loss
                    val_accuracies.append(accuracy)
                    val_losses.append(val_loss.item())
                    val_jaccard(predicted, labels)

        # Mean performance for this epoch   
        train_accuracy = sum(accuracies) / len(accuracies)
        train_loss = sum(losses) / len(losses)
        val_accuracy = sum(val_accuracies) / len(val_accuracies)
        val_loss = sum(val_losses) / len(val_losses)
        train_miou = train_jaccard.compute().item()
        val_miou = val_jaccard.compute().item()

        print(
            f"Train Pixelwise Accuracy: {train_accuracy:.4f} \
            Train Loss: {train_loss:.4f} \
            Train MIoU: {train_miou:.4f} \
            Val Pixelwise Accuracy: {val_accuracy:.4f} \
            Val Loss: {val_loss:.4f} \
            Val MIoU: {val_miou:.4f}"
        )
    
        # TODO: logger class
        log_metrics(epoch, train_accuracy, train_loss, train_miou, val_accuracy, val_loss, val_miou,
                            logs_path=log_dir + "/logs.csv")
        
        # Update checkpoints
        if (epoch_counter == cfg["update_checkpoint_frequency"]):
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), loss, 
                            checkpoint_path=log_dir + f"/checkpoints/last__epoch-{epoch}__acc-{val_accuracy:.3f}__miou-{val_miou:.3f}.pt")
            delete_old_checkpoint(type="last", checkpoint_dir=log_dir + "/checkpoints/")
            epoch_counter = 0

        if val_accuracy > best_val_metric:
            best_val_metric = val_accuracy
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), loss,
                            checkpoint_path=log_dir + f"/checkpoints/best__epoch-{epoch}__acc-{val_accuracy:.3f}__miou-{val_miou:.3f}.pt")
            delete_old_checkpoint(type="best", checkpoint_dir=log_dir + "/checkpoints/")

        # Reset metrics for next epoch    
        train_jaccard.reset()
        val_jaccard.reset()

        epoch_counter += 1

        if early_stopper.early_stop(val_accuracy):
            break

    # Finished training
    plot_curves(log_dir, log_file="/logs.csv")


if __name__ == "__main__":
    t_start = datetime.datetime.now()
    main()
    t_train = datetime.datetime.now() - t_start
    print(f"Training took {(t_train.total_seconds() / 3600):.2f} hours.")