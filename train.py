import torch
import datetime
import uuid

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from utils.common import common_paths
from utils.misc import initialize_session, save_checkpoint, log_metrics, log_train_config, delete_old_checkpoint
from utils.visualization import plot_curves
from utils.early_stopper import EarlyStopper
from configs.default_cfg import default_cfg
from dataset.dataset_utils import cls_dict, get_transforms
from dataset.disaster_dataset import DisasterSegDataset
from evaluation.metrics import IoUTable


def main():
    cfg = default_cfg
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg['IS_OFFICE'] else "cpu")

    # Create directories for this session
    session_id = '{date:%Y-%m-%d}__'.format(date=datetime.datetime.now()) + uuid.uuid4().hex
    log_dir = common_paths['train_runs'] + session_id
    initialize_session(local_dir=common_paths["train_runs"], session_id=session_id)

    # Augmentations
    train_transforms = get_transforms(mode='train')
    val_transforms = None

    # Datasets
    feature_extractor = SegformerImageProcessor(reduce_zero_labels=True)

    train_dataset = DisasterSegDataset(
        root_dir=common_paths["dataset_root"],
        feature_extractor=feature_extractor,
        train=True,
        transforms=train_transforms,
    )
    valid_dataset = DisasterSegDataset(
        root_dir=common_paths["dataset_root"],
        feature_extractor=feature_extractor,
        train=False,
        transforms=val_transforms,
    )

    # Use subsets for testing
    if cfg['IS_OFFICE']:
        train_dataset = Subset(train_dataset, torch.arange(0,6))
        valid_dataset = Subset(train_dataset, torch.arange(0,2))

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'])

    # Model
    id2label = dict(cls_dict)
    label2id = {v: k for k, v in cls_dict.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg['model_type'],
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )  
    model.to(device) 

    #optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    early_stopper = EarlyStopper(patience=20, min_delta=0.01)


    # Continue training or start from scratch
    """ TODO: this may be necessary, once training is conducted with more data
    if cfg['continue']:
        ...
    else:
        start_epoch = 1
    """
    start_epoch = 1

    train_iou_table = IoUTable(cfg, cls_dict)
    val_iou_table = IoUTable(cfg, cls_dict)

    # Best metric variable for checkpointing
    best_val_metric = 0.0

    log_train_config(log_dir=log_dir,
                     cfg={"Session ID": session_id,
                          "Train Dataset Size": len(train_dataset),
                          "Val Dataset Size": len(valid_dataset),
                          "Batch Size": cfg['batch_size'],
                          "Using Device": str(device),
                          "Epochs": cfg['num_epochs'],
                          "Learning Rate": cfg['lr']
                          })

    epoch_counter = 1
    # Iterate through epochs
    for epoch in range(start_epoch, cfg['num_epochs'] + 1):
        print("Epoch:", epoch, "/", cfg['num_epochs'])
        pbar = tqdm(
            train_dataloader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        # Metrics for this epoch. 
        # TODO: IoUTable should be changed to follow this pattern
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
            mask = (labels != 255)  # exclude background class in the accuracy calculation
            pred_labels = predicted[mask].detach().cpu().numpy()
            true_labels = labels[mask].detach().cpu().numpy()

            # Get train metrics
            accuracy = accuracy_score(pred_labels, true_labels)
            loss = outputs.loss
            accuracies.append(accuracy)
            losses.append(loss.item())
            train_iou_table.get_iou_per_class_epoch(predicted, labels)

            pbar.set_postfix({'Pixelwise Acc': sum(accuracies) / len(accuracies), 'Loss': sum(losses) / len(losses)})

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
                    mask = (labels != 255)  # exclude background class in the accuracy calculation
                    pred_labels = predicted[mask].cpu().detach().numpy()
                    true_labels = labels[mask].cpu().detach().numpy()

                    # Get val metrics for one batch
                    accuracy = accuracy_score(pred_labels, true_labels)
                    val_loss = outputs.loss
                    val_accuracies.append(accuracy)
                    val_losses.append(val_loss.item())
                    val_iou_table.get_iou_per_class_epoch(predicted, labels)

        # Mean performance for this epoch   
        train_accuracy = sum(accuracies) / len(accuracies)
        train_loss = sum(losses) / len(losses)
        val_accuracy = sum(val_accuracies) / len(val_accuracies)
        val_loss = sum(val_losses) / len(val_losses)

        # TODO: remove
        #val_losses.append(val_loss)

        # Get MIoU
        train_iou_table.update_data()
        train_miou = train_iou_table.mean_iou
        
        val_iou_table.update_data()
        val_miou = val_iou_table.mean_iou

        print(
            f"Train Pixelwise Accuracy: {train_accuracy:.4f} \
            Train Loss: {train_loss:.4f} \
            Train MIoU: {train_miou:.4f} \
            Val Pixelwise Accuracy: {val_accuracy:.4f} \
            Val Loss: {val_loss:.4f} \
            Val MIoU: {val_miou:.4f}"
        )
    
        log_metrics(epoch, train_accuracy, train_loss, train_miou, val_accuracy, val_loss, val_miou,
                            logs_path=log_dir + "/logs.csv") # better use a logger
        
        # Update checkpoints
        if (epoch_counter == cfg["update_frequency"]):
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), loss, 
                            checkpoint_path=log_dir + f"/checkpoints/last__epoch-{epoch}__acc-{val_accuracy:.3f}__miou-{val_miou:.3f}.pt")
            delete_old_checkpoint(type="last", checkpoint_dir=log_dir + "/checkpoints/")
            epoch_counter = 0

        if val_accuracy > best_val_metric:
            best_val_metric = val_accuracy
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), loss,
                            checkpoint_path=log_dir + f"/checkpoints/best__epoch-{epoch}__acc-{val_accuracy:.3f}__miou-{val_miou:.3f}.pt")
            delete_old_checkpoint(type="best", checkpoint_dir=log_dir + "/checkpoints/")
            
        epoch_counter += 1

        if early_stopper.early_stop(val_accuracy):
            print("\nTerminating due to Early Stop.")
            break

    # Finished training
    plot_curves(log_dir, log_file="/logs.csv")


if __name__ == "__main__":
    t_start = datetime.datetime.now()
    main()
    t_train = datetime.datetime.now() - t_start
    print(f"Training took {(t_train.total_seconds() / 3600):.2f} hours.")