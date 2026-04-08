"""
Training Entry-Point
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

# Import your custom modules
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

class DiceLoss(nn.Module):
    """Dice loss for Segmentation.
    """
    def __init__(self, eps: float = 1e-6):
        """
        Initialize the DiceLoss module.
        Args:
            eps: Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred_logits: torch.Tensor, target_masks: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
        """
        Compute the Dice Loss between predicted logits and target masks.
        Args:
            pred_logits: [B, C, H, W] predicted logits by the model.
            target_masks: [B, H, W] containing and integer for class indices at each location.
        """
        # 1. Convert Logits and Probabilities
        pred_probs = torch.softmax(pred_logits, dim=1)

        # 2. One Hot encoding Target Masks
        one_hot_target_masks = F.one_hot(target_masks.long(), num_classes=num_classes)
        one_hot_target_masks = one_hot_target_masks.permute(0, 3, 1, 2).to(torch.float)

        # 3. Finding out the Dice Loss
        intersection_per_pixel = pred_probs * one_hot_target_masks
        intersection = intersection_per_pixel.sum(dim=(2,3))
        prob_sum = pred_probs.sum(dim=(2,3))
        target_sum = one_hot_target_masks.sum(dim=(2,3))
        denominator = prob_sum + target_sum

        mean_dice = 2 * (intersection + self.eps) / (denominator + self.eps)
        dice_loss = 1 - mean_dice

        return dice_loss.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Isolated Perception Tasks")

    # --- The Master Switch ---
    parser.add_argument("-t", "--task", type=str, required=True, choices=["classification", "localization", "segmentation"], help="Which task to train")
    
    # --- Base Config ---
    parser.add_argument("-d", "--data_dir", type=str, default="./data/oxford-iiit-pet", help="Path to dataset")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    # --- W&B Experiment Tracking ---
    parser.add_argument("-rn", "--run_name", type=str, default="baseline_run", help="Name of the wandb run")
    parser.add_argument("-d_p", "--dropout_p", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("-u_b", "--use_batchnorm", type=lambda x: (str(x).lower() == 'true'), default=True, help="Toggle BatchNorm")
    parser.add_argument("-f_s", "--freeze_strategy", type=str, default="none", choices=["none", "partial", "strict"], help="Backbone freezing strategy")

    return parser.parse_args()

def main():
    """
    Main training function.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initiate Wandb Project
    wandb.init(
        project="DA6401_Assignment_02",
        name=args.run_name,
        config=vars(args)
    )

    # 2. Data Transforms
    # Standard ImageNet normalization values
    vgg_mean = (0.485, 0.456, 0.406)
    vgg_std = (0.229, 0.224, 0.225)

    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=vgg_mean, std=vgg_std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc', 
        label_fields=['class_labels'],
        clip=True,
        min_visibility=0.1
    ))

    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=vgg_mean, std=vgg_std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # 3. Datasets & Dataloaders
    full_train_dataset = OxfordIIITPetDataset(data_dir=args.data_dir, transforms=train_transforms)
    full_val_dataset = OxfordIIITPetDataset(data_dir=args.data_dir, transforms=val_transforms)

    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))

    np.random.seed(42) 
    np.random.shuffle(indices)

    val_split = int(np.floor(0.2 * dataset_size))
    val_indices, train_indices = indices[:val_split], indices[val_split:]

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)

    optimal_workers = os.cpu_count()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, # Shuffle batches every epoch
        num_workers=optimal_workers, 
        pin_memory=True # Speeds up data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # No need to shuffle validation
        num_workers=optimal_workers, 
        pin_memory=True
    )

    # 4. Model Initialization
    if args.task == "classification":
        model = VGG11Classifier(dropout_p=args.dropout_p, use_batchnorm=args.use_batchnorm)
        ce_loss = nn.CrossEntropyLoss()

    elif args.task == "localization":
        model = VGG11Localizer(dropout_p=args.dropout_p, use_batchnorm=args.use_batchnorm)

        if os.path.exists("checkpoints/classifier.pth"):
            class_checkpoint = torch.load("checkpoints/classifier.pth", map_location="cpu")
            class_state = class_checkpoint.get("state_dict", class_checkpoint)
            class_encoder = {k.replace("encoder.", ""): v for k, v in class_state.items() if k.startswith("encoder.")}
            model.encoder.load_state_dict(class_encoder, strict=False)
        else:
            print("[WARNING] classifier.pth missing. Initializing localizer encoder with random weights.")

        mse_loss = nn.MSELoss()
        iou_loss = IoULoss(reduction="mean")

    elif args.task == "segmentation":
        model = VGG11UNet(dropout_p=args.dropout_p, use_batchnorm=args.use_batchnorm)

        if os.path.exists("checkpoints/classifier.pth"):
            class_checkpoint = torch.load("checkpoints/classifier.pth", map_location="cpu")
            class_state = class_checkpoint.get("state_dict", class_checkpoint)
            class_encoder = {k.replace("encoder.", ""): v for k, v in class_state.items() if k.startswith("encoder.")}
            model.encoder.load_state_dict(class_encoder, strict=False)
        else:
            print("[WARNING] classifier.pth missing. Initializing UNet encoder with random weights.")

        ce_loss = nn.CrossEntropyLoss()
        dice_loss = DiceLoss()

    model = model.to(device)

    # Freeze strategy implementation
    if args.freeze_strategy == "strict":
        print("--> Strategy: Strict Feature Extractor. Freezing the entire encoder...")
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif args.freeze_strategy == "partial":
        print("--> Strategy: Partial Fine-Tuning. Freezing early blocks...")
        # Freeze blocks 1, 2, and 3, and leave 4 and 5 to train.
        for name, param in model.encoder.named_parameters():
            if any(block in name for block in ["block1", "block2", "block3"]):
                param.requires_grad = False
    elif args.freeze_strategy == "none":
        print("--> Strategy: Full Fine-Tuning. Training all layers end-to-end...")

    # 5. Optimizers
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_metric = 0.0

    scaler = torch.amp.GradScaler('cuda')
    # 6. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for images, class_labels, bboxes, masks, has_bbox in train_pbar:
            images = images.to(device)
            class_labels = class_labels.to(device)
            bboxes = bboxes.to(device)
            masks = masks.to(device)
            has_bbox = has_bbox.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                if args.task == "classification":
                    loss = ce_loss(outputs, class_labels)
                elif args.task == "localization":
                    if not has_bbox.any():
                        continue

                    valid_outputs = outputs[has_bbox]
                    valid_bboxes = bboxes[has_bbox]

                    batch_iou_loss = iou_loss(valid_outputs, valid_bboxes)
                    norm_outputs = valid_outputs / 224.0
                    norm_bboxes = valid_bboxes / 224.0
                    loss = mse_loss(norm_outputs, norm_bboxes) + batch_iou_loss
                elif args.task == "segmentation":
                    loss = ce_loss(outputs, masks) + dice_loss(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # 7. Validation Loop
        model.eval()
        val_loss = 0.0

        # Accumulators for metrics
        all_class_preds = []
        all_class_targets = []
        val_iou_total = 0.0
        val_dice_total = 0.0
        val_pixel_acc_total = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            for images, class_labels, bboxes, masks, has_bbox in val_pbar:
                images = images.to(device)
                class_labels = class_labels.to(device)
                bboxes = bboxes.to(device)
                masks = masks.to(device)
                has_bbox = has_bbox.to(device)

                outputs = model(images)

                if args.task == "classification":
                    loss = ce_loss(outputs, class_labels)
                    preds = torch.argmax(outputs, dim=1)
                    all_class_preds.extend(preds.cpu().numpy())
                    all_class_targets.extend(class_labels.cpu().numpy())

                elif args.task == "localization":

                    if not has_bbox.any():
                        continue
                    valid_outputs = outputs[has_bbox]
                    valid_bboxes = bboxes[has_bbox]

                    batch_iou_loss = iou_loss(valid_outputs, valid_bboxes)
                    # Normalize coordinates to [0, 1] purely for the MSE calculation
                    norm_outputs = valid_outputs / 224.0
                    norm_bboxes = valid_bboxes / 224.0
                    loss = mse_loss(norm_outputs, norm_bboxes) + batch_iou_loss
                    # Reverse engineer the IoU score from the loss (IoU = 1 - IoULoss)
                    val_iou_total += (1.0 - batch_iou_loss.item())

                elif args.task == "segmentation":
                    batch_dice_loss = dice_loss(outputs, masks)
                    loss = ce_loss(outputs, masks) + batch_dice_loss
                    # Reverse engineer the Dice score from the loss (Dice = 1 - DiceLoss)
                    val_dice_total += (1.0 - batch_dice_loss.item())

                    # Calculate Pixel Accuracy
                    preds = torch.argmax(outputs, dim=1)
                    pixel_acc = (preds == masks).float().mean().item()
                    # You'll need to initialize val_pixel_acc_total = 0.0 before the loop
                    val_pixel_acc_total += pixel_acc

                val_loss += loss.item()

                val_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        
        scheduler.step() # Step the learning rate scheduler after each epoch

        avg_val_loss = val_loss / len(val_loader)

        # 8. Metric Calculation and Wandb Logging
        if args.task == "classification":
            # Calculate Macro F1 Score using sklearn
            current_f1 = f1_score(all_class_targets, all_class_preds, average='macro')
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_f1_macro": current_f1, "epoch": epoch + 1})
            metric_display = f"F1: {current_f1:.4f}"
            
        elif args.task == "localization":
            avg_iou = val_iou_total / len(val_loader)
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_iou": avg_iou, "epoch": epoch + 1})
            metric_display = f"IoU: {avg_iou:.4f}"
            
        elif args.task == "segmentation":
            avg_dice = val_dice_total / len(val_loader)            
            avg_pixel_acc = val_pixel_acc_total / len(val_loader)
            wandb.log({
                "train_loss": avg_train_loss, 
                "val_loss": avg_val_loss, 
                "val_dice": avg_dice,
                "val_pixel_acc": avg_pixel_acc,
                "epoch": epoch + 1
            })
            metric_display = f"Dice: {avg_dice:.4f} | Pixel Acc: {avg_pixel_acc:.4f}"

        print(f"Epoch [{epoch+1}/{args.epochs}] - Task: {args.task} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | {metric_display}")

        # 9. Simple Checkpointing
        # Determine if this is the best model
        is_best = False
        if args.task == "classification" and current_f1 > best_val_metric:
            best_val_metric = current_f1
            is_best = True
        elif args.task == "localization" and avg_iou > best_val_metric: # Changed to avg_iou >
            best_val_metric = avg_iou
            is_best = True
        elif args.task == "segmentation" and avg_dice > best_val_metric: # Changed to avg_dice >
            best_val_metric = avg_dice
            is_best = True
            
        if is_best:
            print(f"--> Saving best {args.task} model...")
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_payload = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_val_metric,
                "classes": full_train_dataset.classes
            }
            
            if args.task == "classification":
                torch.save(checkpoint_payload, "checkpoints/classifier.pth")
            elif args.task == "localization":
                torch.save(checkpoint_payload, "checkpoints/localizer.pth")
            elif args.task == "segmentation":
                torch.save(checkpoint_payload, "checkpoints/unet.pth")

    wandb.finish()

if __name__ == "__main__":
    main()