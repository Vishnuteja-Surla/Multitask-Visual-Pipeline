"""
Custom Dice Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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