"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Reduction can only take mean (or) sum (or) none as values")

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        
        # Unpack Pred Boxes
        pred_xc, pred_yc, pred_w, pred_h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

        pred_w = torch.clamp(pred_w, min=0.0)
        pred_h = torch.clamp(pred_h, min=0.0)

        # Reconvert pred to min-max format
        pred_xmin = pred_xc - (pred_w / 2)
        pred_xmax = pred_xc + (pred_w / 2)
        pred_ymin = pred_yc - (pred_h / 2)
        pred_ymax = pred_yc + (pred_h / 2)

        # Unpack Target Boxes
        target_xc, target_yc, target_w, target_h = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

        # Reconvert target to min-max format
        target_xmin = target_xc - (target_w / 2)
        target_xmax = target_xc + (target_w / 2)
        target_ymin = target_yc - (target_h / 2)
        target_ymax = target_yc + (target_h / 2)

        # Find the intersection coordinates
        inter_xmin = torch.max(pred_xmin, target_xmin)
        inter_ymin = torch.max(pred_ymin, target_ymin)
        inter_xmax = torch.min(pred_xmax, target_xmax)
        inter_ymax = torch.min(pred_ymax, target_ymax)

        # Area calculation
        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h= torch.clamp(inter_ymax - inter_ymin, min=0)
        
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        inter_area = inter_w * inter_h
        union_area = pred_area + target_area - inter_area

        union_area = torch.clamp(union_area, min=self.eps)

        # Loss calculation
        iou_score = inter_area / (union_area + self.eps)
        iou_loss = 1 - iou_score

        if self.reduction == "mean":
            iou_loss_value = torch.mean(iou_loss)
        elif self.reduction == "sum":
            iou_loss_value = torch.sum(iou_loss)
        else:
            iou_loss_value = iou_loss

        return iou_loss_value