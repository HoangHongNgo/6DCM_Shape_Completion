import torch
import numpy as np
from extensions.chamfer_distance.chamfer_distance import ChamferDistance


def compute_iou(segMask_pre, seg):
    """
    Compute the Intersection over Union (IoU) for each class.

    Args:
        segMask_pre (torch.Tensor): Predicted segmentation mask.
        seg (torch.Tensor): Ground truth segmentation mask.

    Returns:
        dict: IoU for each class.
    """
    with torch.no_grad():
        present_classes = torch.unique(seg)
        ious = {}

        for cls in present_classes:
            if cls == -1:  # Ignore background or undefined class
                continue

            # Compute intersection and union for each class
            intersection = torch.sum(
                (segMask_pre == cls) & (seg == cls)).item()
            union = torch.sum((segMask_pre == cls) | (seg == cls)).item()

            if union == 0:
                # Handle the case where the class is not present in either prediction or ground truth
                iou = float('nan')
            else:
                iou = intersection / union
            ious[int(cls)] = iou

    return ious


def mean_iou(segMask_pre, seg):
    """
    Compute the mean Intersection over Union (mIoU) across all classes.

    Args:
        segMask_pre (torch.Tensor): Predicted segmentation mask.
        seg (torch.Tensor): Ground truth segmentation mask.

    Returns:
        tuple: Mean IoU and IoU for each class.
    """
    with torch.no_grad():
        ious = compute_iou(segMask_pre, seg)
        valid_ious = [iou for iou in ious.values(
        ) if not torch.isnan(torch.tensor(iou))]
        mIoU = torch.mean(torch.tensor(valid_ious)
                          ) if valid_ious else float('nan')

    return mIoU.item(), ious


def CD_Loss_L1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    with torch.no_grad():
        dist1, dist2 = ChamferDistance(pcs1, pcs2)
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        dist1 = torch.mean(dist1)
        dist2 = torch.mean(dist2)
        res = (dist1 + dist2) / 2.0

    return res
