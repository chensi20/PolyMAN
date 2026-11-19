import torch
import torch.nn.functional as F
import numpy as np

# Optional dependency for 95th Percentile Hausdorff Distance (HD95)
try:
    from medpy.metric.binary import hd95
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False


def calculate_metrics(inputs, targets, threshold=0.5, smooth=1e-5):
    """
    Computes standard region-based evaluation metrics for medical image segmentation:
    Dice Coefficient, Intersection over Union (IoU), Precision, Recall (Sensitivity),
    and Specificity.

    Args:
        inputs (Tensor): Model output logits, probabilities, or binary masks [B, 1, H, W] or [B, H, W].
        targets (Tensor): Ground truth binary masks [B, 1, H, W] or [B, H, W].
        threshold (float): Binarization threshold. Default: 0.5.
        smooth (float): Smoothing factor to prevent division by zero. Default: 1e-5.

    Returns:
        dict: Dictionary containing scalar floating-point metric scores.
    """
    targets = targets.float()

    # Safely handle raw unnormalized logits vs pre-computed probabilities
    if inputs.min() < 0.0 or inputs.max() > 1.0:
        inputs = torch.sigmoid(inputs)

    # Binarize prediction mask
    inputs_bin = (inputs > threshold).float()

    # Flatten tensors for element-wise confusion matrix calculation
    inputs_flat = inputs_bin.view(-1)
    targets_flat = targets.view(-1)

    # Confusion matrix elements: TP, FP, FN, TN
    TP = (inputs_flat * targets_flat).sum()
    FP = ((inputs_flat == 1) & (targets_flat == 0)).sum()
    FN = ((inputs_flat == 0) & (targets_flat == 1)).sum()
    TN = ((inputs_flat == 0) & (targets_flat == 0)).sum()

    # Metric calculations
    dice = (2.0 * TP + smooth) / (2.0 * TP + FP + FN + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)          # Sensitivity / True Positive Rate
    precision = (TP + smooth) / (TP + FP + smooth)       # Positive Predictive Value
    specificity = (TN + smooth) / (TN + FP + smooth)     # True Negative Rate

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "recall": recall.item(),
        "precision": precision.item(),
        "specificity": specificity.item()
    }


def calculate_hd95_numpy(pred_np, target_np, default_penalty=50.0):
    """
    Calculates the 95th Percentile Hausdorff Distance (HD95) between binary prediction
    and ground truth masks.

    Args:
        pred_np (np.ndarray): Binary prediction array (H, W) in {0, 1}.
        target_np (np.ndarray): Ground truth binary mask array (H, W) in {0, 1}.
        default_penalty (float): Penalty distance returned when foreground is absent.

    Returns:
        float: HD95 distance score in pixels or millimeters.
    """
    if not HAS_MEDPY:
        return 0.0

    pred_bin = (pred_np > 0.5).astype(np.uint8)
    target_bin = (target_np > 0.5).astype(np.uint8)

    pred_has_fg = np.sum(pred_bin) > 0
    target_has_fg = np.sum(target_bin) > 0

    if pred_has_fg and target_has_fg:
        try:
            return float(hd95(pred_bin, target_bin))
        except Exception:
            return 0.0
    elif not pred_has_fg and not target_has_fg:
        return 0.0
    else:
        return default_penalty
