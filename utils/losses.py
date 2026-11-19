import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureLoss(nn.Module):
    """
    Structure Loss combines Weighted Binary Cross-Entropy (wBCE) and Weighted IoU (wIoU).
    It assigns higher dynamic weights to hard pixels near boundary edges and high-contrast regions.
    """
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        """
        Args:
            pred (Tensor): Raw network output logits [B, 1, H, W] or [B, H, W]
            mask (Tensor): Ground truth binary mask [B, 1, H, W] or [B, H, W]
        """
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        mask = mask.float()

        # Compute dynamic pixel-wise importance weight map based on local edge density
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        # 1. Weighted Binary Cross-Entropy Loss (wBCE)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (wbce * weit).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)

        # 2. Weighted Intersection over Union Loss (wIoU)
        pred_sig = torch.sigmoid(pred)
        inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
        union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1.0) / (union - inter + 1.0)

        return (wbce + wiou).mean()



class BoundaryIoULoss(nn.Module):
    
    def __init__(self, kernel_size=3):
        super(BoundaryIoULoss, self).__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Logits tensor [B, 1, H, W]
            target (Tensor): Ground truth binary mask [B, 1, H, W]
        """
        pred_sig = torch.sigmoid(pred)

        if target.ndim == 3:
            target = target.unsqueeze(1)
        if pred_sig.ndim == 3:
            pred_sig = pred_sig.unsqueeze(1)

        target = target.float()

        # Differentiable morphological boundary extraction: B = Dilated(G) - Eroded(G)
        eroded = 1.0 - self.pool(1.0 - target)
        dilated = self.pool(target)
        gt_boundary = dilated - eroded  # Boundary band mask B

        # Intersection restricted to boundary region B: |(P \cap G) \cap B|
        intersection = (pred_sig * target * gt_boundary).sum(dim=(2, 3))

        # Union restricted to boundary region B: |(P \cup G) \cap B| = |P \cap B| + |G \cap B| - |(P \cap G) \cap B|
        union = (pred_sig * gt_boundary).sum(dim=(2, 3)) + (target * gt_boundary).sum(dim=(2, 3)) - intersection

        boundary_iou = (intersection + 1e-6) / (union + 1e-6)

        return (1.0 - boundary_iou).mean()



class PolyMANHybridLoss(nn.Module):
    
    def __init__(self, alpha=0.3, ds_weights=[1.0, 0.3, 0.2, 0.1]):
        super(PolyMANHybridLoss, self).__init__()
        self.alpha = alpha
        self.ds_weights = ds_weights
        self.structure_loss = StructureLoss()
        self.boundary_loss = BoundaryIoULoss()

    def forward(self, preds, target):
        
        if isinstance(preds, (tuple, list)):
            total_loss = 0.0
            for idx, pred in enumerate(preds):
                w = self.ds_weights[idx] if idx < len(self.ds_weights) else 0.1
                l_struct = self.structure_loss(pred, target)
                
                # Apply explicit boundary loss coefficient \alpha = 0.3 to the primary output head
                if idx == 0:
                    l_bound = self.boundary_loss(pred, target)
                    head_loss = l_struct + self.alpha * l_bound
                else:
                    head_loss = l_struct
                    
                total_loss += w * head_loss
            return total_loss
        else:
            l_struct = self.structure_loss(preds, target)
            l_bound = self.boundary_loss(preds, target)
            return l_struct + self.alpha * l_bound
