import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from tqdm import tqdm

from config import Config
from dataset import get_loaders
from utils.losses import StructureLoss
from utils.metrics import calculate_metrics

from models.PolyMAN import PolyMAN_UNet as UNet


def seed_everything(seed=42):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 1. Boundary IoU Loss
class BoundaryIoULoss(nn.Module):
    def __init__(self):
        super(BoundaryIoULoss, self).__init__()

    def forward(self, pred, target):
        """
        Boundary IoU Loss Computation
        pred: Logits
        target: Binary Mask
        """
        pred = torch.sigmoid(pred)
        
        if target.ndim == 3: 
            target = target.unsqueeze(1)
        if pred.ndim == 3: 
            pred = pred.unsqueeze(1)

        target = target.float()
        
        pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        eroded = 1 - pool(1 - target)
        dilated = pool(target)     
        gt_boundary = dilated - eroded 
        
        intersection = (pred * target * gt_boundary).sum()
        union = (pred * gt_boundary).sum() + (target * gt_boundary).sum()
        
        boundary_iou = (intersection + 1e-6) / (union - intersection + 1e-6)
        
        return 1 - boundary_iou


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# Deep Supervision Training Function
def train_one_epoch(model, loader, criterion_main, criterion_bound, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=True)
    
    alpha = getattr(Config, 'ALPHA_BOUNDARY_LOSS', 0.3)  # Bind the alpha weight parameter from Config
    ds_weights = getattr(Config, 'DEEP_SUPERVISION_WEIGHTS', [1.0, 0.5, 0.3, 0.2])
    
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        
        # Reshape masks: [B, H, W] -> [B, 1, H, W]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        elif masks.ndim == 4 and masks.shape[-1] == 1:
            masks = masks.permute(0, 3, 1, 2)

        optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            preds = model(images)
            
            # --- Deep Supervision ---
            if isinstance(preds, (tuple, list)):
                out1, out2, out3, out4 = preds
                
                # Main output contains both Structure Loss and Boundary IoU Loss weighted by explicit alpha
                loss1 = criterion_main(out1, masks) + alpha * criterion_bound(out1, masks)
                loss2 = criterion_main(out2, masks)
                loss3 = criterion_main(out3, masks)
                loss4 = criterion_main(out4, masks)
                
                loss = (ds_weights[0] * loss1 + 
                        ds_weights[1] * loss2 + 
                        ds_weights[2] * loss3 + 
                        ds_weights[3] * loss4)
                
            else:
                loss_main = criterion_main(preds, masks)
                loss_bound = criterion_bound(preds, masks)
                loss = loss_main + alpha * loss_bound

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item(), lr=get_lr(optimizer))
        
    return running_loss / len(loader)


def validate(model, loader, criterion_main, criterion_bound, device):
    """
    Evaluate on the 10% independent validation set (145 images split from the Development Pool).
    """
    model.eval()
    running_loss = 0.0
    metrics_score = {"dice": 0.0, "iou": 0.0, "recall": 0.0, "precision": 0.0}
    alpha = getattr(Config, 'ALPHA_BOUNDARY_LOSS', 0.3)
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="[Val]", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            elif masks.ndim == 4 and masks.shape[-1] == 1:
                masks = masks.permute(0, 3, 1, 2)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds = model(images)
                
                if isinstance(preds, (tuple, list)):
                    preds = preds[0]
                
                loss_main = criterion_main(preds, masks)
                loss_bound = criterion_bound(preds, masks)
                loss = loss_main + alpha * loss_bound
            
            running_loss += loss.item()
            
            preds_prob = torch.sigmoid(preds)
            preds_bin = (preds_prob > getattr(Config, 'PRED_THRESHOLD', 0.5)).float() 
            
            batch_metrics = calculate_metrics(preds_bin, masks.float())
            
            for k in metrics_score:
                if k in batch_metrics:
                    metrics_score[k] += batch_metrics[k]

    epoch_loss = running_loss / len(loader)
    for k in metrics_score:
        metrics_score[k] /= len(loader)
        
    return epoch_loss, metrics_score


def train_pipeline(config=Config, seed=42):
    
    seed_everything(seed)
    
    device = config.DEVICE
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"🚀 Training PolyMAN | Device: {device} | Seed: {seed}")
    print("=" * 70)

    # 2. Load 1,305 training samples + 145 independent validation samples from dataset
    
    train_loader, val_loader = get_loaders(config)
    print(f"Data Loaded: Train {len(train_loader)} batches | Val {len(val_loader)} batches")

    # 3. Initialize model
    model = UNet(n_classes=config.NUM_CLASSES).to(device)
    
    # 4. Optimizer & LR Scheduler
    weight_decay = getattr(config, 'WEIGHT_DECAY', 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=getattr(config, 'T_MAX', config.EPOCHS), 
        eta_min=getattr(config, 'ETA_MIN', 1e-6)
    )
    
    criterion_main = StructureLoss()            
    criterion_bound = BoundaryIoULoss().to(device) 
    scaler = GradScaler() 
    
    best_dice = 0.0
    best_ckpt_path = os.path.join(config.SAVE_DIR, f"best_model_seed_{seed}.pth")
    last_ckpt_path = os.path.join(config.SAVE_DIR, f"last_model_seed_{seed}.pth")
    
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion_main, criterion_bound, 
            optimizer, scaler, device, epoch
        )
        
        val_loss, metrics = validate(model, val_loader, criterion_main, criterion_bound, device)
        scheduler.step()
        current_lr = get_lr(optimizer)
        
        is_best = metrics['dice'] > best_dice
        save_msg = ""
        if is_best:
            best_dice = metrics['dice']
            torch.save({
                'epoch': epoch + 1,
                'seed': seed,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_dice,
            }, best_ckpt_path)
            
            # Save a default best_model.pth to ensure compatibility with existing evaluation scripts
            torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, "best_model.pth"))
            save_msg = "🔥 [Best Val Dice Updated!]"
            
        torch.save(model.state_dict(), last_ckpt_path)
        
        print(f"\nEpoch [{epoch+1}/{config.EPOCHS}] Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Val Dice: {metrics['dice']:.4f} | Val IoU: {metrics['iou']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f}")
        print(f"   LR: {current_lr:.6f} {save_msg}")
        print("-" * 60)

    print(f"\n🎉 Training Finished! Best Validation Dice (Seed {seed}): {best_dice:.4f}\n")


if __name__ == "__main__":
    # Execute training pipeline with the main seed
    train_pipeline(config=Config, seed=Config.SEED)
