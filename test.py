import os
import glob
import csv
import datetime
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

try:
    from medpy.metric.binary import hd95
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False

from config import Config
from models.PolyMAN import PolyMAN_UNet as PolyMAN
from train import seed_everything


def calculate_boundary_iou(pred_bin, target_bin):
    kernel = np.ones((3, 3), np.uint8)
    eroded_target = cv2.erode(target_bin.astype(np.uint8), kernel, iterations=1)
    dilated_target = cv2.dilate(target_bin.astype(np.uint8), kernel, iterations=1)
    gt_boundary = dilated_target - eroded_target

    eroded_pred = cv2.erode(pred_bin.astype(np.uint8), kernel, iterations=1)
    dilated_pred = cv2.dilate(pred_bin.astype(np.uint8), kernel, iterations=1)
    pred_boundary = dilated_pred - eroded_pred

    intersection = np.logical_and(pred_boundary, gt_boundary).sum()
    union = np.logical_or(pred_boundary, gt_boundary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def calculate_metrics_comprehensive(pred_prob, target_np):
    threshold = getattr(Config, 'PRED_THRESHOLD', 0.5)
    pred_bin = (pred_prob > threshold).astype(np.uint8)
    target_bin = (target_np > 0.5).astype(np.uint8)

    intersection = np.sum(pred_bin * target_bin)
    total_area = np.sum(pred_bin) + np.sum(target_bin)
    union_area = total_area - intersection

    # 1. Dice & IoU
    dice = (2.0 * intersection + 1e-5) / (total_area + 1e-5)
    iou = (intersection + 1e-5) / (union_area + 1e-5)
    
    # 2. Recall (Sensitivity) & Precision & MAE
    precision = (intersection + 1e-5) / (np.sum(pred_bin) + 1e-5)
    recall = (intersection + 1e-5) / (np.sum(target_bin) + 1e-5)
    mae = np.mean(np.abs(pred_prob - target_np))
    
    # 3. Boundary IoU
    boundary_iou = calculate_boundary_iou(pred_bin, target_bin)

    # 4. HD95
    hd95_val = 0.0
    if HAS_MEDPY:
        pred_has_fg = np.sum(pred_bin) > 0
        target_has_fg = np.sum(target_bin) > 0
        if pred_has_fg and target_has_fg:
            try:
                hd95_val = float(hd95(pred_bin, target_bin))
            except Exception:
                hd95_val = 0.0
        elif not pred_has_fg and not target_has_fg:
            hd95_val = 0.0
        else:
            hd95_val = 50.0

    return float(dice), float(iou), float(hd95_val), float(mae), float(recall), float(precision), float(boundary_iou)


def evaluate_single_dataset(model, img_dir, mask_dir, dataset_name):
    print(f"\n🔍 Evaluating Benchmark Dataset: [{dataset_name}] ...")

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    img_paths = [p for p in img_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    if not img_paths:
        return None

    dice_l, iou_l, hd95_l, mae_l, rec_l, prec_l, biou_l = [], [], [], [], [], [], []
    model.eval()
    device = Config.DEVICE

    for img_path in tqdm(img_paths, desc=f"Testing [{dataset_name}]", leave=False):
        try:
            image_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image_pil.size

            img_resized = image_pil.resize((Config.IMG_SIZE, Config.IMG_SIZE))
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_np - mean) / std
            img_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

            base_name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(base_name)[0]
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp', '.PNG', '.JPG']:
                candidate = os.path.join(mask_dir, name_no_ext + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if not mask_path and os.path.exists(os.path.join(mask_dir, base_name)):
                mask_path = os.path.join(mask_dir, base_name)

            if not mask_path:
                continue

            mask_gt_pil = Image.open(mask_path).convert("L")
            mask_gt_np = np.array(mask_gt_pil).astype(np.float32) / 255.0

            with torch.no_grad():
                preds = model(img_tensor)
                if isinstance(preds, (tuple, list)):
                    preds = preds[0]

                pred_interp = F.interpolate(preds, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(pred_interp).squeeze().cpu().numpy()

            d, i, h, m, r, p, bi = calculate_metrics_comprehensive(pred_prob, mask_gt_np)
            dice_l.append(d)
            iou_l.append(i)
            hd95_l.append(h)
            mae_l.append(m)
            rec_l.append(r)
            prec_l.append(p)
            biou_l.append(bi)

        except Exception as e:
            continue

    if dice_l:
        res = {
            "mDice": float(np.mean(dice_l)),
            "mIoU": float(np.mean(iou_l)),
            "HD95": float(np.mean(hd95_l)),
            "MAE": float(np.mean(mae_l)),
            "Recall": float(np.mean(rec_l)),
            "Precision": float(np.mean(prec_l)),
            "Boundary_IoU": float(np.mean(biou_l))
        }

        print(f"🏆 Summary [{dataset_name}]: mDice: {res['mDice']:.4f} | mIoU: {res['mIoU']:.4f} | HD95: {res['HD95']:.2f} | MAE: {res['MAE']:.4f} | Rec: {res['Recall']:.4f} | Prec: {res['Precision']:.4f} | BIoU: {res['Boundary_IoU']:.4f}")
        return res
    return None


def save_summary_to_csv(all_results, seed):
    log_dir = "./experiment_logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "multi_seed_results.csv")
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Seed", "Dataset", "mDice", "mIoU", "HD95", "MAE", "Recall", "Precision", "Boundary_IoU"])
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for ds_name, metrics in all_results.items():
            writer.writerow([
                timestamp, seed, ds_name,
                f"{metrics['mDice']:.4f}", f"{metrics['mIoU']:.4f}",
                f"{metrics['HD95']:.2f}", f"{metrics['MAE']:.4f}",
                f"{metrics['Recall']:.4f}", f"{metrics['Precision']:.4f}",
                f"{metrics['Boundary_IoU']:.4f}"
            ])
    print(f"📊 Results for Seed {seed} automatically saved to: {csv_path}")


if __name__ == "__main__":
    seed_everything(Config.SEED)
    device = Config.DEVICE

    model = PolyMAN(n_classes=Config.NUM_CLASSES).to(device)
    weights_path = os.path.join(Config.SAVE_DIR, "best_model_seed_2025.pth")

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint)
        print(f"✅ Model weights loaded successfully from: {weights_path}")
    else:
        print(f"❌ Weights file not found: {weights_path}")
        exit(1)

    all_results = {}
    for ds_name, ds_paths in Config.TEST_DATASETS.items():
        if os.path.exists(ds_paths["img"]):
            res = evaluate_single_dataset(model, ds_paths["img"], ds_paths["mask"], ds_name)
            if res:
                all_results[ds_name] = res

    print("\n" + "=" * 120)
    print("📊 OVERALL BENCHMARK EVALUATION SUMMARY TABLE")
    print("=" * 120)
    for ds_name, metrics in all_results.items():
        print(f"Dataset: {ds_name:18s} | mDice: {metrics['mDice']:.4f} | mIoU: {metrics['mIoU']:.4f} | HD95: {metrics['HD95']:.2f} | MAE: {metrics['MAE']:.4f} | Rec: {metrics['Recall']:.4f} | Prec: {metrics['Precision']:.4f} | BIoU: {metrics['Boundary_IoU']:.4f}")
    print("=" * 120)

    save_summary_to_csv(all_results, Config.SEED)
