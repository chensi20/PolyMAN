import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import csv
import cv2 
import datetime
from PIL import Image
from tqdm import tqdm

from models.PolyMAN import PolyMAN_UNet as PolyMAN 
from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_final_log(dataset_name, mean_dice, mean_iou, model_name="PolyMAN"):
    os.makedirs("./experiment_logs", exist_ok=True)
    csv_path = "./experiment_logs/final_results.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Model", "Dataset", "Mean Dice", "Mean IoU", "Note"])
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, model_name, dataset_name, f"{mean_dice:.4f}", f"{mean_iou:.4f}"])
    print(f"✅ [{dataset_name}] results appended to {csv_path}")

def save_detailed_log(dataset_name, img_name, dice, iou):
    os.makedirs(f"./experiment_logs/details/{dataset_name}", exist_ok=True)
    txt_path = f"./experiment_logs/details/{dataset_name}/scores.txt"
    with open(txt_path, "a") as f:
        f.write(f"{img_name}: Dice={dice:.4f}, IoU={iou:.4f}\n")

def calculate_metrics_simple(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    total = pred.sum() + target.sum()
    union = total - intersection
    dice = (2. * intersection + 1e-5) / (total + 1e-5)
    iou = (intersection + 1e-5) / (union + 1e-5)
    return float(dice.item()), float(iou.item())

def test_dataset(model, img_dir, mask_dir, dataset_name):
    print(f"\n🚀 Testing: {dataset_name} ...")
    
  
    save_pred_dir = f"./results/PolyMAN_Visuals/{dataset_name}/Preds"
    os.makedirs(save_pred_dir, exist_ok=True)
    
    detail_dir = f"./experiment_logs/details/{dataset_name}"
    os.makedirs(detail_dir, exist_ok=True)
    detail_log_path = os.path.join(detail_dir, "scores.txt")
    if os.path.exists(detail_log_path): os.remove(detail_log_path)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    img_paths = [p for p in img_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    if not img_paths:
        print(f"⚠️ Image not found: {img_dir}")
        return None, None

    dice_list = []
    iou_list = []

    model.eval()
    
    for img_path in tqdm(img_paths, desc=f"Evaluating {dataset_name}", leave=False):
        try:
       
            image = Image.open(img_path).convert("RGB")
            original_w, original_h = image.size 
            image = image.resize((Config.IMG_SIZE, Config.IMG_SIZE))
            
            img_np = np.array(image).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_np - mean) / std
            img_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            base_name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(base_name)[0]
            
       
            mask_path = None
            possible_mask_names = [base_name, name_no_ext + ".png", name_no_ext + ".jpg"]
            for m_name in possible_mask_names:
                candidate = os.path.join(mask_dir, m_name)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            
            if not mask_path: continue 


            mask_gt = Image.open(mask_path).convert("L")
            mask_np = np.array(mask_gt).astype(np.float32) / 255.0
            mask_tensor = torch.tensor(mask_np > 0.5).float().to(device)

         
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(img_tensor)
                    if isinstance(pred, (tuple, list)): 
                        pred = pred[0]
                
             
                pred = F.interpolate(pred, size=(original_h, original_w), mode='bilinear', align_corners=False)
                pred = torch.sigmoid(pred).squeeze()

   
            d, i = calculate_metrics_simple(pred, mask_tensor)
            dice_list.append(d)
            iou_list.append(i)
            save_detailed_log(dataset_name, base_name, d, i)
            

            pred_save = (pred > 0.5).float().cpu().numpy()
            pred_save = (pred_save * 255).astype(np.uint8)
            save_name = name_no_ext + ".png" 
            cv2.imwrite(os.path.join(save_pred_dir, save_name), pred_save)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if dice_list:
        mean_dice = np.mean(dice_list)
        mean_iou = np.mean(iou_list)
        save_final_log(dataset_name, mean_dice, mean_iou)
        return mean_dice, mean_iou
    else:
        return None, None


if __name__ == "__main__":
    model = PolyMAN(n_classes=1).to(device) 
    weights_path = "./checkpoints/best_model.pth" 

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Loaded best weights from: {weights_path}")
    else:
        print(f"❌ Error: Model weights not found at {weights_path}")
        print("Please run train.py first to generate the weights!")
        exit()

    test_datasets = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
    results_summary = {}

    for dataset_name in test_datasets:
        img_dir = f"{Config.BASE_DIR}/TestDataset/TestDataset/{dataset_name}/images"
        mask_dir = f"{Config.BASE_DIR}/TestDataset/TestDataset/{dataset_name}/masks"
        
        if os.path.exists(img_dir):
            m_dice, m_iou = test_dataset(model, img_dir, mask_dir, dataset_name)
            if m_dice is not None:
                results_summary[dataset_name] = {"dice": m_dice, "iou": m_iou}
        else:
            print(f"⚠️ Skipping {dataset_name}: Path not found ({img_dir})")

  
    print("\n" + "="*50)
    print("🎉 All Datasets Tested! Table 1 Summary:")
    print(f"| {'Dataset':<18} | {'Dice':<6} | {'IoU':<6} |")
    print("|" + "-"*19 + "|" + "-"*8 + "|" + "-"*8 + "|")
    for name, m in results_summary.items():
        print(f"| {name:<18} | {m['dice']:.4f} | {m['iou']:.4f} |")
    print("="*50)
