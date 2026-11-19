import torch
import os

class Config:
  
    IMG_SIZE = 352
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4          
    LR_SCHEDULER = "cosine"     
    T_MAX = 100
    ETA_MIN = 1e-6
    NUM_WORKERS = 2
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 1

   
    SEED = 666                   
    SPLIT_SEED = 42             
    VAL_RATIO = 0.10           
    
    
    MULTI_SEEDS = [42, 3407, 2025, 666, 1234]

    
    ALPHA_BOUNDARY_LOSS = 0.3    
    DEEP_SUPERVISION_WEIGHTS = [1.0, 0.5, 0.3, 0.2]  
    USE_PRETRAINED = True        
    USE_HYBRID_BOTTLENECK = True 
    USE_MAMBA_STREAM = True      
    USE_CNN_STREAM = True        
    USE_GLFM = True              
    USE_SGAM = True              
    
    
    SCANNING_STRATEGY = "1D"     

    
    PRED_THRESHOLD = 0.5
    METRIC_SMOOTH = 1e-5
    
    
    EVAL_BOUNDARY_IOU = True     
    EVAL_HD95 = True             
    EVAL_LESION_FP = True        

    
    PROFILE_BATCH_SIZE = 1
    PROFILE_WARMUP = 100         
    PROFILE_REPEATS = 300        
    PROFILE_FP16 = False         

    
    BASE_DIR = "/kaggle/input/pranet-dataset"

    
    TRAIN_IMG_DIR = f"{BASE_DIR}/TrainDataset/TrainDataset/image"
    TRAIN_MASK_DIR = f"{BASE_DIR}/TrainDataset/TrainDataset/masks"

   
    TEST_DATASETS = {
        "Kvasir": {
            "img": f"{BASE_DIR}/TestDataset/TestDataset/Kvasir/images",
            "mask": f"{BASE_DIR}/TestDataset/TestDataset/Kvasir/masks",
        },
        "CVC-ClinicDB": {
            "img": f"{BASE_DIR}/TestDataset/TestDataset/CVC-ClinicDB/images",
            "mask": f"{BASE_DIR}/TestDataset/TestDataset/CVC-ClinicDB/masks",
        },
        "CVC-ColonDB": {
            "img": f"{BASE_DIR}/TestDataset/TestDataset/CVC-ColonDB/images",
            "mask": f"{BASE_DIR}/TestDataset/TestDataset/CVC-ColonDB/masks",
        },
        "ETIS-LaribPolypDB": {
            "img": f"{BASE_DIR}/TestDataset/TestDataset/ETIS-LaribPolypDB/images",
            "mask": f"{BASE_DIR}/TestDataset/TestDataset/ETIS-LaribPolypDB/masks",
        },
    }

   
    SAVE_DIR = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
