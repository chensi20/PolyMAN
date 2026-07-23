import torch
import os

class Config:
    IMG_SIZE = 352
    BATCH_SIZE = 16   
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 2
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    NUM_CLASSES = 1

    BASE_DIR = "/kaggle/input/pranet-dataset"
    
    TRAIN_IMG_DIR = f"{BASE_DIR}/TrainDataset/TrainDataset/image"
    TRAIN_MASK_DIR = f"{BASE_DIR}/TrainDataset/TrainDataset/masks"
    VAL_IMG_DIR = f"{BASE_DIR}/TestDataset/TestDataset/CVC-ClinicDB/images"
    VAL_MASK_DIR = f"{BASE_DIR}/TestDataset/TestDataset/CVC-ClinicDB/masks"
    

    TEST_IMG_DIR = f"{BASE_DIR}/TestDataset/TestDataset/ETIS-LaribPolypDB/images"
    TEST_MASK_DIR = f"{BASE_DIR}/TestDataset/TestDataset/ETIS-LaribPolypDB/masks"
    

    SAVE_DIR = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
