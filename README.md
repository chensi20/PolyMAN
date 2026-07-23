# PolyMAN: A Hybrid Mamba-Based Framework for Real-Time Polyp Segmentation

Official PyTorch implementation of **"PolyMAN: A Hybrid Mamba-Based Framework for Real-Time Polyp Segmentation"**.

---

## Overview

**PolyMAN** is a hybrid Mamba-based polyp segmentation framework engineered to achieve a superior balance between segmentation accuracy, real-time computational efficiency, and cross-dataset robustness. By effectively incorporating selective Mamba-based state space modeling alongside specialized feature extraction modules, PolyMAN captures long-range contextual dependencies while preserving fine-grained boundary details.

---

## Dataset Preparation

PolyMAN is evaluated on standard public polyp segmentation benchmarks:
* **Kvasir-SEG**
* **CVC-ClinicDB**
* **CVC-ColonDB**
* **ETIS-LaribPolypDB**

### Option 1: Download from Original Public Sources
Please obtain the datasets from their original repositories whenever available:
* **Kvasir-SEG**: [https://datasets.simula.no/kvasir-seg/](https://datasets.simula.no/kvasir-seg/)
* **CVC-ClinicDB / CVC-ColonDB**: [https://pages.cvc.uab.es/CVC-Colon/index.php/databases/](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/)
* **ETIS-LaribPolypDB**: Please refer to the original source cited in the manuscript or use the prepared split below.

### Option 2: Use a Prepared Polyp Segmentation Split
For easier reproduction, you can adopt the widely used data organization format introduced in prior polyp segmentation repositories (such as [PraNet](https://github.com/DengPingFan/PraNet)).

After downloading, organize your dataset directory as follows:

```text
data/
├── TrainDataset/
│   ├── image/
│   └── masks/
└── TestDataset/
    ├── CVC-ClinicDB/
    │   ├── images/
    │   └── masks/
    ├── CVC-ColonDB/
    │   ├── images/
    │   └── masks/
    ├── ETIS-LaribPolypDB/
    │   ├── images/
    │   └── masks/
    └── Kvasir/
        ├── images/
        └── masks/
