# PolyMAN: A Hybrid Mamba-Based Framework for Real-Time Polyp Segmentation

Official PyTorch implementation of **"PolyMAN: A Hybrid Mamba-Based Framework for Real-Time Polyp Segmentation"**.

## Overview
**PolyMAN** is a hybrid Mamba-based polyp segmentation framework engineered to achieve a superior balance between segmentation accuracy, real-time computational efficiency, and cross-dataset robustness. By effectively incorporating selective Mamba-based state space modeling alongside specialized feature extraction modules, PolyMAN captures long-range contextual dependencies while preserving fine-grained boundary details.

## Dataset Preparation
PolyMAN is evaluated on the following public polyp segmentation benchmarks:

- **Kvasir-SEG**
- **CVC-ClinicDB**
- **CVC-ColonDB**
- **ETIS-LaribPolypDB**

### Option 1: Download from original public sources
Please obtain the datasets from their original public sources whenever available:

- **Kvasir-SEG**: `https://datasets.simula.no/kvasir-seg/`
- **CVC-ClinicDB / CVC-ColonDB**: `https://pages.cvc.uab.es/CVC-Colon/index.php/databases/`

For **ETIS-LaribPolypDB**, please refer to the original source cited in the manuscript or use the prepared split in Option 2 below.

### Option 2: Use a prepared polyp segmentation split
For easier reproduction, users may also follow the widely adopted data organization used in prior polyp segmentation repositories. One commonly used reference is the PraNet repository:

- **PraNet repository**: `https://github.com/DengPingFan/PraNet`

The PraNet repository provides a commonly used training/testing data organization for polyp segmentation research and can be used as a reference for preparing local dataset folders.

After downloading, organize the local dataset directory according to the paths used in `dataset.py`. A typical structure is as follows:

```text
datasets/
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
```

Please adjust the folder names if your local implementation uses different paths.

## Pretrained Weights
The pretrained model weights are archived on Zenodo:

- **DOI:** `https://doi.org/10.5281/zenodo.21506256`

### Setup Steps
1. Create a folder named `./checkpoints/` in the project root directory.
2. Download the pretrained checkpoint from the Zenodo record.
3. Place the downloaded file into `./checkpoints/`.

## Environment Setup

The code was tested with the following environment:

- Python 3.12.13
- PyTorch 2.10.0
- torchvision 0.25.0
- CUDA 12.8
- NVIDIA Tesla T4

### Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```
If you prefer to use conda, you may create an environment first:
```bash
conda create -n polyman python=3.12 -y
conda activate polyman
pip install -r requirements.txt
```
Note: PolyMAN is implemented in pure PyTorch and does NOT require extra CUDA-compiled dependencies such as mamba-ssm or causal-conv1d. This ensures seamless, cross-platform reproducibility out of the box.

## Evaluation
To reproduce the reported test results, run:

```bash
python test.py
```

Please make sure that:
- the dataset path is correctly configured
- the pretrained checkpoint is placed in `./checkpoints/`
- the checkpoint filename matches in `config.py`

## Training
To train LiteMamba-Seg from scratch, run:

```bash
python train.py
```

Please modify the dataset paths according to your local setup and training protocol.

## Outputs
Predicted masks and evaluation results will be saved to the output directory defined in the testing script. You may also save qualitative results for visualization and comparison.

## Notes
- Public datasets are **not redistributed** in this repository or in the Zenodo record.
- Please follow the corresponding dataset licenses and usage conditions.
- The dataset links above are provided for reproducibility. Please refer to the original publications for detailed dataset descriptions and citation information.
- The PraNet repository link is provided as a reference for dataset organization and reproducibility rather than as the official source of all datasets.
- This repository is intended for research and reproducibility purposes only.

## Citation
If you find this repository useful, please cite the corresponding paper:

```bibtex
@article{PolyMAN,
  title={PolyMAN: A Hybrid Mamba-Based Framework for Real-Time Polyp Segmentation},
  author={...},
  journal={...},
  year={2026}
}
```
