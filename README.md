# Brain Tumour Detection

A deep learning project that uses a Convolutional Neural Network (CNN) built with PyTorch to classify brain MRI images into four categories: **glioma tumour**, **meningioma tumour**, **pituitary tumour**, and **no tumour**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samuel-Mbah/Brain-Tumour-Detection/blob/main/Brain_Tumour_Detection.ipynb)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)

---

## Overview

Brain tumour detection is a critical task in medical imaging. Early and accurate classification can significantly improve patient outcomes. This project trains a CNN from scratch on labelled MRI scan images to distinguish between four classes of brain conditions.

---

## Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/) and contains MRI images divided into training and testing sets.

| Split    | Images |
|----------|--------|
| Training | 2,870  |
| Testing  | 394    |

**Classes:**

| Label               | Description                          |
|---------------------|--------------------------------------|
| `glioma_tumor`      | Tumour originating in glial cells    |
| `meningioma_tumor`  | Tumour arising from the meninges     |
| `pituitary_tumor`   | Tumour of the pituitary gland        |
| `no_tumor`          | Healthy brain scan (no tumour)       |

The dataset folder structure expected by the project is:

```
/content/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

---

## Model Architecture

The custom CNN (`BraTSModel`) consists of:

- **Conv Block 1:** Conv2d(3 → 32) → ReLU → MaxPool2d
- **Conv Block 2:** Conv2d(32 → 64) → ReLU → MaxPool2d
- **Conv Block 3:** Conv2d(64 → 128) → ReLU → MaxPool2d
- **Fully Connected 1:** Linear(128 × 28 × 28 → 512) → ReLU
- **Fully Connected 2:** Linear(512 → 4) *(output layer)*

Input images are resized to **224 × 224** pixels with 3 colour channels (RGB).

---

## Requirements

Install the required Python packages before running the notebook:

```bash
pip install torch torchvision matplotlib numpy
```

> **Note:** A GPU is strongly recommended for training. The notebook automatically detects and uses a CUDA-enabled GPU if available.

---

## Usage

### Running in Google Colab (recommended)

1. Click the **Open In Colab** badge at the top of this README.
2. Mount your Google Drive and upload `archive.zip` (the dataset) to `MyDrive/`.
3. Run all cells in order.

### Running locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Samuel-Mbah/Brain-Tumour-Detection.git
   cd Brain-Tumour-Detection
   ```
2. Download the dataset and extract it so the folder structure matches the layout shown in the [Dataset](#dataset) section.
3. Open `Brain_Tumour_Detection.ipynb` in Jupyter and update the dataset paths accordingly.
4. Run all cells.

---

## Training

The model is trained using the following configuration:

| Hyperparameter | Value       |
|----------------|-------------|
| Optimiser      | SGD         |
| Learning rate  | 0.005       |
| Weight decay   | 0.00001     |
| Loss function  | Cross-Entropy |
| Epochs         | 20          |
| Batch size     | 64          |

Data augmentation applied to the training set:

- Random rotation (±30°)
- Random resized crop (224 × 224)
- Random horizontal flip
- Normalisation (mean = 0.5, std = 0.5 per channel)

The trained model weights are saved to `BraTSModelClassifier.pth` after training completes.

---

## Results

Training and testing accuracy/loss curves are plotted at the end of the notebook to visualise model performance over epochs.


![brats_one](https://github.com/TraflagarLaw/Brain-Tumour-Detection/assets/100414625/8bd9cda0-3870-4c8a-bdd6-7203a5acbef6)
