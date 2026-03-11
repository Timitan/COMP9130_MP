# Mini Project 8: Image Segmentation — Natural Disaster Damage Assessment

## Problem Description and Motivation

Semantic segmentation of aerial disaster imagery is a high-impact application of computer vision with direct relevance to emergency response operations. For this project, we train a U-Net model to perform binary pixel-level segmentation on aerial and UAV images of flood-affected areas, classifying every pixel as either **flood** or **non-flood**.

Unlike image classification or object detection, semantic segmentation produces a dense prediction map at full image resolution, making it suitable for precise flood extent mapping. Emergency response teams need rapid, accurate flood maps to prioritize rescue operations, route personnel around impassable roads, and allocate resources effectively — tasks that become critical in the hours immediately following a flood event when ground access is limited.

## Dataset Description

**Source**  
Flood Area Segmentation — Kaggle:  
https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation

**Task**  
Binary semantic segmentation of aerial/UAV flood imagery

**Classes**
- Non-Flood (0)
- Flood (1)

**Dataset Size**
- Total: 290 image-mask pairs
- Train: 208 images
- Validation: 24 images
- Test: 58 images

**Class Distribution (Training Set — 85,196,800 pixels)**
- Non-Flood: 50,212,115 pixels (58.94%)
- Flood: 34,984,685 pixels (41.06%)
- Class weights applied: Non-flood 0.848, Flood 1.218

## Project Structure
```
mini-project-8/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── Group2_Mini_Project_VIII_COMP_9130.ipynb
|   └── 256x256_Group2_Mini_Project_VIII_COMP_9130.ipynb    (Tests for 256x256 image resolution)
├── results/
│   ├── cm_pixel_level_result.png
│   ├── fig_augmentation_visuals.png
│   ├── fig_example_visuals.png
│   ├── fig_good_predictions.png
│   ├── fig_poor_predictions.png
│   └── fig_training_curves.png
└── flood_dataset/                  (downloaded via kagglehub — not committed, see Setup)
    ├── Image/
    │   ├── 38.jpg
    │   └── ...
    └── Mask/
        ├── 38.png
        └── ...
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <repo-url>
cd mini-project-8
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:  
The notebook downloads the dataset automatically via `kagglehub`. You will need a Kaggle account and your `kaggle.json` API credentials configured. Run the dataset download cell at the top of the notebook, which will pull the dataset to `/content/flood_dataset/` (or your local cache).

   Alternatively, download manually from:  
   https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation  
   and place the `Image/` and `Mask/` folders inside a `flood_dataset/` directory in the project root. Update `DATASET_PATH` in the configuration cell accordingly.

4. Run the notebook:
   - Open `notebooks/Group2_Mini_Project_VIII_COMP_9130.ipynb` in Jupyter Lab or Google Colab
   - A GPU runtime is strongly recommended (Tesla T4 or equivalent)
   - Sections run in order: Setup → Preprocessing → Augmentation → Model → Training → Evaluation

## Methods Used

**Model**
- U-Net (custom implementation in TensorFlow/Keras)
- Encoder: 4 downsampling blocks (32 → 64 → 128 → 256 filters)
- Bottleneck: 512 filters
- Decoder: 4 upsampling blocks with skip connections
- BatchNormalization after every Conv2D layer
- Output: 1×1 Conv2D with sigmoid activation
- Total parameters: 7,771,873

**Training Configuration**
- Epochs: 25
- Image size: 640×640
- Batch size: 8
- Optimizer: Adam (lr = 1e-3)
- Loss function: Dice Loss
- LR schedule: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience=5, monitor=val_loss
- Class weights: {0: 0.848, 1: 1.218}

**Augmentation**
- Random horizontal flip (image + mask, synchronized)
- Random vertical flip (image + mask, synchronized)
- Random brightness adjustment (image only)
- Random contrast adjustment (image only)

**Evaluation Metrics**
- Per-class IoU (Intersection over Union)
- mIoU (mean IoU)
- Per-class Dice coefficient
- Mean Dice
- Pixel-level confusion matrix
- Sensitivity, Specificity, F1-Score

## Results Summary

**Overall Performance (Test Set — 58 images)**
- mIoU: 0.7195
- Mean Dice: 0.8364
- Sensitivity: 0.8688
- Specificity: 0.8205
- F1-Score: 0.8128

**Per-Class Performance**

| Class | IoU | Dice |
|-------|-----|------|
| Non-Flood | 0.7543 | 0.8600 |
| Flood | 0.6847 | 0.8128 |
| **Mean** | **0.7195** | **0.8364** |

**Pixel-Level Confusion Matrix (Test Set — 23,756,800 pixels)**

| | Pred: Non-Flood | Pred: Flood |
|---|---|---|
| **True: Non-Flood** | 11,686,703 (TN) | 2,557,312 (FP) |
| **True: Flood** | 1,248,487 (FN) | 8,264,298 (TP) |

**Key Findings**

Non-flood segmentation (IoU: 0.754) outperforms flood segmentation (IoU: 0.685) by 6.9 points. Flood water in this dataset frequently appears as muddy brown, visually similar to wet soil and sediment — making it the harder class. Error analysis shows that for good predictions (Dice 0.869–0.947), errors concentrate at class boundaries as thin orange fringes, while class interiors are correctly predicted. The worst failure cases involve fragmented flood patches interspersed with dry land, where the model struggles to identify disconnected flood regions.

The best model checkpoint was saved at epoch 25 with val Dice = 0.817, val IoU = 0.694. The learning rate was reduced from 1e-3 to 5e-4 at epoch 13 via ReduceLROnPlateau, after which validation performance improved steadily.

## Team Member Contributions

* **Henry Chen**
  * Wrote Report with analysis, discussion, and references
  * Made README.md
  * Made prediction visualizations 

* **Timothy Tan**
  * Did data pre-processing, UNet model and training
  * Made result visualizations
  * Ran tests on 256x256 image resolutions
