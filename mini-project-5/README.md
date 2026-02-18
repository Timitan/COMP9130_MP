# 🛰️ Satellite Image Classification using CNN

A deep learning project that classifies satellite imagery into four land-cover categories using Convolutional Neural Networks built with TensorFlow/Keras. The project progresses from a baseline CNN through multiple debugging iterations to a final model achieving **99.7% validation accuracy**.

---

## 📋 Table of Contents
- [Problem Description & Motivation](#-problem-description--motivation)
- [Dataset](#-dataset)
- [Setup & Running Instructions](#️-setup--running-instructions)
- [Model Architecture](#-model-architecture)
- [Results Summary](#-results-summary)
- [Debugging Journey](#-debugging-journey)
- [Team Member Contributions](#-team-member-contributions)

---

## 🎯 Problem Description & Motivation

Satellite image classification is a fundamental task in remote sensing and geospatial analysis. Accurate, automated classification of land-cover types enables a wide range of real-world applications:

- **Environmental monitoring** — tracking deforestation, desertification, and changes in water bodies over time
- **Urban planning** — identifying green spaces, built-up areas, and water infrastructure at scale
- **Disaster response** — rapidly mapping affected terrain after floods or wildfires without ground-level access
- **Agricultural management** — distinguishing cultivated land from natural vegetation across large regions

Manual classification of satellite imagery is slow, costly, and difficult to scale. This project investigates whether a CNN trained from scratch can reliably distinguish between four land-cover classes at high accuracy. It also documents the full debugging journey — from a model that silently failed to apply augmentation, through a double-rescaling bug that caused training collapse, to a final model with less than 0.3% error.

---

## 📦 Dataset

| Property | Detail |
|----------|--------|
| **Name** | Satellite Image Classification |
| **Source** | [Kaggle — mahmoudreda55](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification) |
| **Total Images** | 5,631 |
| **Classes** | `cloudy`, `desert`, `green_area`, `water` |
| **Train Split** | 80% — 4,505 images |
| **Validation Split** | 20% — 1,126 images |
| **Image Size** | Resized to 256 × 256 pixels (RGB) |

### Class Distribution

| Class | Count | Notes |
|-------|-------|-------|
| cloudy | 1,500 | Native resolution 256×256 |
| desert | 1,131 | ⚠️ Minority class — ~25% fewer samples than others |
| green_area | 1,500 | Native resolution 64×64, upscaled to 256×256 |
| water | 1,500 | Native resolution 64×64, upscaled to 256×256 |

### Data Quality Notes

- `green_area` and `water` images are natively 64×64 and are upscaled to 256×256, introducing interpolation artifacts
- `cloudy` images contain an alpha channel (RGBA) which is dropped during loading
- Some `cloudy` and `desert` images exhibit banding artifacts where pixel values shift abruptly
- `desert` class imbalance (~1,131 vs ~1,500) may introduce mild bias toward majority classes

---

## ⚙️ Setup & Running Instructions

### Prerequisites

- Python 3.9+
- TensorFlow 2.x
- GPU strongly recommended (tested on NVIDIA T4 via Google Colab — ~3× faster than CPU)

### 1. Install dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn kagglehub pandas
```

### 2. Download the dataset

The notebook handles this automatically via `kagglehub`:

```python
import kagglehub
dataset_path = kagglehub.dataset_download("mahmoudreda55/satellite-image-classification")
```

Alternatively, download manually from [Kaggle](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification) and point `data_dir` to the extracted folder containing the four class subfolders.

### 3. Run the notebook

Open and execute `satellite_cnn_v4_warmup_fixed.ipynb` sequentially from top to bottom:

```bash
jupyter notebook satellite_cnn_v4_warmup_fixed.ipynb
```

> **Google Colab users**: Enable GPU before running via `Runtime → Change runtime type → T4 GPU`.

### 4. Expected runtime (T4 GPU)

| Model | Epochs Run | Approximate Time |
|-------|------------|-----------------|
| Baseline CNN | 15 | ~4 minutes |
| Improved CNN | 15 | ~7 minutes |
| GAP Experiment | ~8 (early stopped) | ~3 minutes |

---

## 🏗️ Model Architecture

### Baseline CNN

A straightforward 3-block CNN with a Flatten-based classification head. The Flatten layer produces a 115,200-dimensional vector from the final feature maps, which connects to a Dense(128) layer — resulting in approximately **14.8 million parameters** (56.6 MB). Despite its size, the large parameter count is concentrated in the Flatten→Dense connection and does not meaningfully help generalisation.

**Layer stack:**
```
Input (256×256×3)
→ Rescaling (÷255)
→ Conv2D(32, 3×3, relu) → MaxPooling2D
→ Conv2D(64, 3×3, relu) → MaxPooling2D
→ Conv2D(128, 3×3, relu) → MaxPooling2D
→ Flatten
→ Dense(128, relu)
→ Dense(4, softmax)
```

### Improved CNN (Final Model)

A 4-block CNN replacing Flatten with `GlobalAveragePooling2D`, adding `BatchNormalization` after each Conv block, L2 regularisation on all learnable layers, and data augmentation embedded as the first processing step. Total parameters: **391,364** (~1.49 MB) — **38× fewer than the baseline** while achieving significantly higher accuracy.

**Layer stack:**
```
Input (256×256×3)
→ Rescaling (÷255)
→ DataAugmentation [RandomFlip, RandomRotation(±15%), RandomZoom(±15%), RandomContrast(±10%)]
→ Conv2D(32,  3×3, relu, L2) → BatchNormalization → MaxPooling2D
→ Conv2D(64,  3×3, relu, L2) → BatchNormalization → MaxPooling2D
→ Conv2D(128, 3×3, relu, L2) → BatchNormalization → MaxPooling2D
→ Conv2D(256, 3×3, relu, L2) → BatchNormalization → MaxPooling2D
→ GlobalAveragePooling2D
→ Dropout(0.5)
→ Dense(4, softmax, L2)
```

**Training schedule:**
- Linear LR warmup from `1e-5` → `2e-4` over the first 3 epochs (stabilises BatchNormalization running statistics before large weight updates occur)
- `ReduceLROnPlateau`: factor=0.5, patience=4, min_lr=1e-6
- `EarlyStopping`: patience=10, restores best weights

### GAP Experiment (Bonus)

A lighter 3-block variant without BatchNormalization or L2 regularisation, used to isolate the contribution of GlobalAveragePooling2D vs Flatten. Achieves 92.3% validation accuracy — confirming that GAP alone is beneficial but BN and regularisation provide the majority of the improvement.

---

## 📊 Results Summary

### Model Comparison

| Model | Parameters | Val Accuracy | Train Accuracy | Overfit Gap |
|-------|-----------|-------------|----------------|-------------|
| Baseline CNN | 14,839,492 | 94.7% | 93.9% | −0.8% ✅ |
| **Improved CNN** | **391,364** | **99.7%** | **99.2%** | **−0.6% ✅** |
| GAP Experiment | ~149,000 | 92.3% | 92.7% | +0.7% ✅ |

All three models generalise well (no large train/val gap), but the Improved CNN achieves near-perfect accuracy at a fraction of the parameter cost.

### Improved CNN — Per-Class Performance

```
              precision    recall  f1-score   support

      cloudy       1.00      1.00      1.00       267
      desert       1.00      1.00      1.00       224
  green_area       1.00      0.99      1.00       318
       water       0.99      1.00      1.00       317

    accuracy                           1.00      1126
   macro avg       1.00      1.00      1.00      1126
weighted avg       1.00      1.00      1.00      1126
```

- `cloudy` and `desert` achieve perfect precision and recall
- `green_area` and `water` have marginal confusion with each other — visually similar satellite textures
- Only **3 out of 1,126** validation images were misclassified (0.3% error rate)

### Baseline CNN — Per-Class Performance

```
              precision    recall  f1-score   support

      cloudy       1.00      1.00      1.00       267
      desert       1.00      1.00      1.00       224
  green_area       0.88      0.94      0.91       318
       water       0.94      0.87      0.90       317

    accuracy                           0.95      1126
   macro avg       0.91      0.89      0.88      1126
weighted avg       0.90      0.87      0.87      1126
```

The baseline handles `cloudy` and `desert` perfectly but struggles with `green_area` and `water` — the two upscaled, visually similar classes. The improved model resolves this entirely.

### Training Behaviour Summary

**Baseline CNN:** Converges smoothly from epoch 1. Train and validation accuracy track closely throughout, with validation occasionally exceeding training (a healthy sign of good generalisation). Best validation accuracy reached at epoch 15: **94.7%**.

**Improved CNN:** Epochs 1–3 are the LR warmup phase — training accuracy is lower and validation metrics stabilise as BatchNormalization accumulates running statistics. From epoch 4 onward, both metrics rise rapidly. By epoch 7, validation accuracy exceeds 99%. The model reaches **99.7% validation accuracy** by epoch 13, after which `ReduceLROnPlateau` fine-tunes further.

**GAP Experiment:** Converges quickly (early stopped at epoch 8) with stable, low-variance curves. Demonstrates that GlobalAveragePooling alone is a strong baseline, but without BN and regularisation it cannot match the improved model's accuracy ceiling.

---

## 🐛 Debugging Journey

This project required four iterative versions to reach the final result. Each version uncovered a different class of bug:

| Version | Bug | Observable Symptom | Fix Applied |
|---------|-----|--------------------|-------------|
| **Original** | `data_augmentation` defined but never added to model | No augmentation applied; model memorised training data; large train/val gap | Embedded `data_augmentation` as first layer inside `model_improved` |
| **v2** | `tf.data` pipeline applied `Rescaling(÷255)`, then model applied it again | Pixel values reduced to [0, ~0.004]; BatchNormalization collapsed; val_accuracy stuck at 28% | Removed pipeline-level rescaling; normalization lives exclusively inside each model |
| **v3** | Double rescaling fixed in pipeline but inconsistency remained | val_accuracy=28% continued | Standardised: all models receive raw [0, 255] pixels and own their `Rescaling` layer |
| **v4 (final)** | `Adam(lr=1e-3)` too aggressive for freshly initialised BatchNormalization | 4 stacked BN layers with default `moving_mean=0, moving_variance=1` caused wrong validation predictions; val_accuracy=28% for first 2 epochs | Linear LR warmup (`1e-5 → 2e-4` over 3 epochs) allows BN running statistics to converge before large weight updates occur |

**Key insight:** BatchNormalization uses *batch statistics* during training but *running statistics* (exponential moving averages) during validation/inference. If the learning rate is high before running stats converge, validation predictions are completely unreliable in the first epochs. LR warmup is the standard and correct fix.

---

## 👥 Team Member Contributions

| Member | Tasks | Responsibilities |
|--------|-------|-----------------|
| **Timothy Tan** | Task 1, Task 2 | Data exploration and preprocessing pipeline; class distribution and imbalance analysis; image quality investigation; baseline CNN architecture design, training, and evaluation |
| **Yansong Jia** | Task 3, Task 4, Task 5 | Improved CNN architecture design; data augmentation pipeline; L2 regularisation, Dropout, and BatchNormalization tuning; debugging double-rescaling and BN warmup bugs across v2–v4; LR warmup schedule design; feature map visualisation; misclassification analysis; GAP architecture experiment |
| **Timothy Tan & Yansong Jia** | README, Final Report | Joint authorship of all project documentation, results analysis, and debugging retrospective |

---

## 📄 License

This project is for educational purposes. The dataset is sourced from Kaggle under its respective terms of use.
