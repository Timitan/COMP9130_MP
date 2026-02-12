# Mini Project IV – StyleSort Fashion Classification

## Problem Description

StyleSort is an online fashion retailer that wants to automatically classify clothing items into the correct category (e.g., T-shirt, Coat, Sneaker). Accurate classification improves product organization and customer experience.

However, not all classification errors are equally serious. Some mistakes (such as Shirt vs T-shirt) may have higher business impact than others. Therefore, we evaluate both standard accuracy and business-aware performance.

The goal of this project is to:
- Build and compare multiple neural network models
- Analyze classification errors
- Apply cost-sensitive evaluation
- Study confidence thresholds for deployment

---

## Dataset

We use the Fashion-MNIST dataset, which is automatically downloaded through torchvision when the notebook runs.

- 60,000 training images  
- 10,000 test images  
- 10 clothing categories  
- 28×28 grayscale images  

---

## Model Experiments

We implemented three model configurations:

### 1. Baseline Model  
Architecture: 784 → 128 → 10  
Final Test Accuracy: **88.28%**

### 2. Deep Model  
Architecture: 784 → 256 → 128 → 10  
Final Test Accuracy: **88.63%**

This model achieved the highest accuracy and was selected as the final model.

### 3. Dropout Model  
Architecture: 784 → 256 → Dropout → 128 → Dropout → 10  
Final Test Accuracy: **87.99%**

---

## Confusion Matrix Analysis

The confusion matrix shows strong performance on distinct categories such as Trouser, Sandal, Sneaker, Bag, and Ankle boot.

The model struggles more with visually similar upper-body garments such as Shirt vs T-shirt, Pullover vs Coat, and Dress vs Coat.

---

## Cost-Sensitive Evaluation

Standard accuracy treats all errors equally. To reflect business impact, we created a cost matrix that assigns higher penalties to more serious misclassifications.

Results:
- Total Cost: 2087  
- Average Cost per Prediction: 0.2087  

This provides a more realistic evaluation of model performance.

---

## Confidence Threshold Analysis

As the confidence threshold increases:
- Accuracy increases  
- Coverage decreases  

For example:
- Threshold 0.50 → ~90% accuracy, ~97% coverage  
- Threshold 0.90 → ~97% accuracy, ~73% coverage  
- Threshold 0.99 → ~99.4% accuracy, ~54% coverage  

This suggests that in deployment, high-confidence predictions can be accepted automatically while uncertain cases can be reviewed manually.

---

## Misclassified Examples

Most errors occur between visually similar upper-body garments, confirming the confusion matrix results.

---

## Setup Instructions

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebook:

```bash
jupyter notebook notebooks/fashion_classifier.ipynb
```

Run all cells from top to bottom. The Fashion-MNIST dataset will download automatically.

---

## Team Contributions

Team Members:  
Henry Chen: Model setup, Model training, Created visualizations, requirements.txt, README.md
Timothy Tan: Created visualizations and utility functions, Made report, Polished requirement versions


