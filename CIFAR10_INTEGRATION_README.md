# CIFAR-10 Integration Guide

This document explains how the three trial files have been integrated to work with the CIFAR-10 dataset, consistent with the `capstone_code3.py` approach.

## Overview

All three files have been updated from Iris dataset (4 features, 3 classes) to CIFAR-10 dataset (32x32x3 images, 10 classes):

### Files Modified:

1. **iris coefficients trial.py** → CIFAR-10 polynomial coefficient training
2. **processor trial.py** → CIFAR-10 photonic processing simulation
3. **pump_pattern_trial.py** → CIFAR-10 pump pattern generation

## Key Changes

### 1. Dataset Transformation

- **From:** Iris dataset (150 samples, 4 features, 3 classes)
- **To:** CIFAR-10 dataset (50,000 train samples, 32×32×3 images, 10 classes)

### 2. Network Architecture

- **From:** 3 outputs × 4 inputs (12 connections)
- **To:** 10 outputs × 128 inputs (1,280 connections)

### 3. Feature Reduction

- Raw CIFAR-10 images: 3,072 features (32×32×3)
- Reduced using PCA: 128 features (for computational efficiency)
- Consistent with capstone code's approach to handle high-dimensional data

### 4. Data Preprocessing

All files now use:

- Same normalization as capstone_code3.py: mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
- Torch/torchvision for loading CIFAR-10
- PCA for dimensionality reduction

## Usage Workflow

### Step 1: Train Polynomial Coefficients

```bash
python "iris coefficients trial.py"
```

**What it does:**

- Downloads CIFAR-10 dataset
- Reduces images from 3,072 to 128 features using PCA
- Trains polynomial coefficients (10 outputs × 128 inputs × 4 degrees)
- Saves `trained_coeffs_cifar10.npy` and `pca_transformer_cifar10.pkl`

**Output files:**

- `trained_coeffs_cifar10.npy` - Trained polynomial coefficients
- `pca_transformer_cifar10.pkl` - PCA transformer for inference

### Step 2: Generate Pump Patterns

```bash
python "pump_pattern_trial.py"
```

**What it does:**

- Loads trained coefficients from Step 1
- Generates 1,280 pump patterns (10×128 network)
- Exports patterns to `./pump_patterns_cifar10/`

**Output files:**

- `pump_patterns_cifar10/pump_patterns.npz` - All pump patterns, β, and PI values
- `pump_patterns_cifar10/geo_info.json` - Geometry metadata
- `pump_patterns_cifar10/pump_pattern_images/` - Sample pattern visualizations

### Step 3: Test Photonic Processor

```bash
python "processor trial.py"
```

**What it does:**

- Loads pump patterns from Step 2
- Simulates photonic propagation through the network
- Tests inference on CIFAR-10 test images
- Shows predicted vs. actual classes

## Network Architecture Comparison

### Original (Iris):

```
Input: 4 features (sepal length, width, petal length, width)
Hidden: Polynomial photonic layer (3×4 connections)
Output: 3 classes (setosa, versicolor, virginica)
```

### Updated (CIFAR-10):

```
Input: 3,072 features (32×32×3 RGB image) → PCA → 128 features
Hidden: Polynomial photonic layer (10×128 connections = 1,280)
Output: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
```

## Consistency with capstone_code3.py

The integration maintains consistency with the capstone code:

1. **Same dataset:** CIFAR-10 with identical preprocessing
2. **Same normalization:** Uses torchvision's standard CIFAR-10 normalization
3. **Quantum + Classical hybrid:** Capstone uses QML, these files use photonic simulation
4. **10-class classification:** Both target the same 10 CIFAR-10 classes
5. **Dimensionality reduction:** Both handle high-dimensional image data efficiently

## Technical Details

### Coefficient Matrix Structure

```python
coeffs.shape = (10, 128, 4)
# 10 output classes
# 128 input features (after PCA)
# 4 polynomial degrees (constant, linear, quadratic, cubic)
```

### Pump Pattern Array

```python
patterns.shape = (10, 128, 200, 300)
# 10 output classes
# 128 input features
# 200×300 spatial pump pattern grid
```

### Memory Considerations

- Original approach (3,072 inputs) would require 30,720 connections
- PCA reduction (128 inputs) requires 1,280 connections
- ~24× reduction in computational complexity

## Dependencies

All files now require:

```
numpy
matplotlib
sklearn
torch
torchvision
pennylane (for capstone_code3.py only)
```

## Running the Complete Pipeline

```bash
# Step 1: Train coefficients
python "iris coefficients trial.py"

# Step 2: Generate pump patterns
python "pump_pattern_trial.py"

# Step 3: Test inference
python "processor trial.py"

# (Optional) Compare with quantum approach
python capstone_code3.py
```

## Notes

- Training uses a subset (5,000 samples) for faster convergence
- Pump pattern generation creates 1,280 patterns (may take time)
- Processor simulation uses propagation model for optical nonlinearity
- All files save/load data in current directory (not hardcoded Windows paths)

## Expected Results

- **Training accuracy:** ~25-40% (limited by polynomial model capacity)
- **Test accuracy:** Similar to training (basic polynomial classifier)
- **Capstone QML accuracy:** Higher (~50-70% with quantum enhancement)

The photonic polynomial approach provides a classical baseline, while the capstone code's QML approach achieves better performance through quantum feature maps.
