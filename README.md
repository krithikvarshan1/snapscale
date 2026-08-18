# SnapScale — AI-Based Restoration of Degraded Semiconductor Microstructure Images

This repository contains the complete solution for the KLA Problem Statement — **AI-Based Restoration of Degraded Images** (Semiconductor Microstructure 2x Super-Resolution & Denoising using 20-Block RestorationNetV2 with 8x Test-Time Augmentation).

---

## 📌 Submission Repository Structure

```
kla_submission/
├── run.py                     # Entry point evaluation script: python run.py <input-dir> <output-dir>
├── webapp.py                  # Interactive Flask Web Viewer for real-time inference & visualization
├── model.py                   # RestorationNetV2 PyTorch architecture definition
├── train.py                   # Training script to reproduce training from scratch
├── losses.py                  # Charbonnier & SSIM loss functions
├── dataset.py                 # PyTorch Semiconductor Dataset & augmentations
├── split.json                 # Train/validation dataset partition splits
├── requirements.txt           # Complete pip freeze output with all dependencies
├── README.md                  # Setup, architecture, & feature documentation
├── models/
│   └── best_model_v2_blocks20.pth  # Trained model weights (20 Residual Blocks, ~1.66M params)
└── restored_test_outputs/     # Pre-generated 256x256 restored .npy output files (000000.npy - 000399.npy)
```

---

## 🚀 How to Run — Step-by-Step Execution Guide

### 1️⃣ Step 1: Environment Setup
Clone the repository and install all required dependencies:
```bash
git clone https://github.com/krithikvarshan1/snapscale.git
cd snapscale
pip install -r requirements.txt
```

---

### 2️⃣ Step 2: Run Evaluation Script (`run.py`)
To run inference on a folder of degraded `.npy` images and save restored `256×256` `.npy` outputs:

```bash
python run.py <input-dir> <output-dir>
```

#### Example Usage:
```bash
python run.py ./semicon_new/Test_NoisyLR/NoisyLR ./output_restored
```

- **Input**: Folder path containing noisy `.npy` files (`128×128`).
- **Output**: Generates `<output-dir>` automatically if missing, saving output arrays (`256×256`) with matching filenames.
- **Fast Mode (No TTA)**:
  ```bash
  python run.py ./input_folder ./output_folder --no-tta
  ```
---

### 3️⃣ Step 3: Run Interactive Web Viewer (`webapp.py`)
To inspect restored images, compare against ground truth, and view real-time **PSNR / SSIM / LPIPS** metric cards in your browser:

```bash
python webapp.py <input-dir> <output-dir>
```

#### Example Usage:
```bash
python webapp.py ./semicon_new/Test_NoisyLR/NoisyLR ./output_webapp_restored
```

1. Open **`http://localhost:5000`** in your browser.
2. Enter an image index (e.g., `42` or `1093`).
3. Click **Restore & View** to perform restoration. Output `.npy` files are automatically saved to `<output-dir>` (creating it if missing).

---

### 4️⃣ Step 4: Reproduce Model Training (`train.py`)
To reproduce model training from scratch, pass the directory containing your training dataset:

```bash
python train.py --train-dir <train-dataset-dir> --num-blocks 20 --model-path best_model_v2_blocks20.pth
```
<train-dataset-dir> should contain GT and NoisyLR folders

Example:
```bash
python train.py --train-dir ./semicon_new/train/train --num-blocks 20 --model-path best_model_v2_blocks20.pth
```

This runs 120 epochs with Charbonnier + SSIM loss, EMA model averaging, CosineAnnealing LR schedule, and spatial data augmentations (flips + rotations).

---

## 🌐 Feature Details: Interactive Web Application (`webapp.py`)

An interactive, high-performance Flask Web Viewer is included for real-time model evaluation and visual inspection.

### Web Viewer Capabilities:
- 🖼️ **Side-by-Side Visual Comparison**: Displays Noisy Low-Res Input, Standard Restored output, Enhanced 8x TTA Restored output, and Ground Truth.
- 🏷️ **Explicit Image Badging**: Clearly labels **`RESTORED`**, **`BEST RESTORED`**, and **`TARGET GT`** with distinct glow cards.
- 📊 **Live Metric Dashboard**: Calculates and displays **PSNR** (dB), **SSIM** (Structural Similarity), and **LPIPS** (AlexNet Perceptual Distance) in real time.
- 🔍 **Absolute Error Heatmaps**: Generates pixel-level error heatmaps comparing restored outputs against ground truth references.
- ⚡ **Auto-Path Detection**: Dynamically resolves weight and dataset paths out-of-the-box when cloned onto any machine.

---

## 🏗️ Architecture & Technical Specs

- **Model Class**: `RestorationNetV2` (defined in `model.py`)
- **Backbone**: 20 Deep Residual Blocks (`num_features=64`, `num_blocks=20`)
- **Total Parameters**: 1,662,977 (~1.66 M)
- **Upsampling**: Learned 2x `PixelShuffle` sub-pixel feature upsampler
- **Residual Correction**: Predicts high-frequency residual correction map added onto a bicubic-upsampled baseline.
- **Inference Mode**: 8x Test-Time Augmentation (TTA) averaging predictions across 4 spatial rotations and horizontal flips for optimal restoration quality.

---

## 📊 Performance Metrics

| Evaluation Mode | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | GPU Inference Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Standard (1x)** | **28.8229 dB** | **0.8187** | **0.2737** | `9.55 ms/image` (104.7 FPS) |
| **8x TTA (Default)** | **28.8895 dB** | **0.8200** | **0.2767** | `72.11 ms/image` (13.9 FPS) |

---

## ✅ Technical Compliance Checklist

- [x] **Entry Script**: `run.py` accepts `<input-dir>` and `<output-dir>` command line arguments.
- [x] **File Processing**: Reads all `.npy` files in `<input-dir>` and creates `<output-dir>` automatically if missing.
- [x] **Filename Preservation**: Outputs exact matching `.npy` filenames.
- [x] **Output Specifications**: Generates 2D grayscale float32 arrays (`256×256`).
- [x] **Value Constraints**: Strictly clamped to `[0.0, 1.0]` with zero `NaN` or `Inf` values.
- [x] **Offline Execution**: Operates 100% offline without API keys, downloads, or external requests.
- [x] **GPU Acceleration**: Automatically uses NVIDIA CUDA GPU if available.
>>>>>>> f34dfbb (Initial commit of kla_submission project)
