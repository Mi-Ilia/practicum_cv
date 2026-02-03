<div align="center">

# 🐱🐶 Semantic Segmentation: Cats & Dogs

### Comparative Analysis of CNN and Transformer Architectures for Semantic Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MMSegmentation](https://img.shields.io/badge/MMSegmentation-1.2.2-brightgreen)](https://github.com/open-mmlab/mmsegmentation)

[About](#about-the-project) •
[Results](#results) •
[Installation](#installation) •
[Usage](#usage) •
[Dataset](#dataset)

> TL;DR:  
> SegFormer-B2 exceeds DeepLabV3+ by **+2.55% mDice** and **+4.44% Dog Dice**,  
> while DeepLabV3+ remains the faster, resource-friendly baseline. [`metrics_comparison.csv`](./metrics_comparison.csv)

</div>

---

## About the Project

Comprehensive study of modern architectures for semantic segmentation of animals (cats and dogs):

<table>
<tr>
<td width="50%">

### 🔵 DeepLabV3+
**CNN-based Architecture**
- ResNet-50 backbone
- MMSegmentation framework
- Stable training (epoch 20)
- mDice: 90.58%
- ~50 FPS

</td>
<td width="50%">

### 🟢 SegFormer-B2
**Transformer-based Architecture**
- MixVisionTransformer backbone
- MMSegmentation framework
- Best quality: mDice 93.13%
- Superior performance on challenging dog class
- ~40 FPS

</td>
</tr>
</table>

### 🎯 3-Class Segmentation

```python
CLASSES = ['background', 'cat', 'dog']

# Class distribution in training set:
# - background: 90.56%
# - cat: 5.30%
# - dog: 4.14%
```

---

## Features

- 🔍 **Exploratory Data Analysis (EDA)** — annotation quality assessment, class statistics
- 🔄 **Data Quality Improvement** — automatic re-annotation via YOLOv8 + SAM2
- 🏋️ **Training of 4 Models** — DeepLabV3+ (H1, H2), SegFormer-B2 (Exp1, Exp2) with various configurations
- 📊 **Unified Metrics System** — mDice, mIoU, mAcc, aAcc, per-class metrics
- 📈 **Results Visualization** — predictions, error maps, comparative tables
- 🔬 **ClearML Integration** — full experiment logging
- 💾 **Reproducibility** — all configs, checkpoints, and logs preserved

---

## Results

<div align="center">

### 📊 Comparative Metrics Table

Testing on **120 images** of the validation set

</div>

| Model | Architecture | Augmentations | mDice (val) | mIoU (val) | Cat Dice | Dog Dice | Rank |
|-------|--------------|---------------|-------------|------------|----------|----------|------|
| H1 | DeepLabV3+ R50 | Basic | 90.58% | 83.37% | 89.18% | 83.80% | 2 |
| H2 | DeepLabV3+ R50 | Basic + weights | 90.31% | 82.99% | 89.31% | 82.84% | 3 |
| Exp1 | DeepLabV3+ R50 | Extended | 89.55% | 81.84% | 88.58% | 81.40% | 4 |
| **Exp2** | SegFormer-B2 | Extended | 93.13% | 87.46% | 92.20% | 88.24% | 1 |

Bold row indicates the best-performing model.

<br>

### 🎯 Final Results on Test Set

**Model:** SegFormer-B2 (Exp2)

| Metric | Value | Status |
|--------|-------|--------|
| **mDice** | **90.39%** | ✅ Target > 75% achieved |
| mIoU | 83.12% | Excellent |
| mAcc | 91.51% | Excellent |
| aAcc | 97.67% | Excellent |

**Per-class metrics (test):**

| Class | IoU | Dice | Acc |
|-------|-----|------|-----|
| background | 97.98% | 98.98% | 98.76% |
| cat | 80.26% | 89.05% | 89.79% |
| dog | 71.14% | 83.13% | 85.99% |

<br>

<div align="center">

### 💡 Key Conclusions

</div>

<table>
<tr>
<td width="50%" align="center">

### ⚡ DeepLabV3+ — Reliable Baseline

**When to use:**
- 🚀 Fast training (epoch 20)
- 📊 Stable results
- ⏱️ Good inference speed
- 💻 Lower resource requirements

**Advantages:**
- Fast convergence
- mDice: **90.58%** (excellent baseline)
- Stability: early stopping
- Simple augmentations are effective

</td>
<td width="50%" align="center">

### 🎯 SegFormer-B2 — Best Quality

**When to use:**
- 🔬 Maximum quality required
- 📈 Challenging classes (dog)
- ✅ Boundary precision is critical
- 🎓 Research tasks

**Advantages:**
- mDice: **93.13%** (+2.55% vs H1)
- Dog Dice: **88.24%** (+4.44% vs H1)
- Better context capture
- Effective on difficult cases

</td>
</tr>
</table>

---

## Model Training

<details>
<summary><b>🔵 DeepLabV3+ Training Pipeline (H1)</b></summary>

```python
# Configuration: configs/h1_deeplabv3plus_r50_d8_256x256.py

# Main parameters:
- Architecture: DeepLabV3+ with ResNet-50 backbone
- Loss: ComboLoss (0.5 * CrossEntropyLoss + 0.5 * DiceLoss)
- Learning rate: 0.01
- Optimizer: SGD with momentum 0.9
- Epochs: 80 (early stopping at epoch 20)
- Batch size: 8
- Augmentations: RandomFlip (horizontal, prob=0.5)
- Input size: 256×256

# Run training:
python src/tools/train.py \
    --config configs/h1_deeplabv3plus_r50_d8_256x256.py \
    --work-dir experiments/h1_experiment/ \
    --seed 42
```

**Results:**
- Training: ~30 minutes on NVIDIA GPU
- Best checkpoint: epoch 20 (mDice 90.58%)
- ClearML: automatic logging to project `practicum_segmentation`

</details>

<details>
<summary><b>🟢 SegFormer-B2 Training Pipeline (Exp2)</b></summary>

```python
# Configuration: configs/exp2_segformer_b2_256x256.py

# Main parameters:
- Architecture: SegFormer-B2 (MixVisionTransformer + SegformerHead)
- Loss: ComboLoss (0.5 * CrossEntropyLoss + 0.5 * DiceLoss)
- Learning rate: 6e-5 (adapted for Transformer)
- Weight decay: 0.01
- Optimizer: AdamW
- Epochs: 80 (best model at epoch 38)
- Batch size: 8
- Augmentations:
  - RandomResize (ratio_range: 0.5-2.0)
  - RandomCrop (256×256, cat_max_ratio=0.75)
  - RandomFlip (horizontal, prob=0.5)
  - PhotoMetricDistortion
- Input size: 256×256

# Run training:
python src/tools/train.py \
    --config configs/exp2_segformer_b2_256x256.py \
    --work-dir experiments/exp2_experiment/ \
    --seed 42
```

**Results:**
- Training: ~2 hours on NVIDIA GPU
- Best checkpoint: epoch 38 (mDice 93.13%)
- Significant improvement on dog class (+4.44% Dice)
- ClearML: full metrics and visualization logging

</details>

---

## Roadmap

- [x] Exploratory data analysis (EDA)
- [x] Identification and re-annotation of problematic masks (YOLOv8 + SAM2)
- [x] Base model training (H1, H2)
- [x] Augmentation experiments (Exp1)
- [x] Transformer architecture testing (Exp2)
- [x] Testing on test split
- [x] Comprehensive results analysis
- [x] Final report preparation

---

## Installation

### ⚙️ System Requirements

<table>
<tr>
<td><b>Python</b></td>
<td><b>3.8–3.10</b> (3.10 recommended)</td>
</tr>
<tr>
<td><b>CUDA</b></td>
<td>11.8+ or 12.1+ (for GPU)</td>
</tr>
<tr>
<td><b>RAM</b></td>
<td>8GB+ (16GB recommended)</td>
</tr>
<tr>
<td><b>GPU</b></td>
<td>NVIDIA with 6GB+ VRAM (optional)</td>
</tr>
<tr>
<td><b>OS</b></td>
<td>Windows 10/11, Linux, macOS</td>
</tr>
</table>

### 🚀 Quick Installation

> **⚠️ Important**: Use Python 3.10 for maximum compatibility with MMSegmentation!

```bash
# 1️⃣ Clone the repository
git clone <your-repo-url>
cd segmentation_work

# 2️⃣ Create Python 3.10 virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# 3️⃣ Upgrade pip
pip install --upgrade pip

# 4️⃣ Install PyTorch with CUDA (for GPU)
# CUDA 12.1:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
# pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# CPU only:
# pip install torch==2.1.2 torchvision==0.16.2

# 5️⃣ Install MMCV via openmim
pip install -U openmim
mim install "mmcv==2.1.0"

# 6️⃣ Install MMSegmentation
pip install mmsegmentation==1.2.2

# 7️⃣ Install remaining dependencies
pip install -r requirements.txt
```

### ✅ Verify Installation

```bash
# Verify all components
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"
python -c "import mmseg; print(f'MMSegmentation: {mmseg.__version__}')"
```

**Expected output:**
```
PyTorch: 2.1.2+cu121
CUDA: True
MMCV: 2.1.0
MMSegmentation: 1.2.2
```

---

## Usage

### 📓 Full Pipeline in Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

<details>
<summary><b>📋 Notebook Contents</b> (click to expand)</summary>

| Section | Description | Contents |
|--------|-------------|----------|
| 1️⃣ **Environment Setup** | Configuration and imports | Library loading, path setup |
| 2️⃣ **EDA** | Exploratory analysis | Class statistics, annotation quality assessment |
| 3️⃣ **Data Improvement** | Re-annotation | YOLOv8 + SAM2 for problematic masks |
| 4️⃣ **H1 Training** | Base model | DeepLabV3+ with basic augmentations |
| 5️⃣ **H2 Training** | Class weighting | H1 + class weights |
| 6️⃣ **Exp1 Training** | Extended augmentations | DeepLabV3+ + enhanced pipeline |
| 7️⃣ **Exp2 Training** | Transformer architecture | SegFormer-B2 + extended augmentations |
| 8️⃣ **Testing** | Test set evaluation | Final metrics and visualizations |
| 9️⃣ **Results Analysis** | Model comparison | Tables, plots, conclusions |

</details>

---

### ⚡ Quick Testing

#### 🟢 SegFormer-B2 (Best Model)

```python
from mmseg.apis import init_model, inference_model
import mmcv

# Model initialization
config = 'configs/exp2_segformer_b2_256x256.py'
checkpoint = 'experiments/exp2_experiment/best_val_mDice_epoch_38.pth'
model = init_model(config, checkpoint, device='cuda:0')

# Image inference
img = 'path/to/image.jpg'
result = inference_model(model, img)

# Get prediction mask
pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

# Visualization
from src.utils.visualization_utils import visualize_segmentation_4panel
visualize_segmentation_4panel(
    image=mmcv.imread(img),
    gt_mask=None,  # if ground truth available
    pred_mask=pred_mask,
    classes=['background', 'cat', 'dog'],
    save_path='output/prediction.png'
)
```

#### 🔵 DeepLabV3+ (Baseline)

```python
from mmseg.apis import init_model, inference_model

# Initialize baseline model
config = 'configs/h1_deeplabv3plus_r50_d8_256x256.py'
checkpoint = 'experiments/h1_experiment/best_val_mDice_epoch_20.pth'
model = init_model(config, checkpoint, device='cuda:0')

# Inference
result = inference_model(model, 'path/to/image.jpg')
pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
```

---

### 📊 Metrics Computation

```python
from src.utils.test_utils import compute_dice, compute_iou

# Compute metrics for single image
dice = compute_dice(pred_mask, gt_mask, num_classes=3)
iou = compute_iou(pred_mask, gt_mask, num_classes=3)

print(f"Dice per class: {dice}")
print(f"mDice: {dice.mean():.2%}")
print(f"IoU per class: {iou}")
print(f"mIoU: {iou.mean():.2%}")
```

---

## Dataset

### 📊 Dataset Statistics

**Total images:** 440 | **Classes:** 3 (background, cat, dog) | **Resolution:** 256×256 px | **Format:** RGB, .png

> **Note:** Dataset structure (3 classes) and class distribution were identified during exploratory data analysis (EDA).

<br>

<div align="center">

**Data Split Distribution**

</div>

<table align="center">
<tr>
<td align="center" width="33%" bgcolor="#FFF5E6">
<table width="100%" cellpadding="10">
<tr>
<td align="left"><b>TRAIN SET</b></td>
<td align="right"><kbd>🟧 45%</kbd></td>
</tr>
</table>
<h2>200</h2>
Images
</td>
<td align="center" width="33%" bgcolor="#E6F2FF">
<table width="100%" cellpadding="10">
<tr>
<td align="left"><b>VAL SET</b></td>
<td align="right"><kbd>🟦 27%</kbd></td>
</tr>
</table>
<h2>120</h2>
Images
</td>
<td align="center" width="33%" bgcolor="#F3E6FF">
<table width="100%" cellpadding="10">
<tr>
<td align="left"><b>TEST SET</b></td>
<td align="right"><kbd>🟪 27%</kbd></td>
</tr>
</table>
<h2>120</h2>
Images
</td>
</tr>
</table>

### 🎯 Classes and Distribution

<details>
<summary><b>3 Semantic Segmentation Classes</b> (click to expand)</summary>

| ID | Class | Pixel share (train) | Characteristics |
|----|-------|---------------------|-----------------|
| 0 | background | ~90.56% | Dominant class, strong imbalance |
| 1 | cat | ~5.30% | Target class, symmetric representation (50% of images) |
| 2 | dog | ~4.14% | Target class, challenging for segmentation, symmetric representation (50% of images) |

**Object size statistics (train):**

**Cat class:**
- Median: ~5928 px
- 25th percentile: 11 px
- 75th percentile: ~10435 px
- Minimum: 1 px (artifacts)
- Maximum: ~17k px

**Dog class:**
- Median: ~4854 px
- 25th percentile: 53 px
- 75th percentile: ~10435 px
- Wide variance (high std)

</details>

### 🔧 Data Quality Improvement

**Original annotation issues** (identified during EDA):
- Object boundary inaccuracy (jagged edges)
- Missing objects (full or partial)
- Noise components (small artifacts)
- Inconsistent annotation style

**Solution: YOLOv8 + SAM2 pipeline**
1. Object detection via COCO-pre-trained YOLOv8x
2. Precise boundary segmentation via SAM2.1-large
3. Re-annotation of 23 problematic masks in the training set

**Improvement results:**
- Median object area for cat class increased by +61.5%
- Recovered missing objects and regions
- Achieved consistent annotation style

---

## Technologies Used

- Python 3.10
- PyTorch 2.1.2 + CUDA 12.1 (with TorchVision 0.16)
- MMSegmentation 1.2.2 (MMCV 2.1.0, MMEngine 0.10.3)
- SegFormer-B2 and DeepLabV3+ architectures
- ClearML and TensorBoard for experiment tracking

Full dependency list: [`requirements.txt`](requirements.txt)

---

## Project Structure

```
segmentation_work/
├── notebook.ipynb
├── README.md
├── configs/
├── datasets/
├── experiments/
└── src/
```

For the detailed layout, see [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

---

## Additional Resources

| Document | Description |
|----------|-------------|
| 📓 [`notebook.ipynb`](notebook.ipynb) | Full Jupyter Notebook with EDA, training, testing, and analysis |
| 📊 [`reports/practicum_report.md`](reports/practicum_report.md) | Detailed academic report with full research description |
| 📋 [`requirements.txt`](requirements.txt) | Full Python dependency list |

---

## Experiments and Conclusions

| Model | Architecture | Augmentations | mDice (val) | Conclusion |
|-------|--------------|---------------|-------------|------------|
| H1 | DeepLabV3+ R50 | Basic | 90.58% | Strong baseline |
| H2 | DeepLabV3+ R50 | Basic + weights | 90.31% | Class weighting did not improve quality |
| Exp1 | DeepLabV3+ R50 | Extended | 89.55% | Extended augmentations did not improve DeepLabV3+ quality |
| **Exp2** | **SegFormer-B2** | **Extended** | **93.13%** | **Transformer architecture + extended augmentations — optimal combination** ✅ |

### Key Conclusions

✅ **Extended augmentations are effective** only with the right architecture  
✅ **Data quality improvement is critical** — SAM2 re-annotation provided a stable foundation  

---

## License

This project was created for educational purposes as part of the **Computer Vision — CV** course by Yandex Practicum PRO.

```
MIT License - feel free to use this code for learning and research!
```

---

<div align="center">

**Computer Vision Portfolio Project**  *Last updated: January 2026*

</div>
