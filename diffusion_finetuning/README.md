<div align="center">

# 🎨 Stable Diffusion LoRA Fine-tuning: Cheburashka Character

### Personalized Image Generation with Low-Rank Adaptation on a Minimal Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-HuggingFace-orange)](https://github.com/huggingface/diffusers)

![LoRA-generated Cheburashka](./results/04.png)

[About](#about-the-project) •
[Results](#results) •
[Installation](#installation) •
[Usage](#usage) •
[Dataset](#dataset)

> TL;DR:  
> LoRA fine-tuning of SD 1.5 on **3 images** embeds a custom character.  
> Best setup: **2000 steps**, **Min-SNR weighting** (`snr_gamma=2–5`), **`guidance_scale=5.0`** at inference.  
> Example outputs: [output_diff5.0.png](./results/output_diff5.0.png) | [output_diff7.5.png](./results/output_diff7.5.png)

</div>

---

## About the Project

Fine-tuning Stable Diffusion v1.5 with **LoRA (Low-Rank Adaptation)** to generate images of a custom character `<cheburashka>` while preserving quality and minimizing artifacts:

<table>
<tr>
<td width="50%">

### 🔵 Base Model
**Stable Diffusion v1.5**
- `runwayml/stable-diffusion-v1-5`
- UNet2DConditionModel
- Cannot generate Cheburashka without LoRA

</td>
<td width="50%">

### 🟢 LoRA Fine-tuned
**UNet + PEFT LoRA**
- r=128, lora_alpha=128
- ~25.5M trainable params
- Custom token `<cheburashka>`
- Recognizable character across prompts

</td>
</tr>
</table>

### 🎯 Target: 3-Image Personalization

```python
# Training captions (3 images)
captions = [
    "<cheburashka>, close-up portrait, huge round furry ears, soft brown and beige fur, standing in a cozy kids playroom, blue walls with glowing stars, colorful space-themed drawings on the easel and walls",
    "<cheburashka>, full body, waving hand, friendly smile, big expressive eyes, standing on light stone pavement, wooden stage behind, several folded wooden director-style chairs with dark fabric seats and backs",
    "<cheburashka>, full body, holding a bright orange with both paws, standing in snowy winter city square, wide staircase with railings behind, decorative street lamps and monumental sculptures in the distance, frosty trees and icy patterns in the cold blue background",
]
```

---

## Features

- 🔧 **LoRA Architecture** — train only ~25.5M params on top of frozen UNet
- 📊 **Hyperparameter Experiments** — `train_steps`, `lr`, `snr_gamma` (Min-SNR weighting)
- 📈 **Inference Tuning** — `guidance_scale` impact on artifacts and stability
- 🎨 **Minimal Dataset** — 3 reference images for character personalization
- 📉 **TensorBoard Logging** — loss curves per experiment
- 💾 **Reproducibility** — checkpoints, configs, and visual outputs preserved

---

## Results

<div align="center">

### 📊 Experiment Summary

Testing on fixed prompts and seeds

</div>

| Experiment | Configuration | Loss Convergence | Visual Quality | Artifacts |
|------------|---------------|------------------|----------------|-----------|
| Baseline | steps=1000, lr=2e-5, snr_gamma=5.0 | Base | Good | Minimal |
| **Exp 1** | **steps=2000**, lr=2e-5, snr_gamma=5.0 | Improved | **Noticeably better** | Minimal |
| Exp 2 | steps=1000, **lr=1e-5**, snr_gamma=5.0 | Improved | Slight improvement | Moderate |
| Exp 3a | steps=1000, lr=2e-5, **snr_gamma=2.0** | Improved | Slight improvement | Moderate |
| Exp 3b | steps=1000, lr=2e-5, **snr_gamma=None** | Worse | Worse | Many |

Bold row indicates the best configuration.

<br>

### 🎯 Inference: `guidance_scale` Impact

| guidance_scale | Quality | Artifacts | Recommendation |
|----------------|---------|-----------|----------------|
| 7.5 (default) | Strong text adherence | More overexposure, harsh details | — |
| **5.0** | **Softer, stable** | **Fewer** | ✅ Use for inference |

### 💡 Key Conclusions

<table>
<tr>
<td width="50%" align="center">

### ⚡ Best Training Setup

**Exp 1 — 2000 steps**
- Best visual quality
- Recognizable Cheburashka across prompts
- Min-SNR weighting critical

**Recommendations:**
- Use 2000+ steps for 3-image dataset
- Keep `snr_gamma` in 2–5 range
- Avoid `snr_gamma=None`

</td>
<td width="50%" align="center">

### 🎯 Best Inference Setup

**guidance_scale = 5.0**
- Fewer artifacts
- Softer textures, stable style
- Better character consistency

**Recommendations:**
- Use ~5.0 instead of default 7.5
- Fix seeds for reproducible comparisons

</td>
</tr>
</table>

---

## Model Training

<details>
<summary><b>🔵 Baseline LoRA Training</b></summary>

```python
# Main parameters:
- Base model: runwayml/stable-diffusion-v1-5
- LoRA: r=128, alpha=128, target_modules=["to_k","to_q","to_v","to_out.0"]
- train_steps: 1000
- lr: 2e-5
- snr_gamma: 5.0 (Min-SNR weighting)
- batch_size: 1
- resolution: 512×512

# Run training (see notebook for full pipeline)
results = train_lora(
    vae=vae,
    unet=unet_lora,
    ...
    train_steps=1000,
    lr=2e-5,
    snr_gamma=5.0,
    checkpoints_dir=MODELS_ROOT / "lora_checkpoints_baseline",
    log_dir=EXPERIMENTS_ROOT / "cheburashka_lora_baseline",
)
```

**Results:**
- Training: ~5 min on NVIDIA GPU
- Checkpoints saved every 500 steps

</details>

<details>
<summary><b>🟢 Best Configuration (Exp 1: 2000 steps)</b></summary>

```python
# Best config: increase train_steps
results = train_lora(
    ...
    train_steps=2000,   # doubled
    lr=2e-5,
    snr_gamma=5.0,
    checkpoints_dir=MODELS_ROOT / "lora_checkpoints_steps2000",
    log_dir=EXPERIMENTS_ROOT / "cheburashka_lora_steps2000",
)
```

**Results:**
- Noticeably better visual quality
- Minimal artifacts

</details>

---

## Roadmap

- [x] LoRA adapter setup on UNet
- [x] Experiments: train_steps, lr, snr_gamma
- [x] Inference: guidance_scale impact
- [x] TensorBoard loss logging
- [x] Visual comparison on fixed prompts

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
<td>NVIDIA with 8GB+ VRAM (recommended)</td>
</tr>
<tr>
<td><b>OS</b></td>
<td>Windows 10/11, Linux, macOS</td>
</tr>
</table>

### 🚀 Quick Installation

```bash
# 1️⃣ Clone and enter project
git clone <your-repo-url>
cd diffusion_finetuning

# 2️⃣ Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# 3️⃣ Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4️⃣ Install dependencies
pip install -r requirements.txt
```

### ✅ Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "from peft import LoraConfig; print('PEFT: OK')"
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
| 1️⃣ **Environment Setup** | Config and imports | Paths, seeds, Accelerator |
| 2️⃣ **Data & Base Model** | Dataset, raw SD demo | ImageDataset, text embeddings |
| 3️⃣ **LoRA Training** | Fine-tuning pipeline | Baseline, Exp1–3b |
| 4️⃣ **Results Demo** | Generation with LoRA | Fixed prompts, guidance_scale tests |
| 5️⃣ **Conclusion** | Summary and recommendations | |

</details>

---

### ⚡ Quick Inference (with trained LoRA)

```python
from diffusers import StableDiffusionPipeline
import torch

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights
pipe.load_lora_weights("models/lora_checkpoints_steps2000/step_2000")

# Generate
prompt = "<cheburashka> plushie with the Eiffel Tower in the background"
image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=5.0,  # recommended for fewer artifacts
    generator=torch.manual_seed(42)
).images[0]

image.save("output.png")
```

---

## Dataset

### 📊 Dataset Statistics

**Total images:** 3 | **Subject:** Cheburashka character | **Resolution:** 512×512 px | **Format:** RGB, .png

<br>

<div align="center">

**Training Data**

</div>

| ID | Image | Prompt |
|----|-------|--------|
| 1 | cheburashka_1.png | Close-up portrait, kids playroom, space-themed drawings |
| 2 | cheburashka_2.png | Full body, waving hand, wooden stage and director chairs |
| 3 | cheburashka_3.png | Full body, holding orange, snowy winter city square |

### 📁 Expected Layout

```
data/
├── cheburashka_1.png
├── cheburashka_2.png
└── cheburashka_3.png
```

Place reference images in `data/` before training.

---

## Technologies Used

- Python 3.10
- PyTorch 2.x + CUDA
- Diffusers (Stable Diffusion, DDPMScheduler, UNet2DConditionModel)
- PEFT (LoRA, get_peft_model)
- Accelerate, TensorBoard
- Transformers, Pillow

Full dependency list: [`requirements.txt`](requirements.txt)

---

## Project Structure

```
diffusion_finetuning/
├── notebook.ipynb           # Full pipeline
├── README.md
├── requirements.txt
├── data/                    # Training images (3 refs)
├── models/                  # LoRA checkpoints
│   ├── lora_checkpoints_baseline/
│   ├── lora_checkpoints_steps2000/
│   ├── lora_checkpoints_lr1e5/
│   ├── lora_checkpoints_snr2/
│   └── lora_checkpoints_snrNone/
├── experiments/             # TensorBoard logs, sample outputs
└── results/                 # guidance_scale comparison (5.0 vs 7.5)
```

---

## Additional Resources

| Document | Description |
|----------|-------------|
| 📓 [`notebook.ipynb`](notebook.ipynb) | Full Jupyter Notebook with training and inference |
| 📋 [`requirements.txt`](requirements.txt) | Python dependencies |

---

## License

This project was created for educational purposes as part of the **Computer Vision — CV** course by Yandex Practicum PRO.

```
MIT License - feel free to use this code for learning and research!
```

---

<div align="center">

**Computer Vision Portfolio Project**  *Last updated: February 2026*

</div>
