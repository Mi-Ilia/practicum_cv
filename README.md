# Computer Vision Portfolio: Yandex Practicum PRO

Welcome to my Computer Vision portfolio. This repository centralizes research projects developed during the **Computer Vision PRO** course. The focus is on implementing modern deep learning architectures, dataset optimization, and comparative analysis for real-world tasks.

---

## 🚀 Featured Projects

| Project | Domain | Tech Stack | Achievement |
| :--- | :--- | :--- | :--- |
| **[Minecraft Object Detection](./minecraft_fcos_yolo)** | Object Detection | MMDetection, FCOS, YOLOv8, Roboflow | YOLOv8 delivers ~80% higher FPS, while FCOS keeps Precision +0.21 above YOLO across 17 classes. |
| **[Semantic Segmentation](./segmentation_work)** | Semantic Segmentation | MMSegmentation, SegFormer-B2, DeepLabV3+, ClearML, SAM2 | SegFormer-B2 beats DeepLabV3+ by **+2.55% mDice** and **+4.44% Dog Dice**, hitting **90.39% mDice** on the test set. |
| **[Stable Diffusion LoRA](./diffusion_finetuning)** | Generative / Fine-tuning | Diffusers, PEFT (LoRA), Stable Diffusion 1.5, TensorBoard | LoRA fine-tuning of SD 1.5 on **3 images** embeds custom character; best setup: 2000 steps, `guidance_scale=5.0`. |

---

## 🛠️ Skills & Methodologies

Through these projects, I have demonstrated proficiency in:
- **Architecture Exploration**: Implementing and fine-tuning CNNs (ResNet, DeepLab), Transformers (SegFormer), and diffusion models (Stable Diffusion).
- **Data Engineering**: Enhancing dataset quality using **SAM2 (Segment Anything Model)** and **YOLOv8** for automated re-annotation; minimal-data personalization (3-image LoRA).
- **Experiment Tracking**: Managing multi-stage hypotheses with **ClearML** and **TensorBoard**.
- **Model Benchmarking**: Evaluating performance through mDice, mIoU, mAcc, and inference speed (FPS).

---

## 📂 Project Deep Dives

### [1. Minecraft Mob Detection (FCOS vs YOLO)](./minecraft_fcos_yolo)
- **Goal**: Benchmark anchor-free FCOS vs anchor-based YOLOv8 on a 17-class Minecraft dataset.
- **Highlights**: YOLOv8 runs at ~90 FPS for live streams, while FCOS sustains Precision 0.87 (+31.9% vs YOLO) for analytics pipelines.
- **Artifacts**: Comparative metrics, video inference demos, unified inference scripts.
- [Go to project →](./minecraft_fcos_yolo)

### [2. Semantic Segmentation (Cats & Dogs)](./segmentation_work)
- **Goal**: Contrast CNN (DeepLabV3+) and transformer (SegFormer-B2) backbones for 3-class animal segmentation.
- **Highlights**: Data quality uplift via YOLOv8 + SAM2, SegFormer-B2 reaches 93.13% mDice (val) and 90.39% mDice (test), best-in-class Dog Dice +4.44%.
- **Artifacts**: Training configs, ClearML experiment logs, re-annotation toolkit.
- [Go to project →](./segmentation_work)

### [3. Stable Diffusion LoRA Fine-tuning (Cheburashka)](./diffusion_finetuning)
- **Goal**: Personalize SD 1.5 with LoRA on 3 reference images for custom character generation.
- **Highlights**: Exp 1 (2000 steps) gives best visual quality; `guidance_scale=5.0` reduces artifacts vs default 7.5; Min-SNR weighting critical.
- **Artifacts**: LoRA checkpoints, TensorBoard logs, guidance_scale comparison.
- [Go to project →](./diffusion_finetuning)

---

## 📬 Contact & Links

- **GitHub**: [Mi-Ilia](https://github.com/Mi-Ilia)
- **Email**: [mikhaylov.is@yandex.ru](mailto:mikhaylov.is@yandex.ru)

---
<div align="center">

**Computer Vision Portfolio Project**  *Last updated: February 2026*

</div>
