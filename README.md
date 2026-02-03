# Computer Vision Portfolio: Yandex Practicum PRO

Welcome to my Computer Vision portfolio. This repository centralizes research projects developed during the **Computer Vision PRO** course. The focus is on implementing modern deep learning architectures, dataset optimization, and comparative analysis for real-world tasks.

---

## 🚀 Featured Projects

| Project | Domain | Key Tech Stack | Achievement |
| :--- | :--- | :--- | :--- |
| **[Minecraft Object Detection](./minecraft_fcos_yolo)** | Object Detection | Python 3.10, PyTorch 2.1.2, MMDetection 3.3.0, YOLOv8 | YOLOv8 delivers ~80% higher FPS, while FCOS keeps Precision +0.21 above YOLO across 17 classes. |
| **[Semantic Segmentation](./segmentation_work)** | Semantic Segmentation | Python 3.10, PyTorch 2.1.2, MMSegmentation 1.2.2, SegFormer-B2 | SegFormer-B2 beats DeepLabV3+ by **+2.55% mDice** and **+4.44% Dog Dice**, hitting **90.39% mDice** on the test set. |

---

## 🛠️ Skills & Methodologies

Through these projects, I have demonstrated proficiency in:
- **Architecture Exploration**: Implementing and fine-tuning CNNs (ResNet, DeepLab) and Transformers (SegFormer).
- **Data Engineering**: Enhancing dataset quality using **SAM2 (Segment Anything Model)** and **YOLOv8** for automated re-annotation.
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

---

## 📬 Contact & Links

- **GitHub**: [Mi-Ilia](https://github.com/Mi-Ilia)
- **Email**: [mikhaylov.is@yandex.ru](mailto:mikhaylov.is@yandex.ru)

---
<div align="center">

**Computer Vision Portfolio Project**  *Last updated: February 2026*

</div>
