## Project Layout

```
📁 minecraft_fcos_yolo/
│
├── 📓 notebook.ipynb                 # Main notebook with full pipeline
├── 📄 conclusion.md                  # Detailed conclusions and recommendations
├── 📄 README.md                      # Primary documentation
├── 📋 requirements.txt               # Python dependencies
├── 📖 mmdetection_setup.md           # MMDetection installation guide
│
├── 📂 configs/                       # Model configurations
│   └── fcos/
│       └── fcos_minecraft.py         # FCOS config for MMDetection
│
├── 📂 datasets/                      # Datasets (not included in repository)
│   ├── minecraft/                    # Pascal VOC + COCO format
│   │   ├── train/ (2307 images)
│   │   ├── valid/ (422 images)
│   │   ├── test/ (155 images)
│   │   └── annotations/
│   │       ├── instances_train.json
│   │       ├── instances_val.json
│   │       └── test_annotations.voc.json
│   └── minecraft_yolo/               # YOLO format
│       ├── train/ (2307 jpg + 2307 txt)
│       ├── valid/ (422 jpg + 422 txt)
│       ├── test/ (155 jpg + 155 txt)
│       ├── dataset.yaml
│       └── classes.txt
│
├── 📂 artifacts/                     # Experiment outputs
│   ├── fcos/
│   │   └── minecraft_fcos_r50_v1/
│   │       ├── best_*.pth            # Best checkpoint
│   │       ├── epoch_*.pth           # Per-epoch checkpoints
│   │       └── logs/                 # TensorBoard logs
│   ├── yolo/
│   │   └── minecraft_yolo/
│   │       ├── weights/
│   │       │   ├── best.pt           # Best model
│   │       │   └── last.pt           # Last model
│   │       ├── results.csv           # Training metrics
│   │       └── *.png                 # Plots (F1, PR, confusion matrix)
│   ├── inference/
│   │   ├── fcos/vis/ (155 jpg)       # FCOS visualizations
│   │   └── yolo/ (155 jpg)           # YOLO visualizations
│   ├── metrics/
│   │   └── metrics_comparison.csv    # Summary table
│   └── videos/
│       ├── fcos_inference.mp4        # FCOS video inference
│       └── yolo_inference.mp4        # YOLO video inference
│
└── 📂 mmdetection/                   # MMDetection framework (submodule)
```
