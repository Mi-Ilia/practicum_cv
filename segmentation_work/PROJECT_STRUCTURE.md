## Project Layout

```
segmentation_work/
│
├── notebook.ipynb                 # Main notebook with the full pipeline
├── README.md                      # Primary documentation
├── requirements.txt               # Python dependencies
│
├── configs/                       # Model configurations
│   ├── h1_deeplabv3plus_r50_d8_256x256.py
│   ├── h2_deeplabv3plus_r50_d8_256x256.py
│   ├── exp1_deeplabv3plus_r50_d8_256x256.py
│   └── exp2_segformer_b2_256x256.py
│
├── src/
│   ├── mmseg_custom/losses/combo_loss.py
│   ├── tools/                     # train.py, test.py
│   └── utils/                     # visualization_utils.py, test_utils.py, clearml_utils.py
│
├── datasets/
│   └── train_dataset_for_students/
│       ├── images/                # 440 images (train/val/test)
│       ├── labels/                # Original masks
│       └── labels_sam2/           # Improved masks after re-annotation
│
├── experiments/
│   ├── h1_experiment/
│   ├── h2_experiment/
│   ├── exp1_experiment/
│   └── exp2_experiment/
│
├── supplementary/viz/             # EDA charts and predictions
├── reports/practicum_report.md    # Detailed academic report
└── mmsegmentation/                # MMSegmentation framework (submodule)
```
