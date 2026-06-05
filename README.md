1.🛠 Environment Setup

This project provides a minimal, inference-specific dependency list (`requirements.txt`), removing irrelevant packages to avoid version conflicts.
1.Checkpoints & Weights Download
Please download the required weight files in advance and strictly place them according to the directory structure below.

2.📥 Download Links
Model Checkpoints Download Link: 👉 Click [here](https://drive.google.com/drive/folders/1w6NUxo5z_A_HVzhu8xSft7bcniHzgAw2?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto) to download UniMed-ICL Checkpoints (Please replace this with your actual download URL)

BiomedBERT Files Download Link: 👉 Click [here](https://drive.google.com/drive/folders/1w6NUxo5z_A_HVzhu8xSft7bcniHzgAw2?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto) to download BiomedBERT Files (Please replace this with your actual download URL)

📂 Directory Structure Configuration
Downloaded files must be placed in the respective subdirectories under the UniMed_ICL folder. The final directory layout should look exactly as follows:
```text
UniMed-ICL/
├── run_inference.sh
├── eval_script/
│   ├── eval_ICL.py          # 3D ICL evaluation script
│   ├── eval_ICL_2D.py       # 2D ICL evaluation script
│   ├── eval_text.py         # Text-guided / Language-driven evaluation script
│   ├── eval_interactive.py  # Interactive evaluation scrips
│   ├── config.py 
│   └── dataloader.py        
├── UniMed_ICL/
│   ├── checkpoints/         # 📌 CRITICAL: Create this checkpoints folder
│   │   └── [Place the downloaded UniMed-ICL model checkpoints (e.g., .ckpt or .pt) here]
│   ├── weights/             # 📌 CRITICAL: Create this weights folder
│   │   └── [Place the downloaded BiomedBERT pretrained files/weights here]
│   ├── models/
│   └── tool/
├── Brain/                   # Brain dataset directory for Text-guided
│   └── dataset.json
└── Liver/                   # Liver dataset directory
    └── dataset.json
```

3.Dataset Preparation
The repository includes built-in test entry points for Brain and Liver modalities. Please ensure that:

The dataset path aligns perfectly with the configurations defined in dataset.json.

No unified spatial/voxel resampling is applied during preprocessing (No spatial resampling is applied).

4.Running Inference Scripts
You can trigger the entire evaluation workflow using the one-click shell script located in the root directory.
Run via One-Click Script:
```text
./run_inference.sh
```
