# Chest Cancer Detection Using Deep Learning

This repository contains a student project on lung cancer image classification using deep learning.

The main work is documented in the notebook `ChestCancer/utforskeData.ipynb`, where we explore the dataset, build baseline models, test CNN-based approaches, and evaluate a transfer learning model.

## Project Contents

- `ChestCancer/utforskeData.ipynb`
  - Main notebook with data exploration, model training, and evaluation
- `ChestCancer/app.py`
  - Simple Gradio demo for image classification
- `ChestCancer/model.keras`
  - Saved model exported from the notebook for the demo
- `ChestCancer/Data/`
  - Dataset folders for training, validation, and testing

## Classes

The model predicts four classes:

- Adenocarcinoma
- Large cell carcinoma
- Normal
- Squamous cell carcinoma

## How To Run

1. Install the required packages:

```bash
pip install -r ChestCancer/requirements.txt
```

2. Run the demo app:

```bash
cd ChestCancer
python app.py
```

## Notes

- This is a student project and not a medical tool.
- The notebook is the best place to understand the full workflow and thought process.
- In the transfer learning section, the frozen MobileNetV2-based model is the intended final model because it gives the best validation result.
