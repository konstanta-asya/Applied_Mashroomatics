# Applied Mashroomatics

A deep learning project for mushroom species classification using the Danish Fungi 2020 (DF20M) dataset.

## Overview

This project implements a PyTorch-based image classification pipeline for identifying mushroom species from photographs. It supports both single-image classification and multi-view classification where multiple photos of the same observation are used together.

## Dataset

The project uses the **DF20M dataset** (Danish Fungi 2020 Mini) from GBIF, containing:
- ~32,000 mushroom images
- 180 species classes
- Rich metadata including geographic coordinates, substrate type, habitat, and observation date

## Project Structure

```
Applied_Mashroomatics/
├── data/
│   └── raw/
│       └── DF20M/              # Image files
├── notebooks/
│   ├── 01_data_check.ipynb     # Data exploration and validation
│   └── AM_EDA.ipynb            # Exploratory data analysis
├── src/
│   └── data/
│       ├── mushroom_dataset.py # Custom PyTorch Dataset
│       ├── transforms.py       # Image augmentation transforms
│       └── data_setup.py       # DataLoader creation utilities
└── requirements.txt
```

## Features

- **Single-image mode**: Standard image classification
- **Group mode**: Multi-view classification using multiple images per observation
- **Data augmentation**: Random crops, flips, rotations with ImageNet normalization
- **Metadata support**: Leverages geographic, temporal, and substrate information

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Requirements

- Python 3.11+
- PyTorch
- torchvision
- timm
- pandas
- scikit-learn
- Pillow
- matplotlib
- seaborn
- jupyter

## Usage

```python
from src.data.data_setup import create_dataloaders

train_loader, val_loader, num_classes = create_dataloaders(
    csv_path='data/raw/DF20M-metadata/DF20M-train_metadata_PROD.csv',
    root_dir='data/raw/DF20M/',
    batch_size=32,
    mode='single'  # or 'group' for multi-view
)
```

## License

This project uses data from GBIF (Global Biodiversity Information Facility).