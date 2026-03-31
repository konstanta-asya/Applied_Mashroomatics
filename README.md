# Applied Mashroomatics

A deep learning project for mushroom species classification using the Danish Fungi 2020 (DF20M) dataset.

## Overview

This project implements a PyTorch-based image classification system for identifying mushroom species from photographs. It includes multiple model architectures (CNN and ViT), a Streamlit web application, and training notebooks designed for Google Colab.

## Dataset

The project uses the **DF20M dataset** (Danish Fungi 2020 Mini) from GBIF, containing:
- ~32,000 mushroom images
- 180 species classes
- Rich metadata including geographic coordinates, substrate type, habitat, and observation date

## Project Structure

```
Applied_Mashroomatics/
├── app/
│   ├── streamlit_app.py        # Main Streamlit application
│   ├── pages/                  # App pages (Classifier, About, Species, Team)
│   ├── species_info.json       # Species metadata
│   └── species_mapping.json    # Class ID to species mapping
├── notebooks/
│   ├── 01_data_check.ipynb     # Data exploration and validation
│   ├── AM_EDA.ipynb            # Exploratory data analysis
│   ├── train_cnn.ipynb         # CNN training (EfficientNet-B0)
│   └── train_vit_colab.ipynb   # ViT training
├── src/
│   ├── models/
│   │   ├── vit.py              # Vision Transformer model
│   │   ├── mushroom_vit.py     # ViT with metadata encoder
│   │   └── metadata_encoder.py # Metadata processing
│   ├── data/
│   │   ├── mushroom_dataset.py # Custom PyTorch Dataset
│   │   ├── transforms.py       # Image augmentation transforms
│   │   └── data_setup.py       # DataLoader creation utilities
│   ├── api/                    # FastAPI backend
│   └── train_vit.py            # ViT training script
├── run_app.py                  # Launch Streamlit app
├── run_api.py                  # Launch API server
└── requirements.txt
```

## Models

- **CNN**: EfficientNet-B0 pretrained on ImageNet, fine-tuned for mushroom classification (~73% accuracy)
- **ViT**: Vision Transformer with optional metadata encoding

## Features

- **Web Application**: Streamlit app to upload mushroom photos and compare CNN/ViT predictions
- **Multi-model comparison**: Side-by-side results from different architectures
- **Species information**: Detailed info about identified mushroom species
- **Data augmentation**: Random crops, flips, rotations with ImageNet normalization
- **Colab support**: Training notebooks designed for Google Colab with GPU

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the App

```bash
python run_app.py
# or
streamlit run app/streamlit_app.py
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- torchvision
- timm
- streamlit
- pandas
- pillow
- gdown
- scikit-learn
- matplotlib
- seaborn

## License

This project uses data from GBIF (Global Biodiversity Information Facility).