import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np


class MushroomDataset(Dataset):
    def __init__(self, metadata_file, root_dir, transform=None, mode='single', images_per_group=3):
        self.annotations = pd.read_csv(metadata_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.images_per_group = images_per_group

        # Create class_id if it doesn't exist
        if 'class_id' not in self.annotations.columns:
            # Use species column to create class labels
            unique_species = self.annotations['species'].unique()
            self.species_to_id = {sp: idx for idx, sp in enumerate(unique_species)}
            self.annotations['class_id'] = self.annotations['species'].map(self.species_to_id)
            self.num_classes = len(unique_species)
        else:
            self.num_classes = self.annotations['class_id'].nunique()

        if self.mode == 'group':
            # Group by gbifID for group mode
            self.groups = self.annotations.groupby('gbifID')
            self.group_ids = list(self.groups.groups.keys())

    def __len__(self):
        if self.mode == 'group':
            return len(self.group_ids)
        return len(self.annotations)

    def _load_image(self, img_path):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        if self.mode == 'group':
            return self._get_group_item(index)
        return self._get_single_item(index)

    def _get_single_item(self, index):
        row = self.annotations.iloc[index]
        img_path = os.path.join(self.root_dir, row['image_path'])
        image = self._load_image(img_path)

        label = torch.tensor(int(row['class_id']))

        lat = float(row['Latitude']) if not pd.isna(row['Latitude']) else 0.0
        lon = float(row['Longitude']) if not pd.isna(row['Longitude']) else 0.0

        return {
            "image": image,
            "label": label,
            "meta": {
                "gbif_id": row['gbifID'],
                "month": row['month'],
                "substrate": str(row['Substrate']),
                "geo": torch.tensor([lat, lon], dtype=torch.float32)
            }
        }

    def _get_group_item(self, index):
        gbif_id = self.group_ids[index]
        group_df = self.groups.get_group(gbif_id)

        # Get image paths for this group
        image_paths = group_df['image_path'].tolist()

        # Sample or pad to get exactly images_per_group images
        if len(image_paths) >= self.images_per_group:
            # Random sample without replacement
            selected_paths = np.random.choice(image_paths, self.images_per_group, replace=False)
        else:
            # Sample with replacement if not enough images
            selected_paths = np.random.choice(image_paths, self.images_per_group, replace=True)

        # Load images
        images = []
        for path in selected_paths:
            img_path = os.path.join(self.root_dir, path)
            images.append(self._load_image(img_path))

        # Stack images: [images_per_group, C, H, W]
        images = torch.stack(images)

        # Get label and metadata from first row
        first_row = group_df.iloc[0]
        label = torch.tensor(int(first_row['class_id']))

        lat = float(first_row['Latitude']) if not pd.isna(first_row['Latitude']) else 0.0
        lon = float(first_row['Longitude']) if not pd.isna(first_row['Longitude']) else 0.0

        return {
            "images": images,
            "label": label,
            "meta": {
                "gbif_id": gbif_id,
                "month": first_row['month'],
                "substrate": str(first_row['Substrate']),
                "geo": torch.tensor([lat, lon], dtype=torch.float32)
            }
        }