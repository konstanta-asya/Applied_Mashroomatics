import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np


class MushroomDataset(Dataset):
    def __init__(self, metadata_file, root_dir, transform=None, mode='single',
                 images_per_group=3, species_to_id=None,
                 habitat_vocab=None, substrate_vocab=None):
        self.annotations = pd.read_csv(metadata_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.images_per_group = images_per_group

        # Metadata vocabularies (UNK index = len(vocab))
        self.habitat_vocab = habitat_vocab or {}
        self.substrate_vocab = substrate_vocab or {}

        # Handle column name variations
        self.habitat_col = 'Habitat' if 'Habitat' in self.annotations.columns else 'habitat'
        self.substrate_col = 'Substrate' if 'Substrate' in self.annotations.columns else 'substrate'

        if 'class_id' not in self.annotations.columns:
            if species_to_id is not None:
                # Use provided mapping for consistent labels across train/val
                self.species_to_id = species_to_id
            else:
                # Create new mapping (only use when processing full dataset)
                unique_species = sorted(self.annotations['species'].unique())
                self.species_to_id = {sp: idx for idx, sp in enumerate(unique_species)}
            self.annotations['class_id'] = self.annotations['species'].map(self.species_to_id)
            self.num_classes = len(self.species_to_id)
        else:
            self.num_classes = self.annotations['class_id'].nunique()

        if self.mode == 'group':
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

        label = torch.tensor(int(row['class_id']), dtype=torch.long)

        # Encode metadata with UNK fallback (UNK index = len(vocab))
        habitat_val = row.get(self.habitat_col, None)
        habitat_id = self.habitat_vocab.get(habitat_val, len(self.habitat_vocab)) if pd.notna(habitat_val) else len(self.habitat_vocab)

        substrate_val = row.get(self.substrate_col, None)
        substrate_id = self.substrate_vocab.get(substrate_val, len(self.substrate_vocab)) if pd.notna(substrate_val) else len(self.substrate_vocab)

        # Month: default to 6 (June) if missing (month 0 is invalid for sinusoidal encoding)
        month_val = row.get('month', None)
        month = int(month_val) if pd.notna(month_val) and month_val > 0 else 6

        lat = float(row['Latitude']) if not pd.isna(row.get('Latitude')) else 0.0
        lon = float(row['Longitude']) if not pd.isna(row.get('Longitude')) else 0.0

        return {
            "image": image,
            "label": label,
            "habitat_id": torch.tensor(habitat_id, dtype=torch.long),
            "substrate_id": torch.tensor(substrate_id, dtype=torch.long),
            "month": torch.tensor(month, dtype=torch.long),
            "meta": {
                "gbif_id": row['gbifID'],
                "month": month,
                "substrate": str(substrate_val) if pd.notna(substrate_val) else "UNK",
                "habitat": str(habitat_val) if pd.notna(habitat_val) else "UNK",
                "geo": torch.tensor([lat, lon], dtype=torch.float32)
            }
        }

    def _get_group_item(self, index):
        gbif_id = self.group_ids[index]
        group_df = self.groups.get_group(gbif_id)

        image_paths = group_df['image_path'].tolist()

        if len(image_paths) >= self.images_per_group:
            selected_paths = np.random.choice(image_paths, self.images_per_group, replace=False)
        else:
            selected_paths = np.random.choice(image_paths, self.images_per_group, replace=True)

        images = []
        for path in selected_paths:
            img_path = os.path.join(self.root_dir, path)
            images.append(self._load_image(img_path))

        images = torch.stack(images)

        first_row = group_df.iloc[0]
        label = torch.tensor(int(first_row['class_id']), dtype=torch.long)

        # Encode metadata with UNK fallback
        habitat_val = first_row.get(self.habitat_col, None)
        habitat_id = self.habitat_vocab.get(habitat_val, len(self.habitat_vocab)) if pd.notna(habitat_val) else len(self.habitat_vocab)

        substrate_val = first_row.get(self.substrate_col, None)
        substrate_id = self.substrate_vocab.get(substrate_val, len(self.substrate_vocab)) if pd.notna(substrate_val) else len(self.substrate_vocab)

        month_val = first_row.get('month', None)
        month = int(month_val) if pd.notna(month_val) and month_val > 0 else 6

        lat = float(first_row['Latitude']) if not pd.isna(first_row.get('Latitude')) else 0.0
        lon = float(first_row['Longitude']) if not pd.isna(first_row.get('Longitude')) else 0.0

        return {
            "images": images,
            "label": label,
            "habitat_id": torch.tensor(habitat_id, dtype=torch.long),
            "substrate_id": torch.tensor(substrate_id, dtype=torch.long),
            "month": torch.tensor(month, dtype=torch.long),
            "meta": {
                "gbif_id": gbif_id,
                "month": month,
                "substrate": str(substrate_val) if pd.notna(substrate_val) else "UNK",
                "habitat": str(habitat_val) if pd.notna(habitat_val) else "UNK",
                "geo": torch.tensor([lat, lon], dtype=torch.float32)
            }
        }