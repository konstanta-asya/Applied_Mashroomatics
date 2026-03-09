import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .mushroom_dataset import MushroomDataset
from .transforms import get_transforms


def create_dataloaders(
        csv_path,
        root_dir,
        batch_size=32,
        mode='single',
        split_ratio=0.2,
        num_workers=4,
        augmentation_strength='standard'
):

    df = pd.read_csv(csv_path)

    # Remove rows with missing species
    df = df.dropna(subset=['species']).reset_index(drop=True)

    # Create consistent label mapping from FULL dataset before splitting
    unique_species = sorted(df['species'].unique())
    species_to_id = {sp: idx for idx, sp in enumerate(unique_species)}

    unique_ids = df['gbifID'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=split_ratio, random_state=42)

    train_df = df[df['gbifID'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['gbifID'].isin(val_ids)].reset_index(drop=True)


    train_csv = 'temp_train.csv'
    val_csv = 'temp_val.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_ds = MushroomDataset(
        metadata_file=train_csv,
        root_dir=root_dir,
        transform=get_transforms('train', augmentation_strength=augmentation_strength),
        mode=mode,
        species_to_id=species_to_id
    )

    val_ds = MushroomDataset(
        metadata_file=val_csv,
        root_dir=root_dir,
        transform=get_transforms('val'),
        mode=mode,
        species_to_id=species_to_id
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    os.remove(train_csv)
    os.remove(val_csv)

    return train_loader, val_loader, train_ds.num_classes