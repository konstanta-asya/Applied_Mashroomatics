"""
Generate species to sample image mapping
"""

import pandas as pd
import json
from pathlib import Path

def main():
    # Load metadata
    metadata_path = Path(__file__).parent.parent / "data" / "raw" / "DF20M-metadata" / "DF20M-train_metadata_PROD.csv"

    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} records")

    # Get one sample image per species
    species_images = {}

    for species in df['species'].unique():
        if pd.isna(species):
            continue

        # Get first image for this species
        sample = df[df['species'] == species].iloc[0]
        image_path = sample['image_path']

        species_images[species] = {
            "image_path": image_path,
            "month": int(sample['month']) if pd.notna(sample['month']) else None,
            "habitat": sample['Habitat'] if pd.notna(sample.get('Habitat')) else None,
            "substrate": sample['Substrate'] if pd.notna(sample.get('Substrate')) else None
        }

    # Save mapping
    output_path = Path(__file__).parent.parent / "data" / "species_images.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(species_images, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(species_images)} species to {output_path}")


if __name__ == "__main__":
    main()