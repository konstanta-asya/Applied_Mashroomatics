from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API Configuration"""

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    model_path: Path = project_root / "src" / "models" / "best_model.pth"
    species_info_path: Path = project_root / "data" / "species_info.json"
    species_metadata_path: Path = project_root / "data" / "species_metadata.json"

    # Metadata boost
    metadata_boost: float = 0.15  # How much to boost matching species

    # Model settings
    model_name: str = "vit_large_patch16_224"
    img_size: int = 224
    device: str = "cuda"  # "cuda" or "cpu"

    # Inference settings
    confidence_threshold: float = 0.5
    top_k: int = 5
    use_tta: bool = False
    tta_augments: int = 5

    # API settings
    api_title: str = "Mushroom Classifier API"
    api_version: str = "1.0.0"
    max_image_size: int = 10 * 1024 * 1024  # 10 MB

    class Config:
        env_prefix = "MUSHROOM_"


settings = Settings()