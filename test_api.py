#!/usr/bin/env python
"""Test script for the Mushroom Classification API."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")

    from src.api.config import settings
    print(f"  Config loaded: {settings.api_title}")

    from src.api.schemas import PredictionResponse, SpeciesPrediction
    print("  Schemas loaded")

    from src.api.inference import MushroomClassifier, get_classifier
    print("  Inference module loaded")

    from src.api.main import app
    print("  FastAPI app loaded")

    print("All imports OK!")
    return True


def test_classifier():
    """Test classifier initialization and model loading."""
    print("\nTesting classifier...")

    from src.api.inference import MushroomClassifier
    from src.api.config import settings

    classifier = MushroomClassifier()
    print(f"  Classifier created")

    model_path = settings.model_path
    print(f"  Model path: {model_path}")
    print(f"  Model exists: {model_path.exists()}")

    if model_path.exists():
        print("  Loading model...")
        classifier.load_model(model_path)
        print(f"  Model loaded!")
        print(f"  Device: {classifier.device}")
        print(f"  Num classes: {classifier.num_classes}")
        print(f"  Sample species: {list(classifier.id_to_species.items())[:5]}")
        return True
    else:
        print("  Model file not found, skipping load test")
        return False


def test_prediction():
    """Test prediction with a sample image."""
    print("\nTesting prediction...")

    from src.api.inference import get_classifier
    from PIL import Image
    import numpy as np

    classifier = get_classifier()

    if not classifier.is_loaded:
        print("  Model not loaded, loading now...")
        classifier.load_model()

    # Create a dummy image for testing
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    result = classifier.predict(dummy_image, top_k=3)
    print(f"  Prediction successful!")
    print(f"  Top prediction: {result.top_prediction.species_name}")
    print(f"  Confidence: {result.top_prediction.confidence:.4f}")
    print(f"  Is confident: {result.is_confident}")

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Mushroom API Test Suite")
    print("=" * 50)

    try:
        test_imports()
        test_classifier()
        test_prediction()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)