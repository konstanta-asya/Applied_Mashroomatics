import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .config import settings
from .schemas import PredictionResponse, HealthResponse, ErrorResponse, MetadataInput
from .inference import get_classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    classifier = get_classifier()
    if settings.model_path.exists():
        classifier.load_model(settings.model_path)
        print(f"Model loaded: {settings.model_name}")
        print(f"Device: {classifier.device}")
        print(f"Classes: {classifier.num_classes}")
    else:
        print(f"WARNING: Model not found at {settings.model_path}")
    yield


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    classifier = get_classifier()
    return HealthResponse(
        status="healthy",
        model_loaded=classifier.is_loaded,
        device=str(classifier.device) if classifier.device else "not loaded",
        num_classes=classifier.num_classes
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = 5,
    use_tta: bool = False,
    month: int = None,
    habitat: str = None,
    substrate: str = None
):
    """
    Classify mushroom from uploaded image.

    - **file**: Image file (JPEG, PNG)
    - **top_k**: Number of top predictions to return (default: 5)
    - **use_tta**: Use test-time augmentation for better accuracy (slower)
    - **month**: Month of observation (1-12) for seasonal filtering
    - **habitat**: Habitat type (e.g., "coniferous", "deciduous")
    - **substrate**: Substrate type (e.g., "soil", "dead wood")
    """
    classifier = get_classifier()

    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image."
        )

    # Read and validate image
    try:
        contents = await file.read()

        if len(contents) > settings.max_image_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Max size: {settings.max_image_size // (1024*1024)} MB"
            )

        image = Image.open(io.BytesIO(contents))

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {str(e)}"
        )

    # Build metadata if provided
    metadata = None
    if month or habitat or substrate:
        metadata = MetadataInput(month=month, habitat=habitat, substrate=substrate)

    # Run prediction
    try:
        result = classifier.predict(
            image,
            top_k=top_k,
            use_tta=use_tta,
            metadata=metadata
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/url", response_model=PredictionResponse)
async def predict_from_url(
    image_url: str,
    top_k: int = 5,
    use_tta: bool = False
):
    """
    Classify mushroom from image URL.

    - **image_url**: URL to mushroom image
    - **top_k**: Number of top predictions to return
    - **use_tta**: Use test-time augmentation
    """
    import httpx

    classifier = get_classifier()

    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=10.0)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not fetch image from URL: {str(e)}"
        )

    result = classifier.predict(image, top_k=top_k, use_tta=use_tta)
    return result


@app.get("/species", response_model=dict)
async def list_species():
    """List all supported mushroom species."""
    classifier = get_classifier()

    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "num_classes": classifier.num_classes,
        "species": classifier.id_to_species
    }


@app.get("/species/{species_id}")
async def get_species_info(species_id: int):
    """Get detailed information about a specific species."""
    classifier = get_classifier()

    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if species_id not in classifier.id_to_species:
        raise HTTPException(status_code=404, detail="Species not found")

    species_name = classifier.id_to_species[species_id]
    info = classifier.species_info.get(species_name, {})

    return {
        "species_id": species_id,
        "species_name": species_name,
        **info
    }


@app.post("/attention")
async def get_attention_map(file: UploadFile = File(...)):
    """
    Generate attention rollout heatmap for the uploaded image.

    Returns base64-encoded PNG image showing where the model focuses.
    """
    classifier = get_classifier()

    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    attention_map = classifier.get_attention_map(image)

    if attention_map is None:
        raise HTTPException(
            status_code=501,
            detail="Attention maps not supported for this model architecture"
        )

    return {"attention_map": attention_map}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
