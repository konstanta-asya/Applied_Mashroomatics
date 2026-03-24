import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional
import json
import timm
import io
import base64

from .config import settings
from .schemas import SpeciesPrediction, PredictionResponse, EdibilityStatus, MetadataInput

# Import transforms (handle both relative and absolute imports)
try:
    from src.data.transforms import get_transforms, get_tta_transforms
except ImportError:
    from ..data.transforms import get_transforms, get_tta_transforms


class MushroomClassifier:
    """
    Mushroom classification inference engine.

    Usage:
        classifier = MushroomClassifier()
        classifier.load_model("path/to/checkpoint.pth")
        result = classifier.predict(image)
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.species_to_id: dict = {}
        self.id_to_species: dict = {}
        self.num_classes: int = 0
        self.species_info: dict = {}
        self.species_metadata: dict = {}  # For metadata-based boosting
        self.transform = None
        self.tta_transforms = None
        self._is_loaded = False

    def load_model(self, checkpoint_path: Union[str, Path] = None) -> None:
        """Load model from checkpoint."""
        checkpoint_path = checkpoint_path or settings.model_path

        # Set device
        if settings.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Extract metadata
        self.species_to_id = checkpoint.get("species_to_id", {})
        self.id_to_species = {v: k for k, v in self.species_to_id.items()}
        self.num_classes = checkpoint.get("num_classes", len(self.species_to_id))

        # Get model config from checkpoint or use defaults
        config = checkpoint.get("config", {})
        model_name = config.get("model_name", settings.model_name)
        img_size = config.get("img_size", settings.img_size)

        # Determine which weights to load
        if "ema_state_dict" in checkpoint:
            state_dict = checkpoint["ema_state_dict"]
        else:
            state_dict = checkpoint["model_state_dict"]

        # Detect model type from state_dict keys
        # torchvision ViT uses: class_token, conv_proj, encoder.layers
        # timm ViT uses: cls_token, patch_embed, blocks
        all_keys = list(state_dict.keys())
        is_torchvision = any("encoder.layers" in k or k == "class_token" for k in all_keys)

        if is_torchvision:
            # Use torchvision ViT
            from torchvision.models import vit_b_16, vit_l_16
            import torch.nn as nn

            # Detect model size from hidden dimension
            head_weight = state_dict.get("heads.head.weight")
            if head_weight is not None:
                hidden_dim = head_weight.shape[1]
            else:
                hidden_dim = 768  # default to base

            # ViT-B: 768, ViT-L: 1024
            if hidden_dim >= 1024:
                self.model = vit_l_16(weights=None)
            else:
                self.model = vit_b_16(weights=None)

            # Replace classification head
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, self.num_classes)
        else:
            # Use timm model
            self.model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=self.num_classes
            )

        # Load weights
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Setup transforms
        self.transform = get_transforms(mode="val", img_size=img_size)
        if settings.use_tta:
            self.tta_transforms = get_tta_transforms(
                img_size=img_size,
                n_augments=settings.tta_augments
            )

        # Load species info if available
        self._load_species_info()

        self._is_loaded = True

    def _load_species_info(self) -> None:
        """Load additional species information (edibility, common names, etc.)"""
        if settings.species_info_path.exists():
            with open(settings.species_info_path, "r", encoding="utf-8") as f:
                self.species_info = json.load(f)

        # Load metadata for boosting
        if settings.species_metadata_path.exists():
            with open(settings.species_metadata_path, "r", encoding="utf-8") as f:
                self.species_metadata = json.load(f)

    def _calculate_metadata_boost(
        self,
        species_name: str,
        metadata: MetadataInput
    ) -> float:
        """Calculate boost factor based on metadata match."""
        if metadata is None:
            return 0.0

        sp_meta = self.species_metadata.get(species_name, {})
        if not sp_meta:
            return 0.0

        boost = 0.0
        matches = 0

        # Month boost
        if metadata.month and sp_meta.get("months"):
            month_counts = sp_meta["months"]
            total = sum(month_counts.values())
            month_key = str(metadata.month)
            if month_key in month_counts:
                # Higher boost for months where this species is common
                month_ratio = month_counts[month_key] / total
                boost += month_ratio * settings.metadata_boost
                matches += 1

        # Habitat boost
        if metadata.habitat and sp_meta.get("habitats"):
            habitat_counts = sp_meta["habitats"]
            total = sum(habitat_counts.values())
            # Check for partial match
            for hab, count in habitat_counts.items():
                if metadata.habitat.lower() in hab.lower():
                    habitat_ratio = count / total
                    boost += habitat_ratio * settings.metadata_boost
                    matches += 1
                    break

        # Substrate boost
        if metadata.substrate and sp_meta.get("substrates"):
            substrate_counts = sp_meta["substrates"]
            total = sum(substrate_counts.values())
            for sub, count in substrate_counts.items():
                if metadata.substrate.lower() in sub.lower():
                    substrate_ratio = count / total
                    boost += substrate_ratio * settings.metadata_boost
                    matches += 1
                    break

        return boost

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def preprocess_tta(self, image: Image.Image) -> torch.Tensor:
        """Preprocess with TTA (multiple augmented versions)."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensors = []
        for transform in self.tta_transforms:
            tensors.append(transform(image))

        return torch.stack(tensors).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, str, Path],
        top_k: int = None,
        use_tta: bool = None,
        metadata: MetadataInput = None
    ) -> PredictionResponse:
        """
        Predict mushroom species from image.

        Args:
            image: PIL Image, or path to image
            top_k: Number of top predictions to return
            use_tta: Whether to use test-time augmentation
            metadata: Optional metadata (month, habitat, substrate) for boosting

        Returns:
            PredictionResponse with predictions
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        top_k = top_k or settings.top_k
        use_tta = use_tta if use_tta is not None else settings.use_tta

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Inference
        if use_tta and self.tta_transforms:
            batch = self.preprocess_tta(image)
            logits = self.model(batch)
            # Average predictions across augmentations
            probs = F.softmax(logits, dim=1).mean(dim=0)
        else:
            batch = self.preprocess(image)
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1).squeeze(0)

        # Apply metadata boost if provided
        if metadata is not None and self.species_metadata:
            probs_list = probs.tolist()
            for idx in range(len(probs_list)):
                species_name = self.id_to_species.get(idx, "")
                boost = self._calculate_metadata_boost(species_name, metadata)
                probs_list[idx] = min(1.0, probs_list[idx] + boost)
            probs = torch.tensor(probs_list, device=self.device)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))

        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            species_name = self.id_to_species.get(idx, f"Unknown_{idx}")
            species_data = self.species_info.get(species_name, {})

            pred = SpeciesPrediction(
                species_id=idx,
                species_name=species_name,
                confidence=prob,
                common_name_ua=species_data.get("common_name_ua"),
                edibility=EdibilityStatus(
                    species_data.get("edibility", "unknown")
                )
            )
            predictions.append(pred)

        top_pred = predictions[0]
        is_confident = top_pred.confidence >= settings.confidence_threshold

        # Generate warning for dangerous mushrooms
        warning = None
        if top_pred.edibility in [EdibilityStatus.POISONOUS, EdibilityStatus.DEADLY]:
            warning = f"WARNING: This mushroom ({top_pred.species_name}) is {top_pred.edibility.value.upper()}!"
        elif not is_confident:
            warning = "Low confidence prediction. Please provide a clearer photo."

        return PredictionResponse(
            success=True,
            predictions=predictions,
            top_prediction=top_pred,
            is_confident=is_confident,
            warning=warning
        )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @torch.no_grad()
    def get_attention_map(self, image: Union[Image.Image, str, Path]) -> Optional[str]:
        """
        Generate attention rollout map for the image.

        Returns:
            Base64 encoded PNG image of the attention heatmap overlay
        """
        if not self._is_loaded:
            return None

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Check if model is timm ViT (has blocks attribute)
        if not hasattr(self.model, 'blocks'):
            return None  # Attention maps only for timm ViT models

        # Get patch embedding
        x = self.model.patch_embed(img_tensor)
        B = x.shape[0]

        # Add CLS token
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add position embedding
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        # Collect attention from all layers (attention rollout)
        all_attns = []
        for block in self.model.blocks:
            # Get attention weights
            y = block.norm1(x)
            B, N, C = y.shape

            qkv = block.attn.qkv(y).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            all_attns.append(attn.mean(dim=1))  # Average across heads

            # Continue forward pass
            x = block(x)

        # Attention rollout: multiply attention matrices
        rollout = torch.eye(all_attns[0].shape[-1]).to(self.device)
        for attn in all_attns:
            attn = attn + torch.eye(attn.shape[-1]).to(self.device)  # Add residual connection
            attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize
            rollout = attn @ rollout

        # CLS attention to patches (skip CLS token at index 0)
        cls_attn = rollout[0, 0, 1:]  # (num_patches,)

        # Reshape to grid
        num_patches = cls_attn.shape[0]
        grid_size = int(num_patches ** 0.5)
        cls_attn = cls_attn.reshape(grid_size, grid_size)

        # Upsample to image size
        cls_attn = F.interpolate(
            cls_attn.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        cls_attn = cls_attn.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

        # Create overlay image
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)

        # Create heatmap (using hot colormap manually)
        heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
        # Hot colormap: black -> red -> yellow -> white
        heatmap[:, :, 0] = np.clip(cls_attn * 3, 0, 1) * 255  # Red channel
        heatmap[:, :, 1] = np.clip(cls_attn * 3 - 1, 0, 1) * 255  # Green channel
        heatmap[:, :, 2] = np.clip(cls_attn * 3 - 2, 0, 1) * 255  # Blue channel

        # Blend original image with heatmap
        alpha = 0.5
        overlay = (img_array * (1 - alpha) + heatmap * alpha).astype(np.uint8)

        # Convert to base64 PNG
        overlay_img = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_img.save(buffer, format='PNG')
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode('utf-8')


# Global classifier instance (singleton)
_classifier: MushroomClassifier = None


def get_classifier() -> MushroomClassifier:
    """Get or create the global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = MushroomClassifier()
    return _classifier