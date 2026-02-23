
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def create_vit_model(num_classes, pretrained=True, freeze_backbone=False):
    """
    Create a Vision Transformer model for mushroom classification.

    Args:
        num_classes: Number of mushroom species to classify
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze the backbone layers

    Returns:
        ViT model with modified classification head
    """
    if pretrained:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
    else:
        model = vit_b_16(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classification head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model


def create_vit_small(num_classes, pretrained=True):
    """
    Create a smaller ViT variant using timm library.
    Requires: pip install timm
    """
    try:
        import timm
        model = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        return model
    except ImportError:
        print("timm not installed. Using torchvision ViT instead.")
        return create_vit_model(num_classes, pretrained)