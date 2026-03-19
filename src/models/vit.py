
import torch
import torch.nn as nn
import timm


def create_vit_model(num_classes, pretrained=True, freeze_backbone=False,
                     drop_path_rate=0.2, drop_rate=0.1):
    """
    Create a Vision Transformer model for mushroom classification using timm.

    Args:
        num_classes: Number of mushroom species to classify
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze the backbone layers
        drop_path_rate: Stochastic depth rate for regularization
        drop_rate: Dropout rate for classifier

    Returns:
        ViT model with modified classification head
    """
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

    return model


def create_vit_small(num_classes, pretrained=True):
    """
    Create a smaller ViT variant using timm library.
    """
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=0.1,
        drop_rate=0.1
    )
    return model


def create_vit_large(num_classes, pretrained=True, drop_path_rate=0.2, drop_rate=0.1):
    """
    Create a large ViT variant using timm library.
    """
    model = timm.create_model(
        'vit_large_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model