import torch
import torch.nn as nn
import timm
from timm.layers import trunc_normal_

from .metadata_encoder import MetadataEncoder


class MushroomViTWithMetadata(nn.Module):
    """
    Vision Transformer with early fusion of metadata.
    Injects a learned metadata token into the patch sequence before transformer blocks.

    Architecture:
        - ViT backbone (vit_base_patch16_224)
        - MetadataEncoder for habitat/substrate/month
        - Metadata token injected after CLS token
        - Final sequence: [CLS, META, patch_1, ..., patch_196] = 198 tokens
    """

    def __init__(self, num_classes, habitat_vocab, substrate_vocab,
                 pretrained=True, drop_path_rate=0.2, drop_rate=0.1):
        super().__init__()

        self.habitat_vocab = habitat_vocab
        self.substrate_vocab = substrate_vocab

        # Load ViT backbone without classification head (num_classes=0)
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate
        )

        # Get embed dimension from ViT
        self.embed_dim = self.vit.embed_dim  # 768 for vit_base

        # Metadata encoder
        self.meta_encoder = MetadataEncoder(
            habitat_vocab=habitat_vocab,
            substrate_vocab=substrate_vocab,
            embed_dim=self.embed_dim
        )

        # Learnable metadata position token
        self.meta_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.meta_token, std=0.02)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, images, habitat_ids, substrate_ids, months):
        """
        Forward pass with image and metadata fusion.

        Args:
            images: (B, 3, 224, 224) input images
            habitat_ids: (B,) habitat indices
            substrate_ids: (B,) substrate indices
            months: (B,) month values (1-12)

        Returns:
            (B, num_classes) logits
        """
        B = images.shape[0]

        # 1. Patch embedding
        x = self.vit.patch_embed(images)  # (B, 196, 768)

        # 2. Prepend CLS token and add positional embedding
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls_token, x], dim=1)  # (B, 197, 768)
        x = x + self.vit.pos_embed  # (B, 197, 768)

        # 3. Encode metadata
        meta_feat = self.meta_encoder(habitat_ids, substrate_ids, months)  # (B, 768)
        meta_feat = meta_feat.unsqueeze(1)  # (B, 1, 768)

        # Add learnable meta token position embedding
        meta_feat = meta_feat + self.meta_token  # (B, 1, 768)

        # 4. Insert metadata token after CLS: [CLS, META, patches...]
        x = torch.cat([x[:, :1], meta_feat, x[:, 1:]], dim=1)  # (B, 198, 768)

        # 5. Apply transformer blocks
        x = self.vit.pos_drop(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        # 6. Classification from CLS token
        cls_output = x[:, 0]  # (B, 768)
        logits = self.head(cls_output)  # (B, num_classes)

        return logits

    def get_attention_maps(self, images, habitat_ids, substrate_ids, months):
        """
        Get attention maps for visualization (optional utility method).
        """
        B = images.shape[0]

        x = self.vit.patch_embed(images)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.pos_embed

        meta_feat = self.meta_encoder(habitat_ids, substrate_ids, months).unsqueeze(1)
        meta_feat = meta_feat + self.meta_token
        x = torch.cat([x[:, :1], meta_feat, x[:, 1:]], dim=1)

        x = self.vit.pos_drop(x)

        attention_maps = []
        for block in self.vit.blocks:
            x, attn = block(x, return_attention=True) if hasattr(block, 'return_attention') else (block(x), None)
            if attn is not None:
                attention_maps.append(attn)

        return attention_maps