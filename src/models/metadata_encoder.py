import torch
import torch.nn as nn
import math


class MetadataEncoder(nn.Module):
    """
    Encodes categorical metadata (habitat, substrate) and cyclical month
    into a 768-dim vector for injection into ViT.
    """

    def __init__(self, habitat_vocab, substrate_vocab, embed_dim=768, dropout=0.1):
        super().__init__()

        self.habitat_vocab = habitat_vocab
        self.substrate_vocab = substrate_vocab

        # +1 for UNK token (index = len(vocab))
        self.habitat_emb = nn.Embedding(len(habitat_vocab) + 1, 32)
        self.substrate_emb = nn.Embedding(len(substrate_vocab) + 1, 32)

        # Month uses sinusoidal encoding: 8 frequencies -> 16 dims (sin + cos)
        self.month_dim = 16
        self.num_frequencies = 8

        # Total input: 32 (habitat) + 32 (substrate) + 16 (month) = 80
        input_dim = 32 + 32 + self.month_dim

        # Projection MLP: 80 -> 256 -> 768
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Initialize embeddings
        nn.init.normal_(self.habitat_emb.weight, std=0.02)
        nn.init.normal_(self.substrate_emb.weight, std=0.02)

    def encode_month_sinusoidal(self, months):
        """
        Encode month (1-12) using sinusoidal encoding so that
        month 12 and month 1 are close in embedding space.

        Args:
            months: (B,) tensor of month values (1-12)

        Returns:
            (B, 16) tensor of sinusoidal features
        """
        device = months.device
        B = months.shape[0]

        # Normalize month to [0, 2*pi] cycle
        # Month 1 -> 0, Month 12 -> 11/12 * 2*pi (close to 2*pi, thus close to 0)
        theta = (months.float() - 1) / 12.0 * 2 * math.pi  # (B,)

        # Generate frequencies: 1, 2, 3, ..., num_frequencies
        frequencies = torch.arange(1, self.num_frequencies + 1, device=device).float()  # (8,)

        # Compute sin and cos for each frequency
        # theta: (B,) -> (B, 1), frequencies: (8,) -> (1, 8)
        angles = theta.unsqueeze(1) * frequencies.unsqueeze(0)  # (B, 8)

        sin_enc = torch.sin(angles)  # (B, 8)
        cos_enc = torch.cos(angles)  # (B, 8)

        # Concatenate: (B, 16)
        month_encoding = torch.cat([sin_enc, cos_enc], dim=1)

        return month_encoding

    def forward(self, habitat_ids, substrate_ids, months):
        """
        Args:
            habitat_ids: (B,) tensor of habitat indices
            substrate_ids: (B,) tensor of substrate indices
            months: (B,) tensor of month values (1-12)

        Returns:
            (B, 768) metadata embedding
        """
        # Embed categorical features
        habitat_feat = self.habitat_emb(habitat_ids)      # (B, 32)
        substrate_feat = self.substrate_emb(substrate_ids)  # (B, 32)

        # Encode month with sinusoidal
        month_feat = self.encode_month_sinusoidal(months)  # (B, 16)

        # Concatenate all features
        combined = torch.cat([habitat_feat, substrate_feat, month_feat], dim=1)  # (B, 80)

        # Project to embed_dim
        output = self.proj(combined)  # (B, 768)

        return output