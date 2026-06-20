"""
src/models/hybrid_model.py

HybridGRUTransformer
=====================
The shared intrusion-detection backbone for the Federated IDS project.

A client's network-flow feature vector (length F) is reshaped into a
short sequence (seq_len, input_dim) -- see src/datasets/flnet_dataset.py.
The GRU first models local sequential/feature dependencies, a
Transformer encoder then models longer-range relationships across the
GRU outputs, and a small classifier head produces attack-class logits.

This module has ZERO project-specific dependencies (no Flower, no
training loop) so it can be imported as-is by:
  * Member 3, inside the Flower NumPyClient / server-side aggregation code
  * Member 2, via forward_features() / get_weights() for trust scoring
  * scripts/sanity_check_train.py, for a quick centralized smoke test

Usage
-----
    from src.models.hybrid_model import HybridGRUTransformer

    model = HybridGRUTransformer(input_dim=1, seq_len=64, num_classes=5)
    logits = model(x)                 # x: (batch, seq_len, input_dim)
    probs  = model.predict_proba(x)
    weights = model.get_weights()     # list[np.ndarray] -- Flower-ready
    model.set_weights(weights)
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.models.positional_encoding import PositionalEncoding


class HybridGRUTransformer(nn.Module):
    """GRU encoder -> Transformer encoder -> classification head."""

    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 64,
        num_classes: int = 2,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        gru_bidirectional: bool = True,
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        dropout: float = 0.2,
        pooling: str = "mean",  # "mean" | "last" | "cls"
    ):
        super().__init__()

        if transformer_d_model % transformer_nhead != 0:
            raise ValueError(
                f"transformer_d_model ({transformer_d_model}) must be divisible by "
                f"transformer_nhead ({transformer_nhead})"
            )
        if pooling not in ("mean", "last", "cls"):
            raise ValueError(f"pooling must be one of 'mean'/'last'/'cls', got {pooling!r}")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.pooling = pooling

        # ---- 1. GRU stage --------------------------------------------------
        gru_out_dim = gru_hidden_size * (2 if gru_bidirectional else 1)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=gru_bidirectional,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )

        # ---- 2. Bridge GRU output -> Transformer d_model --------------------
        self.input_proj = nn.Linear(gru_out_dim, transformer_d_model)
        self.pos_encoder = PositionalEncoding(transformer_d_model, max_len=seq_len + 8, dropout=dropout)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- 3. Transformer encoder stage -----------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        # ---- 4. Classification head ------------------------------------------
        self.norm = nn.LayerNorm(transformer_d_model)
        self.classifier = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_d_model // 2, num_classes),
        )

    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the pooled embedding BEFORE the classifier head.
        Useful for Member 2's trust scoring / explainability work
        (e.g. comparing client embeddings, or feeding SHAP)."""
        gru_out, _ = self.gru(x)                       # (B, T, gru_out_dim)
        proj = self.input_proj(gru_out)                # (B, T, d_model)

        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(proj.size(0), -1, -1)
            proj = torch.cat([cls_tokens, proj], dim=1)

        proj = self.pos_encoder(proj)
        encoded = self.transformer_encoder(proj)        # (B, T[+1], d_model)

        if self.pooling == "mean":
            pooled = encoded.mean(dim=1)
        elif self.pooling == "last":
            pooled = encoded[:, -1, :]
        else:  # "cls"
            pooled = encoded[:, 0, :]

        return self.norm(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> logits: (batch, num_classes)"""
        pooled = self.forward_features(x)
        return self.classifier(pooled)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        logits = self.forward(x)
        if was_training:
            self.train()
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).argmax(dim=-1)

    # ---- Flower-friendly weight helpers ---------------------------------
    def get_weights(self) -> List[np.ndarray]:
        """list[np.ndarray] ordered like state_dict() -- pass straight into
        Flower's `ndarrays_to_parameters` / `NumPyClient.get_parameters`."""
        return [val.detach().cpu().numpy() for val in self.state_dict().values()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        state_keys = list(self.state_dict().keys())
        new_state = {k: torch.tensor(w) for k, w in zip(state_keys, weights)}
        self.load_state_dict(new_state, strict=True)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Lightweight config snapshot -- handy for logging / experiment tracking."""
        return {
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "num_classes": self.num_classes,
            "pooling": self.pooling,
            "num_parameters": self.count_parameters(),
        }


if __name__ == "__main__":
    # Quick shape sanity check: python -m src.models.hybrid_model
    batch_size, seq_len, input_dim, num_classes = 8, 64, 1, 6
    model = HybridGRUTransformer(input_dim=input_dim, seq_len=seq_len, num_classes=num_classes)
    dummy = torch.randn(batch_size, seq_len, input_dim)
    out = model(dummy)
    print("Output shape:", tuple(out.shape))          # (8, 6)
    print("Config:", model.get_config())
