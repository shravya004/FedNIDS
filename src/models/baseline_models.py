"""
src/models/baseline_models.py

Lightweight GRU-only and Transformer-only baselines, used purely for the
"baseline comparison" deliverable -- to show the hybrid model's lift over
either component alone. Same I/O contract as HybridGRUTransformer:
input (batch, seq_len, input_dim) -> output logits (batch, num_classes).
"""

import torch
import torch.nn as nn

from src.models.positional_encoding import PositionalEncoding


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int = 1, num_classes: int = 2,
                 hidden_size: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_size, num_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int = 1, seq_len: int = 64, num_classes: int = 2,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 8, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.pos_encoder(self.input_proj(x))
        encoded = self.transformer_encoder(proj)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)
