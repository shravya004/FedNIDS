"""
tests/test_hybrid_model.py

Minimal shape / forward-pass tests for HybridGRUTransformer and the
baseline models. Run with:
    pytest -q
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.models.baseline_models import GRUClassifier, TransformerClassifier
from src.models.hybrid_model import HybridGRUTransformer


def test_hybrid_forward_shape():
    batch_size, seq_len, input_dim, num_classes = 4, 32, 1, 5
    model = HybridGRUTransformer(input_dim=input_dim, seq_len=seq_len, num_classes=num_classes)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch_size, num_classes)


def test_hybrid_predict_proba_sums_to_one():
    model = HybridGRUTransformer(input_dim=1, seq_len=16, num_classes=3)
    x = torch.randn(2, 16, 1)
    probs = model.predict_proba(x)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4)


def test_get_set_weights_roundtrip():
    model = HybridGRUTransformer(input_dim=1, seq_len=16, num_classes=3)
    weights = model.get_weights()
    model.set_weights(weights)  # should not raise
    assert len(weights) == len(list(model.state_dict().keys()))


def test_forward_features_embedding_shape():
    model = HybridGRUTransformer(input_dim=1, seq_len=16, num_classes=3, transformer_d_model=64, transformer_nhead=4)
    x = torch.randn(5, 16, 1)
    emb = model.forward_features(x)
    assert emb.shape == (5, 64)


def test_cls_pooling_mode():
    model = HybridGRUTransformer(input_dim=1, seq_len=16, num_classes=3, pooling="cls")
    x = torch.randn(2, 16, 1)
    out = model(x)
    assert out.shape == (2, 3)


def test_baseline_models_forward_shape():
    x = torch.randn(4, 32, 1)
    gru_model = GRUClassifier(input_dim=1, num_classes=5)
    trans_model = TransformerClassifier(input_dim=1, seq_len=32, num_classes=5)
    assert gru_model(x).shape == (4, 5)
    assert trans_model(x).shape == (4, 5)


def test_invalid_head_dim_raises():
    try:
        HybridGRUTransformer(transformer_d_model=100, transformer_nhead=3)
        assert False, "expected ValueError"
    except ValueError:
        pass
