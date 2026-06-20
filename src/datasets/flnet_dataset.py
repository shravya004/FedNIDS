"""
src/datasets/flnet_dataset.py

PyTorch Dataset wrapper around the per-client CSV shards produced by
src/data/partition.py. Reshapes each tabular flow-feature vector into a
short sequence so it can be fed directly into HybridGRUTransformer
(see src/models/hybrid_model.py).

This is what Member 3 will import for the Flower client-side training
loop, e.g.:

    from src.datasets.flnet_dataset import load_client_loaders

    train_loader, val_loader, seq_len, input_dim = load_client_loaders(client_id=3)
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config

META_COLUMNS = ["Label", "client_id", "attack_family", "target"]


class FLNetDataset(Dataset):
    """Loads ONE client's processed CSV (train.csv or val.csv)."""

    def __init__(self, csv_path, feature_cols: Optional[List[str]] = None,
                 seq_feature_dim: int = config.SEQ_FEATURE_DIM):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        self.df = pd.read_csv(self.csv_path)
        self.feature_cols = feature_cols or [c for c in self.df.columns if c not in META_COLUMNS]
        self.seq_feature_dim = seq_feature_dim

        n_features = len(self.feature_cols)
        if n_features % seq_feature_dim != 0:
            # pad with zero-columns so reshape always works, rather than crashing
            pad_n = seq_feature_dim - (n_features % seq_feature_dim)
            for i in range(pad_n):
                pad_col = f"__pad_{i}"
                self.df[pad_col] = 0.0
                self.feature_cols.append(pad_col)

        self.seq_len = len(self.feature_cols) // seq_feature_dim

        self.X = self.df[self.feature_cols].values.astype(np.float32)
        self.y = self.df["target"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].reshape(self.seq_len, self.seq_feature_dim)
        return torch.from_numpy(x), torch.tensor(self.y[idx])

    @property
    def input_dim(self) -> int:
        return self.seq_feature_dim

    @property
    def num_classes(self) -> int:
        return int(self.df["target"].nunique())


def load_client_loaders(client_id: int, batch_size: int = 64,
                         seq_feature_dim: int = config.SEQ_FEATURE_DIM,
                         shuffle_train: bool = True):
    """
    Convenience helper for Member 3's Flower client code.

    Returns: (train_loader, val_loader, seq_len, input_dim)
    """
    client_dir = config.CLIENT_SPLITS_DIR / f"client_{client_id}"
    train_ds = FLNetDataset(client_dir / "train.csv", seq_feature_dim=seq_feature_dim)
    val_ds = FLNetDataset(client_dir / "val.csv", feature_cols=list(train_ds.feature_cols),
                           seq_feature_dim=seq_feature_dim)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_ds.seq_len, train_ds.input_dim


def load_global_test_loader(batch_size: int = 64, seq_feature_dim: int = config.SEQ_FEATURE_DIM):
    """Loads the aligned, scaled global test set for unbiased global-model evaluation."""
    test_path = config.GLOBAL_TEST_DIR / "global_test.csv"
    test_ds = FLNetDataset(test_path, seq_feature_dim=seq_feature_dim)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False), test_ds.seq_len, test_ds.input_dim
