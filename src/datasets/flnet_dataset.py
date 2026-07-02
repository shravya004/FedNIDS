"""
src/datasets/flnet_dataset.py

PyTorch Dataset wrapper around the per-client CSV shards produced by
src/data/partition.py.

Optimized version:
- Debug mode for faster CPU training
- Easy switch to full dataset
- DataLoader optimization
- Hybrid GRU-Transformer compatible
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)

import config

# ==========================================================
# Debug Configuration
# ==========================================================

DEBUG_MODE = True

MAX_TRAIN_SAMPLES = 40000
MAX_VAL_SAMPLES = 8000

DEFAULT_BATCH_SIZE = 128
NUM_WORKERS = 0

META_COLUMNS = [
    "Label",
    "client_id",
    "attack_family",
    "target",
]


class FLNetDataset(Dataset):
    """
    Loads one processed FLNET2023 client dataset.
    """

    def __init__(
        self,
        csv_path,
        feature_cols: Optional[List[str]] = None,
        seq_feature_dim: int = config.SEQ_FEATURE_DIM,
    ):

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        self.df = pd.read_csv(self.csv_path)

        self.feature_cols = feature_cols or [

            column

            for column in self.df.columns

            if column not in META_COLUMNS

        ]

        self.seq_feature_dim = seq_feature_dim

        feature_count = len(self.feature_cols)

        if feature_count % seq_feature_dim != 0:

            padding = seq_feature_dim - (

                feature_count % seq_feature_dim

            )

            for index in range(padding):

                pad_column = f"__pad_{index}"

                self.df[pad_column] = 0.0

                self.feature_cols.append(

                    pad_column

                )

        self.seq_len = (

            len(self.feature_cols)

            // seq_feature_dim

        )

        self.X = (

            self.df[self.feature_cols]

            .values

            .astype(np.float32)

        )

        self.y = (

            self.df["target"]

            .values

            .astype(np.int64)

        )
        
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.X[idx].reshape(

            self.seq_len,

            self.seq_feature_dim,

        )

        return (

            torch.from_numpy(x),

            torch.tensor(self.y[idx]),

        )

    @property
    def input_dim(self) -> int:

        return self.seq_feature_dim

    @property
    def num_classes(self) -> int:

        return int(

            self.df["target"].nunique()

        )


# ==========================================================
# Client Data Loaders
# ==========================================================

def load_client_loaders(

    client_id: int,

    batch_size: int = DEFAULT_BATCH_SIZE,

    seq_feature_dim: int = config.SEQ_FEATURE_DIM,

    shuffle_train: bool = True,

):
    """
    Returns:

        train_loader,
        val_loader,
        seq_len,
        input_dim
    """

    client_dir = (

        config.CLIENT_SPLITS_DIR

        / f"client_{client_id}"

    )

    # ------------------------------------------
    # Load datasets
    # ------------------------------------------

    train_dataset = FLNetDataset(

        client_dir / "train.csv",

        seq_feature_dim=seq_feature_dim,

    )

    val_dataset = FLNetDataset(

        client_dir / "val.csv",

        feature_cols=list(train_dataset.feature_cols),

        seq_feature_dim=seq_feature_dim,

    )

    # Preserve model configuration BEFORE Subset
    seq_len = train_dataset.seq_len

    input_dim = train_dataset.input_dim

    # ------------------------------------------
    # Debug Mode
    # ------------------------------------------

    if DEBUG_MODE:

        if len(train_dataset) > MAX_TRAIN_SAMPLES:

            train_dataset = Subset(

                train_dataset,

                range(MAX_TRAIN_SAMPLES),

            )

        if len(val_dataset) > MAX_VAL_SAMPLES:

            val_dataset = Subset(

                val_dataset,

                range(MAX_VAL_SAMPLES),

            )

    # ------------------------------------------
    # DataLoaders
    # ------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=shuffle_train,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

    )

    return (

        train_loader,

        val_loader,

        seq_len,

        input_dim,

    )


# ==========================================================
# Global Test Loader
# ==========================================================

def load_global_test_loader(

    batch_size: int = DEFAULT_BATCH_SIZE,

    seq_feature_dim: int = config.SEQ_FEATURE_DIM,

):
    """
    Loads the global test dataset.
    """

    test_path = (

        config.GLOBAL_TEST_DIR

        / "global_test.csv"

    )

    test_dataset = FLNetDataset(

        test_path,

        seq_feature_dim=seq_feature_dim,

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

    )

    return (

        test_loader,

        test_dataset.seq_len,

        test_dataset.input_dim,

    )