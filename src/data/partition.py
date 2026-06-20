"""
src/data/partition.py

Splits the cleaned FLNET2023 dataset into per-client federated learning
shards, mirroring the dataset's natural D1..D10 router/client structure
(so client partitions are realistically non-IID, not artificially
random -- this is good for your trust-scoring experiments too).

Also prepares the held-out global TEST/ set using the EXACT SAME scaler
and feature columns the training data was processed with, so Member 3's
global model evaluation is unbiased and shape-compatible.

Run AFTER src/data/preprocess.py (or simply use the wrapper):
    python scripts/run_preprocessing.py
"""

import json
import logging

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.data.preprocess import clean_dataframe, load_global_test_set

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_processed_dataset() -> pd.DataFrame:
    path = config.DATA_PROCESSED_DIR / "flnet2023_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data.preprocess` "
            f"(or scripts/run_preprocessing.py) first."
        )
    return pd.read_csv(path)


def partition_by_client(df: pd.DataFrame, val_split: float = config.VAL_SPLIT,
                         seed: int = config.SEED) -> dict:
    """
    Returns/saves {client_id: {"train_rows", "val_rows", "class_distribution"}}
    using each client's OWN traffic only -- one CSV pair per client under
    data/processed/clients/client_<id>/{train,val}.csv
    """
    config.CLIENT_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    client_ids = sorted(c for c in df["client_id"].unique() if c > 0)  # -1 == global test rows

    for client_id in client_ids:
        client_df = df[df["client_id"] == client_id].reset_index(drop=True)

        stratify = client_df["target"] if client_df["target"].nunique() >= 2 and len(client_df) >= 20 else None
        train_df, val_df = train_test_split(
            client_df, test_size=val_split, random_state=seed, stratify=stratify
        )

        client_dir = config.CLIENT_SPLITS_DIR / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(client_dir / "train.csv", index=False)
        val_df.to_csv(client_dir / "val.csv", index=False)

        manifest[str(client_id)] = {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "class_distribution": train_df["Label"].value_counts().to_dict(),
        }
        logger.info("Client %s -> train=%d  val=%d", client_id, len(train_df), len(val_df))

    with open(config.CLIENT_SPLITS_DIR / "clients_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def prepare_global_test_set() -> None:
    """Cleans the official TEST/ folder using the SAME scaler + feature columns
    fit during training, so the global model can be evaluated fairly."""
    metadata_path = config.ARTIFACTS_DIR / "metadata.joblib"
    scaler_path = config.ARTIFACTS_DIR / "scaler.joblib"
    if not metadata_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Run preprocessing first to generate scaler/metadata artifacts.")

    metadata = joblib.load(metadata_path)
    scaler = joblib.load(scaler_path)
    feature_cols = metadata["feature_cols"]
    classes = metadata["classes"]

    raw_test_df = load_global_test_set()
    if raw_test_df.empty:
        logger.warning("No global TEST/ folder found -- skipping global test set creation.")
        return

    clean_test_df = clean_dataframe(raw_test_df)

    # align columns exactly to what the model was trained on
    for col in feature_cols:
        if col not in clean_test_df.columns:
            clean_test_df[col] = 0.0
    clean_test_df = clean_test_df[feature_cols + ["Label", "client_id", "attack_family"]]

    if config.LABEL_MODE == "binary":
        clean_test_df["target"] = (clean_test_df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
    else:
        label_to_idx = {c: i for i, c in enumerate(classes)}
        clean_test_df["target"] = clean_test_df["Label"].astype(str).map(
            lambda x: label_to_idx.get(x, -1)
        )
        before = len(clean_test_df)
        clean_test_df = clean_test_df[clean_test_df["target"] != -1]
        dropped = before - len(clean_test_df)
        if dropped:
            logger.warning("Dropped %d test rows with labels unseen during training", dropped)

    clean_test_df[feature_cols] = scaler.transform(clean_test_df[feature_cols])

    config.GLOBAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.GLOBAL_TEST_DIR / "global_test.csv"
    clean_test_df.to_csv(out_path, index=False)
    logger.info("Saved global test set (%d rows) -> %s", len(clean_test_df), out_path)


def run_partition_pipeline() -> None:
    df = _load_processed_dataset()
    partition_by_client(df)
    prepare_global_test_set()


if __name__ == "__main__":
    run_partition_pipeline()
