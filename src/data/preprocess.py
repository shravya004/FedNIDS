"""
src/data/preprocess.py

Loads raw FLNET2023 CSV files (as distributed by the dataset authors),
cleans them, derives labels, and produces ONE clean tabular dataset
(features + label + client_id) that downstream code (src/data/partition.py)
splits into per-client federated learning shards.

Expected raw layout -- place the official download here exactly like this
(this mirrors the dataset author's own README):

    data/raw/FLNET2023/
        Normal/CSV/Dataset-<id>.csv
        DDoS/CSV/Dataset-<id>-<TrafficType>.csv
        DoS/CSV/Dataset-<id>-<TrafficType>.csv
        Infiltration/CSV/Dataset-<id>.csv
        Web/CSV/Dataset-<id>-<TrafficType>.csv
        TEST/CSV/*.csv

If a CSV already has a "Label" column it is trusted as-is; otherwise the
label is derived from the parent attack folder name (e.g. everything
under DDoS/CSV/ is labelled "DDOS").

Run directly for a one-shot run:
    python -m src.data.preprocess
or via the wrapper:
    python scripts/run_preprocessing.py
"""

import logging
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Matches "Dataset-3.csv" or "Dataset-3-TCP.csv"
FILENAME_RE = re.compile(r"Dataset-(\d+)(?:-([A-Za-z]+))?\.csv$", re.IGNORECASE)


def _extract_client_and_traffic(filename: str):
    match = FILENAME_RE.search(filename)
    if not match:
        return None, None
    client_id = int(match.group(1))
    traffic_type = match.group(2)
    return client_id, traffic_type


def _read_single_csv(csv_path: Path, attack_folder: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        logger.warning("Skipping unreadable file %s (%s)", csv_path, exc)
        return None

    if df.empty:
        return None

    client_id, _traffic_type = _extract_client_and_traffic(csv_path.name)
    if client_id is None:
        logger.warning("Could not parse client id from filename %s, skipping", csv_path.name)
        return None

    # Normalise the label column name if the CSV already has one
    label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
    if label_col is not None:
        df = df.rename(columns={label_col: "Label"})
    else:
        df["Label"] = "BENIGN" if attack_folder.lower() == "normal" else attack_folder.upper()

    df["client_id"] = client_id
    df["attack_family"] = attack_folder.upper()
    return df


def load_raw_dataset(raw_dir: Path = config.DATA_RAW_DIR,
                      attack_folders=config.ATTACK_FOLDERS) -> pd.DataFrame:
    """Walks the FLNET2023 raw folder structure and concatenates all client CSVs."""
    frames = []
    for folder in attack_folders:
        csv_dir = raw_dir / folder / "CSV"
        if not csv_dir.exists():
            logger.warning("Folder not found, skipping: %s", csv_dir)
            continue
        csv_files = sorted(csv_dir.glob("*.csv"))
        logger.info("Found %d CSV file(s) in %s", len(csv_files), csv_dir)
        for csv_path in csv_files:
            df = _read_single_csv(csv_path, folder)
            if df is not None:
                frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No usable CSV files found under {raw_dir}.\n"
            f"  -> If you just want to test the pipeline, run:\n"
            f"       python scripts/generate_synthetic_data.py\n"
            f"  -> Otherwise download the real FLNET2023 dataset and place it at "
            f"that path, preserving the folder structure described in the README."
        )

    full_df = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Loaded raw dataset: %d rows, %d columns", *full_df.shape)
    return full_df


def load_global_test_set(raw_dir: Path = config.DATA_RAW_DIR,
                          test_folder: str = config.TEST_FOLDER) -> pd.DataFrame:
    """Loads the dataset author's official held-out TEST/ folder, if present."""
    csv_dir = raw_dir / test_folder / "CSV"
    if not csv_dir.exists():
        logger.warning("TEST folder not found at %s -- no held-out global test set", csv_dir)
        return pd.DataFrame()

    frames = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            logger.warning("Skipping unreadable test file %s (%s)", csv_path, exc)
            continue
        label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
        if label_col is not None:
            df = df.rename(columns={label_col: "Label"})
        else:
            df["Label"] = csv_path.stem.upper()
        df["client_id"] = -1  # global test set is not tied to a single client
        df["attack_family"] = csv_path.stem.upper()
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    test_df = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Loaded global test set: %d rows", len(test_df))
    return test_df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drops identifier columns, coerces to numeric, fixes inf/NaN, drops constants."""
    df = df.copy()

    meta_cols = ["Label", "client_id", "attack_family"]
    drop_cols = [c for c in df.columns if c in config.NON_FEATURE_COLUMNS]
    candidate_feature_cols = [c for c in df.columns if c not in drop_cols + meta_cols]

    numeric_df = df[candidate_feature_cols].apply(pd.to_numeric, errors="coerce")
    # keep only columns that actually contain at least one real numeric value
    numeric_cols = [c for c in numeric_df.columns if numeric_df[c].notna().any()]
    numeric_df = numeric_df[numeric_cols]

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    numeric_df = numeric_df.fillna(0)

    nunique = numeric_df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        logger.info("Dropping %d constant-value columns", len(constant_cols))
        numeric_df = numeric_df.drop(columns=constant_cols)

    cleaned = pd.concat(
        [numeric_df.reset_index(drop=True), df[meta_cols].reset_index(drop=True)], axis=1
    )
    cleaned = cleaned.dropna(subset=["Label"]).reset_index(drop=True)
    return cleaned


def select_features(df: pd.DataFrame, corr_threshold: float = config.CORR_THRESHOLD) -> pd.DataFrame:
    """Drops one column from each pair of highly-correlated numeric features."""
    meta_cols = ["Label", "client_id", "attack_family"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    if len(feature_cols) < 2:
        return df

    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]

    if to_drop:
        logger.info("Dropping %d highly correlated features (threshold=%.2f)", len(to_drop), corr_threshold)
        df = df.drop(columns=to_drop)
    return df


def encode_labels(df: pd.DataFrame, mode: str = config.LABEL_MODE):
    """Adds an integer 'target' column. Returns (df, class_names)."""
    df = df.copy()
    if mode == "binary":
        df["target"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
        classes = ["BENIGN", "ATTACK"]
    else:
        encoder = LabelEncoder()
        df["target"] = encoder.fit_transform(df["Label"].astype(str))
        classes = list(encoder.classes_)
    return df, classes


def fit_scaler(df: pd.DataFrame, feature_cols) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler


def run_preprocessing_pipeline(save: bool = True) -> pd.DataFrame:
    """End-to-end: load -> clean -> select features -> encode labels -> scale -> save."""
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset()
    clean_df = clean_dataframe(raw_df)
    selected_df = select_features(clean_df)
    labeled_df, classes = encode_labels(selected_df, mode=config.LABEL_MODE)

    meta_cols = ["Label", "client_id", "attack_family", "target"]
    feature_cols = [c for c in labeled_df.columns if c not in meta_cols]

    scaler = fit_scaler(labeled_df, feature_cols)
    labeled_df[feature_cols] = scaler.transform(labeled_df[feature_cols])

    if save:
        out_path = config.DATA_PROCESSED_DIR / "flnet2023_clean.csv"
        labeled_df.to_csv(out_path, index=False)
        joblib.dump(scaler, config.ARTIFACTS_DIR / "scaler.joblib")
        joblib.dump(
            {"classes": classes, "feature_cols": feature_cols, "label_mode": config.LABEL_MODE},
            config.ARTIFACTS_DIR / "metadata.joblib",
        )
        logger.info("Saved cleaned dataset -> %s", out_path)
        logger.info("Classes (%d): %s", len(classes), classes)
        logger.info("Feature count: %d", len(feature_cols))

    return labeled_df


if __name__ == "__main__":
    run_preprocessing_pipeline()
