"""
scripts/generate_synthetic_data.py

Generates a SMALL synthetic dataset that mimics the FLNET2023 folder /
filename layout, purely so the rest of the pipeline (preprocessing,
partitioning, model training) can be smoke-tested end-to-end on a laptop
in seconds -- without downloading the real multi-GB dataset.

DO NOT use this synthetic data for actual experiments/results. Replace
data/raw/FLNET2023/ with the real download before running real
experiments -- the preprocessing code expects the exact same folder
structure either way, so nothing else needs to change.

Run:
    python scripts/generate_synthetic_data.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config

NUM_FEATURES = 64
FEATURE_NAMES = [f"feat_{i}" for i in range(NUM_FEATURES)]
NUM_CLIENTS = config.NUM_CLIENTS
ROWS_PER_CLIENT_NORMAL = 400
ROWS_PER_CLIENT_ATTACK = 150

ATTACK_TRAFFIC_TYPES = {
    "Normal": [None],
    "DDoS": ["BOT", "STOMP", "TCP", "DYN"],
    "DoS": ["HULK", "SLOWHTTP"],
    "Web": ["SQL", "XSS", "CMD"],
    "Infiltration": [None],
}

rng = np.random.default_rng(config.SEED)


def _make_rows(n: int, label: str, shift: float = 0.0) -> pd.DataFrame:
    X = rng.normal(loc=shift, scale=1.0, size=(n, NUM_FEATURES))
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["Label"] = label
    return df


def generate() -> None:
    raw_dir = config.DATA_RAW_DIR
    print(f"Generating synthetic FLNET2023-style dataset under:\n  {raw_dir}\n")

    for folder, traffic_types in ATTACK_TRAFFIC_TYPES.items():
        label = "BENIGN" if folder == "Normal" else folder.upper()
        for client_id in range(1, NUM_CLIENTS + 1):
            for traffic in traffic_types:
                # not every client sees every attack type -- mirrors the real,
                # non-IID dataset where attacks were only launched from some nodes
                if folder != "Normal" and rng.random() < 0.4:
                    continue

                shift = 0.0 if folder == "Normal" else rng.uniform(1.5, 3.0)
                n_rows = ROWS_PER_CLIENT_NORMAL if folder == "Normal" else ROWS_PER_CLIENT_ATTACK
                df = _make_rows(n_rows, label, shift=shift)

                csv_dir = raw_dir / folder / "CSV"
                csv_dir.mkdir(parents=True, exist_ok=True)
                fname = f"Dataset-{client_id}.csv" if traffic is None else f"Dataset-{client_id}-{traffic}.csv"
                df.to_csv(csv_dir / fname, index=False)

    # held-out global TEST/ set
    test_dir = raw_dir / "TEST" / "CSV"
    test_dir.mkdir(parents=True, exist_ok=True)
    for folder in ATTACK_TRAFFIC_TYPES:
        label = "BENIGN" if folder == "Normal" else folder.upper()
        shift = 0.0 if folder == "Normal" else rng.uniform(1.5, 3.0)
        df = _make_rows(150, label, shift=shift)
        df.to_csv(test_dir / f"{folder.lower()}.csv", index=False)

    print("Synthetic dataset generation complete.")
    print(f"   -> {raw_dir}")
    print("\nNext step: python scripts/run_preprocessing.py")


if __name__ == "__main__":
    generate()
