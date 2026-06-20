"""
scripts/run_preprocessing.py

End-to-end entry point for Member 1's data pipeline.

Typical workflow:
    1. python scripts/generate_synthetic_data.py   (optional -- testing only)
    2. python scripts/run_preprocessing.py          <-- this script
    3. python scripts/sanity_check_train.py

Produces:
    data/processed/flnet2023_clean.csv
    data/processed/clients/client_<id>/{train,val}.csv
    data/processed/clients/clients_manifest.json
    data/processed/test/global_test.csv
    artifacts/scaler.joblib
    artifacts/metadata.joblib
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.partition import run_partition_pipeline
from src.data.preprocess import run_preprocessing_pipeline


def main() -> None:
    print("=" * 70)
    print("STEP 1/2: Cleaning + feature selection + scaling")
    print("=" * 70)
    run_preprocessing_pipeline()

    print()
    print("=" * 70)
    print("STEP 2/2: Federated client partitioning + global test set")
    print("=" * 70)
    run_partition_pipeline()

    print()
    print("Done. Client splits are ready under data/processed/clients/")
    print("Member 3 can now call: src.datasets.flnet_dataset.load_client_loaders(client_id)")


if __name__ == "__main__":
    main()
