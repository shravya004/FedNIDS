"""
config.py

Single source of truth for paths and pipeline settings. Everything in
src/ and scripts/ imports from here -- change values once, not in five
different files.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

# Place the official FLNET2023 download here, preserving the folder layout
# described in the dataset README (Normal/, DDoS/, DoS/, Infiltration/,
# Web/, TEST/ -- each containing a CSV/ subfolder).
DATA_RAW_DIR = ROOT_DIR / "data" / "raw" / "FLNET2023"

DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CLIENT_SPLITS_DIR = DATA_PROCESSED_DIR / "clients"
GLOBAL_TEST_DIR = DATA_PROCESSED_DIR / "test"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Folders matching the FLNET2023 dataset layout (see dataset README)
ATTACK_FOLDERS = ["Normal", "DDoS", "DoS", "Infiltration", "Web"]
TEST_FOLDER = "TEST"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Federated split settings
# ---------------------------------------------------------------------------
# FLNET2023 was collected from 10 router nodes (D1..D10) -- we map these
# 1:1 onto FL clients to preserve the dataset's real, non-IID structure.
NUM_CLIENTS = 10

# Per-client stratified train/val split ratio
VAL_SPLIT = 0.15

# "binary"      -> BENIGN vs ATTACK
# "multiclass"  -> BENIGN vs DDoS vs DoS vs Infiltration vs Web (attack family)
LABEL_MODE = "multiclass"

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
# Columns CICFlowMeter-style exports typically include that must NEVER be
# used as model features (identifiers / leakage). Extend this list if your
# downloaded CSV header uses slightly different names.
NON_FEATURE_COLUMNS = [
    "Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
    "Src Port", "Source Port", "Dst Port", "Destination Port",
    "Timestamp", "Protocol", "Label", "label", "ClientID",
]

# Drop one of any pair of features whose absolute correlation exceeds this
CORR_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Sequence reshape for the GRU+Transformer hybrid model
# ---------------------------------------------------------------------------
# Each flow's feature vector (length F) is reshaped into
# (seq_len, SEQ_FEATURE_DIM) where seq_len = F / SEQ_FEATURE_DIM.
# Keep at 1 unless you have a specific reason to group features per step.
SEQ_FEATURE_DIM = 1
