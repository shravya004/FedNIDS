"""
config.py

Single source of truth for paths and pipeline settings. Everything in
src/ and scripts/ imports from here -- change values once, not in five
different files.
"""

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw" / "FLNET2023"

DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CLIENT_SPLITS_DIR = DATA_PROCESSED_DIR / "clients"
GLOBAL_TEST_DIR = DATA_PROCESSED_DIR / "test"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

ATTACK_FOLDERS = ["Normal", "DDoS", "DoS", "Infiltration", "Web"]
TEST_FOLDER = "TEST"

SEED = 42
NUM_CLIENTS = 10
VAL_SPLIT = 0.15
LABEL_MODE = "multiclass"

NON_FEATURE_COLUMNS = [
    "Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
    "Src Port", "Source Port", "Dst Port", "Destination Port",
    "Timestamp", "Protocol", "Label", "label", "ClientID",
]

CORR_THRESHOLD = 0.95
SEQ_FEATURE_DIM = 1
