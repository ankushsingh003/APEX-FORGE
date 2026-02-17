import os


# ================================================================================================================================
# Raw Data Path
# ================================================================================================================================

RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"


# ================================================================================================================================
# Processed Data Path
# ================================================================================================================================

PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

# ================================================================================================================================
# Model Path
# ================================================================================================================================

MODEL_DIR = "artifacts/model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")