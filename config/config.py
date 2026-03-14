"""
config/config.py - Central configuration for PHI De-identification System.
"""
import os
from pathlib import Path

ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
REPORTS_DIR     = ROOT_DIR / "reports"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
LOGS_DIR        = ROOT_DIR / "logs"

for d in [DATA_DIR, REPORTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.txt"
DEV_FILE   = DATA_DIR / "dev.txt"
TEST_FILE  = DATA_DIR / "test.txt"
VOCAB_FILE = DATA_DIR / "vocab.txt"
TAG_FILE   = DATA_DIR / "tags.txt"

# ── Model hyperparameters ─────────────────────────────────────────────────────
EMBEDDING_DIM       = 256    # increased from 128
HIDDEN_DIM          = 512    # increased from 256
DROPOUT             = 0.4    # slightly higher to fight overfitting
BATCH_SIZE          = 32
LEARNING_RATE       = 0.0005 # reduced from 0.001 — slower, more stable
EPOCHS              = 30     # increased from 10
MAX_SEQUENCE_LENGTH = 64     # increased from 50
GRAD_CLIP           = 3.0
LR_PATIENCE         = 3      # reduce LR if no improvement for 3 epochs
LR_FACTOR           = 0.5    # multiply LR by this on plateau
MIN_VOCAB_FREQ      = 1      # include all words
LOG_INTERVAL        = 50     # log every N batches

# ── Vocabulary ────────────────────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX   = 0
UNK_IDX   = 1

# ── BIO tag set ───────────────────────────────────────────────────────────────
ENTITY_TYPES = ["NAME", "DOCTOR", "HOSPITAL", "LOCATION", "DATE", "PHONE", "EMAIL", "ID"]
TAGS         = ["O"] + [f"B-{e}" for e in ENTITY_TYPES] + [f"I-{e}" for e in ENTITY_TYPES]
TAG2IDX      = {tag: idx for idx, tag in enumerate(TAGS)}
IDX2TAG      = {idx: tag for tag, idx in TAG2IDX.items()}
NUM_TAGS     = len(TAGS)

# ── Dataset generation ────────────────────────────────────────────────────────
SYNTHETIC_SENTENCES = 150_000   # tripled from 50k
TRAIN_RATIO = 0.70
DEV_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Redaction ─────────────────────────────────────────────────────────────────
REDACT_TEMPLATE  = "[RADICATED]"
BEST_MODEL_PATH  = CHECKPOINTS_DIR / "best_model.pt"
FINAL_MODEL_PATH = CHECKPOINTS_DIR / "final_model.pt"