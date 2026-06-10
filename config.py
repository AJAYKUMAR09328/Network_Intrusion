import os
import torch
import random
import numpy as np

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_PATH = os.path.join(BASE_DIR, "data")
PROCESSED_FILE = os.path.join(BASE_DIR, "processed_data.pt")

# Classes
NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "DoS", "Probe"]

# Model parameters
HIDDEN_DIM = 64
DROPOUT = 0.4


# Data Split (Train / Validation / Test)
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.10
TEST_SPLIT = 0.30
# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Columns not used for training
EXCLUDE_COLS = [
    "Label",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Fwd Header Length.1"
]