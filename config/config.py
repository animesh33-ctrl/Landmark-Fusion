"""
config/config.py
================
Central configuration for the Indian Sign Language Recognition project.
All paths, hyperparameters, and constants are defined here.
"""

import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INDIAN_DATASET_PATH = r"C:\Sign Language\dataset\Indian"

ISL_CSLRT_ROOT = r"C:\Sign Language\dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"
ISL_SENTENCE_FRAMES = os.path.join(ISL_CSLRT_ROOT, "Frames_Sentence_Level")
ISL_WORD_FRAMES = os.path.join(ISL_CSLRT_ROOT, "Frames_Word_Level")
ISL_SENTENCE_VIDEOS = os.path.join(ISL_CSLRT_ROOT, "Videos_Sentence_Level")

REQUIRED_DATASET_PATHS = [
    INDIAN_DATASET_PATH,
    ISL_CSLRT_ROOT,
    ISL_SENTENCE_FRAMES,
    ISL_WORD_FRAMES,
    ISL_SENTENCE_VIDEOS,
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

IMG_SIZE = 128                # resize target (IMG_SIZE x IMG_SIZE)
IMG_CHANNELS = 3              # RGB
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 1e-3
CNN_EPOCHS = 60
CNN_PATIENCE = 8              # early‑stopping patience
CNN_NUM_WORKERS = 4
CNN_LABEL_SMOOTHING = 0.1
CNN_DROPOUT = 0.4
CNN_WEIGHT_DECAY = 1e-4

# Train / val / test split ratios
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

# MediaPipe landmarks:  left_hand(21×3) + right_hand(21×3) = 126
HAND_LANDMARKS = 21
POSE_UPPER_LANDMARKS = 11      # upper‑body subset of 33 pose landmarks
FEATURE_SIZE = (HAND_LANDMARKS * 3) * 2  # 126 – both hands only
# Optionally include upper body pose:  126 + 11*3 = 159
FEATURE_SIZE_WITH_POSE = FEATURE_SIZE + POSE_UPPER_LANDMARKS * 3

SEQUENCE_LENGTH = 30           # frames per sequence
SEQ_BATCH_SIZE = 32
SEQ_LEARNING_RATE = 3e-4
SEQ_EPOCHS = 80
SEQ_PATIENCE = 10
SEQ_NUM_WORKERS = 4
SEQ_LABEL_SMOOTHING = 0.1
SEQ_DROPOUT = 0.3

# Transformer‑specific
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DIM_FF = 512

# LSTM‑specific
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2

SEED = 42

WEBCAM_INDEX = 0
PREDICTION_THRESHOLD = 0.55   # minimum softmax confidence to display
