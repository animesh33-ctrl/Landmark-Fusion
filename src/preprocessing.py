"""
src/preprocessing.py
====================
Image preprocessing and data‑augmentation transforms for the Indian
Sign Language image dataset (CNN pipeline).
"""

import os
import numpy as np
from typing import Tuple, List

from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# ── project imports ──────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import (
    IMG_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED,
)


# ──────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Training transforms with data augmentation.

    Augmentation includes:
        - Random rotation (±15°)
        - Random horizontal flip
        - Random affine (shift + scale)
        - Color jitter
        - Random erasing
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_val_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Validation / test transforms – deterministic, no augmentation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────────────
# Label encoding helpers
# ──────────────────────────────────────────────────────────────────────

def build_label_map(dataset_root: str) -> Tuple[dict, dict]:
    """
    Scan *dataset_root* and return two dicts:
        label_to_idx : { class_folder_name : int }
        idx_to_label : { int : class_folder_name }
    """
    classes = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


# ──────────────────────────────────────────────────────────────────────
# Image + label collection
# ──────────────────────────────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_image_paths_and_labels(
        dataset_root: str) -> Tuple[List[str], List[int], dict, dict]:
    """
    Walk dataset_root (one subfolder per class) and return:
        image_paths, labels, label_to_idx, idx_to_label
    """
    label_to_idx, idx_to_label = build_label_map(dataset_root)
    image_paths: List[str] = []
    labels: List[int] = []

    for class_name, idx in label_to_idx.items():
        class_dir = os.path.join(dataset_root, class_name)
        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(idx)

    print(f"[Preprocessing] Found {len(image_paths)} images across "
          f"{len(label_to_idx)} classes.")
    return image_paths, labels, label_to_idx, idx_to_label


# ──────────────────────────────────────────────────────────────────────
# Train / Val / Test split
# ──────────────────────────────────────────────────────────────────────

def split_dataset(image_paths: List[str], labels: List[int],
                  train_ratio: float = TRAIN_RATIO,
                  val_ratio: float = VAL_RATIO,
                  test_ratio: float = TEST_RATIO,
                  seed: int = SEED):
    """
    Stratified split into train / val / test sets.

    Returns
    -------
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # first split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = \
        train_test_split(
            image_paths, labels,
            test_size=test_ratio,
            random_state=seed,
            stratify=labels,
        )

    # second split: train vs val  (relative ratio within train+val)
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = \
        train_test_split(
            train_val_paths, train_val_labels,
            test_size=relative_val,
            random_state=seed,
            stratify=train_val_labels,
        )

    print(f"[Split] Train: {len(train_paths)}  |  "
          f"Val: {len(val_paths)}  |  Test: {len(test_paths)}")
    return (train_paths, train_labels,
            val_paths, val_labels,
            test_paths, test_labels)
