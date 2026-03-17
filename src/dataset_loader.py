import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import (
    CNN_BATCH_SIZE, CNN_NUM_WORKERS,
    SEQ_BATCH_SIZE, SEQ_NUM_WORKERS,
    SEQUENCE_LENGTH, FEATURE_SIZE,
    SEED,
)
from src.preprocessing import (
    get_train_transforms, get_val_transforms,
    collect_image_paths_and_labels, split_dataset,
)

class SignImageDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int],
                 transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open {img_path}: {e}")

            image = Image.new("RGB", (128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_image_dataloaders(
        dataset_root: str,
        batch_size: int = CNN_BATCH_SIZE,
        num_workers: int = CNN_NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict, dict]:
    
    image_paths, labels, label_to_idx, idx_to_label = \
        collect_image_paths_and_labels(dataset_root)

    (train_p, train_l,
     val_p, val_l,
     test_p, test_l) = split_dataset(image_paths, labels)

    train_ds = SignImageDataset(train_p, train_l,
                                transform=get_train_transforms())
    val_ds = SignImageDataset(val_p, val_l,
                              transform=get_val_transforms())
    test_ds = SignImageDataset(test_p, test_l,
                               transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    print(f"[DataLoader] Train batches: {len(train_loader)}  | "
          f"Val batches: {len(val_loader)}  | "
          f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, label_to_idx, idx_to_label


class KeypointSeqDataset(Dataset):

    def __init__(self, sequences: np.ndarray, labels: List[int],
                 augment: bool = False):
        self.sequences = sequences.astype(np.float32)
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq = self.sequences[idx].copy()  # (T, F)
        label = self.labels[idx]

        if self.augment:
            seq = self._augment(seq)

        return torch.tensor(seq, dtype=torch.float32), label

    @staticmethod
    def _augment(seq: np.ndarray) -> np.ndarray:

        noise = np.random.randn(*seq.shape).astype(np.float32) * 0.005
        seq = seq + noise

        # Random temporal shift (roll ±3 frames)
        shift = np.random.randint(-3, 4)
        seq = np.roll(seq, shift, axis=0)

        # Random scale jitter per frame
        scale = np.random.uniform(0.95, 1.05, size=(seq.shape[0], 1))
        seq = seq * scale.astype(np.float32)

        return seq


def _filter_rare_classes(sequences: np.ndarray, labels: List[int],
                        min_samples: int = 3):

    from collections import Counter
    counts = Counter(labels)
    keep = {cls for cls, cnt in counts.items() if cnt >= min_samples}
    removed = {cls for cls, cnt in counts.items() if cnt < min_samples}

    if removed:
        print(f"  [Filter] Removing {len(removed)} classes with < "
              f"{min_samples} samples ({sum(counts[c] for c in removed)} "
              f"samples dropped)")

    mask = [i for i, lbl in enumerate(labels) if lbl in keep]
    filtered_seq = sequences[mask]
    filtered_lbl = [labels[i] for i in mask]

    # Re‑map labels to contiguous 0..N-1
    unique_sorted = sorted(keep)
    remap = {old: new for new, old in enumerate(unique_sorted)}
    filtered_lbl = [remap[l] for l in filtered_lbl]

    print(f"  [Filter] Kept {len(unique_sorted)} classes, "
          f"{len(filtered_seq)} samples")
    return filtered_seq, filtered_lbl, remap


def get_keypoint_dataloaders(
        sequences: np.ndarray,
        labels: List[int],
        batch_size: int = SEQ_BATCH_SIZE,
        num_workers: int = SEQ_NUM_WORKERS,
        val_ratio: float = 0.15,
        test_ratio: float = 0.10,
        min_samples_per_class: int = 3,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    from sklearn.model_selection import train_test_split
    from collections import Counter

    # Filter classes with too few samples for stratified split
    sequences, labels, remap = _filter_rare_classes(
        sequences, labels, min_samples=min_samples_per_class)

    n_classes = len(set(labels))
    n_samples = len(labels)

    # Adjust test_size if too few samples per class
    min_test = max(n_classes, 1)
    actual_test_size = max(test_ratio, min_test / n_samples + 0.01)
    actual_test_size = min(actual_test_size, 0.30)  # cap at 30%

    try:
        X_tv, X_test, y_tv, y_test = train_test_split(
            sequences, labels,
            test_size=actual_test_size,
            random_state=SEED,
            stratify=labels,
        )
    except ValueError:
        # Fallback: non‑stratified split
        print("  [WARN] Stratified split failed, using random split")
        X_tv, X_test, y_tv, y_test = train_test_split(
            sequences, labels,
            test_size=actual_test_size,
            random_state=SEED,
        )

    rel_val = val_ratio / (1.0 - actual_test_size)
    rel_val = min(rel_val, 0.40)  

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=rel_val,
            random_state=SEED,
            stratify=y_tv,
        )
    except ValueError:
        print("  [WARN] Stratified val split failed, using random split")
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=rel_val,
            random_state=SEED,
        )

    train_ds = KeypointSeqDataset(X_train, y_train, augment=True)
    val_ds = KeypointSeqDataset(X_val, y_val, augment=False)
    test_ds = KeypointSeqDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    print(f"[SeqDataLoader] Train: {len(train_ds)} | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader, remap
