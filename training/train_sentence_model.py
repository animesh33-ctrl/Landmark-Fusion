import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Suppress MediaPipe warnings
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import (
    DEVICE, SEED, CHECKPOINT_DIR,
    ISL_SENTENCE_FRAMES,
    SEQUENCE_LENGTH, FEATURE_SIZE,
    SEQ_BATCH_SIZE, SEQ_LEARNING_RATE, SEQ_EPOCHS, SEQ_PATIENCE,
    SEQ_NUM_WORKERS, SEQ_LABEL_SMOOTHING, SEQ_DROPOUT,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS, TRANSFORMER_DIM_FF,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
)
from src.keypoint_extractor import bulk_extract_from_frame_folders
from src.dataset_loader import get_keypoint_dataloaders
from src.models.transformer_model import SignLanguageTransformer
from src.models.lstm_model import SignLanguageLSTM
from utils.metrics import (
    AverageMeter, EarlyStopping, save_checkpoint,
    compute_accuracy, compute_f1, full_classification_report,
)


torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    loss_meter = AverageMeter("train_loss")
    all_preds, all_labels = [], []

    for seqs, labels in tqdm(loader, desc="  Train", leave=False):
        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(seqs)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), seqs.size(0))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = compute_accuracy(np.array(all_labels), np.array(all_preds))
    return loss_meter.avg, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter("val_loss")
    all_preds, all_labels = [], []

    for seqs, labels in tqdm(loader, desc="  Val  ", leave=False):
        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(seqs)
        loss = criterion(logits, labels)

        loss_meter.update(loss.item(), seqs.size(0))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = compute_accuracy(np.array(all_labels), np.array(all_preds))
    return loss_meter.avg, acc


def train_sentence_model(model_type: str = "transformer"):
    assert model_type in ("transformer", "lstm"), \
        "model_type must be 'transformer' or 'lstm'"

    print("=" * 65)
    print(f"  TRAINING — {model_type.upper()} Sentence Model (ISL CSLRT)")
    print("=" * 65)
    print(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  [WARN] CUDA not available — training on CPU (will be slow)")

    cache_path = os.path.join(CHECKPOINT_DIR, "keypoint_cache.npz")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        print("  [Cache] Loading pre‑extracted keypoints …")
        data = np.load(cache_path, allow_pickle=True)
        sequences = data["sequences"]
        labels = data["labels"].tolist()
        label_map = data["label_map"].item()
    else:
        print("  [Extract] Running MediaPipe keypoint extraction …")
        sequences, labels, label_map = bulk_extract_from_frame_folders(
            ISL_SENTENCE_FRAMES,
            sequence_length=SEQUENCE_LENGTH,
            include_pose=False,
        )
        np.savez_compressed(cache_path,
                            sequences=sequences,
                            labels=np.array(labels),
                            label_map=label_map)
        print(f"  [Cache] Saved keypoints to {cache_path}")

    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}
    print(f"  Sequences: {len(sequences)}  |  Classes: {num_classes}")

    if len(sequences) == 0:
        print("[ERROR] No sequences extracted. Check the dataset folder.")
        return None, None

    train_loader, val_loader, test_loader, remap = get_keypoint_dataloaders(
        sequences, labels,
        batch_size=SEQ_BATCH_SIZE,
        num_workers=SEQ_NUM_WORKERS,
    )

    inv_label_map = {v: k for k, v in label_map.items()}  # old_idx → name
    inv_remap = {new: old for old, new in remap.items()}   # new_idx → old_idx
    num_classes = len(remap)
    idx_to_label = {new: inv_label_map[old]
                    for old, new in remap.items()}
    print(f"  Classes (after filter): {num_classes}")

    if model_type == "transformer":
        model = SignLanguageTransformer(
            input_size=FEATURE_SIZE,
            d_model=TRANSFORMER_D_MODEL,
            nhead=TRANSFORMER_NHEAD,
            num_layers=TRANSFORMER_NUM_LAYERS,
            dim_feedforward=TRANSFORMER_DIM_FF,
            num_classes=num_classes,
            dropout=SEQ_DROPOUT,
            max_len=SEQUENCE_LENGTH + 10,
        ).to(DEVICE)
    else:
        model = SignLanguageLSTM(
            input_size=FEATURE_SIZE,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            num_classes=num_classes,
            dropout=SEQ_DROPOUT,
            bidirectional=True,
        ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=SEQ_LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SEQ_LEARNING_RATE,
                                   weight_decay=1e-4)
    total_steps = SEQ_EPOCHS * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=SEQ_LEARNING_RATE * 10,
                           total_steps=total_steps,
                           pct_start=0.1, anneal_strategy="cos")
    early_stop = EarlyStopping(patience=SEQ_PATIENCE, mode="min")

    tag = model_type
    best_ckpt = os.path.join(CHECKPOINT_DIR, f"best_{tag}_sentence.pth")
    last_ckpt = os.path.join(CHECKPOINT_DIR, f"last_{tag}_sentence.pth")

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    for epoch in range(1, SEQ_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE)
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:>3}/{SEQ_EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
              f"{elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt)

        save_checkpoint(model, optimizer, epoch, val_loss, last_ckpt)

        if early_stop(val_loss):
            print(f"  [EarlyStopping] No improvement for "
                  f"{SEQ_PATIENCE} epochs. Stopping.")
            break

    print("\n" + "=" * 65)
    print("  EVALUATION on TEST SET")
    print("=" * 65)

    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"  Test Loss: {test_loss:.4f}  |  Test Acc: {test_acc:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels_batch in test_loader:
            seqs = seqs.to(DEVICE)
            logits = model(seqs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    class_names = [idx_to_label[i] for i in range(num_classes)]
    report = full_classification_report(
        np.array(all_labels), np.array(all_preds), class_names)
    print(report)

    print(f"\n  Best checkpoint saved to: {best_ckpt}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["transformer", "lstm"],
                        help="Sequence model type")
    args = parser.parse_args()
    train_sentence_model(model_type=args.model)
