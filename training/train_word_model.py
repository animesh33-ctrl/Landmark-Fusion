import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import (
    DEVICE, SEED, CHECKPOINT_DIR,
    INDIAN_DATASET_PATH,
    CNN_BATCH_SIZE, CNN_LEARNING_RATE, CNN_EPOCHS, CNN_PATIENCE,
    CNN_NUM_WORKERS, CNN_LABEL_SMOOTHING, CNN_DROPOUT, CNN_WEIGHT_DECAY,
    IMG_CHANNELS,
)
from src.dataset_loader import get_image_dataloaders
from src.models.cnn_model import SignLanguageCNN
from utils.metrics import (
    AverageMeter, EarlyStopping, save_checkpoint,
    compute_accuracy, compute_f1, full_classification_report,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter("train_loss")
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = compute_accuracy(np.array(all_labels), np.array(all_preds))
    return loss_meter.avg, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter("val_loss")
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss_meter.update(loss.item(), images.size(0))
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = compute_accuracy(np.array(all_labels), np.array(all_preds))
    return loss_meter.avg, acc


def train_word_model():
    print("=" * 65)
    print("  TRAINING — CNN Word Recognition Model (Indian Dataset)")
    print("=" * 65)
    print(f"  Device : {DEVICE}")

    train_loader, val_loader, test_loader, label_to_idx, idx_to_label = \
        get_image_dataloaders(INDIAN_DATASET_PATH,
                              batch_size=CNN_BATCH_SIZE,
                              num_workers=CNN_NUM_WORKERS)

    num_classes = len(label_to_idx)
    print(f"  Classes: {num_classes}")

    model = SignLanguageCNN(num_classes=num_classes,
                            in_channels=IMG_CHANNELS,
                            dropout=CNN_DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=CNN_LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=CNN_LEARNING_RATE,
                                   weight_decay=CNN_WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stop = EarlyStopping(patience=CNN_PATIENCE, mode="min")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_cnn_word.pth")
    last_ckpt = os.path.join(CHECKPOINT_DIR, "last_cnn_word.pth")

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    for epoch in range(1, CNN_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE)

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:>3}/{CNN_EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
              f"{elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt)

        # Save last model every epoch
        save_checkpoint(model, optimizer, epoch, val_loss, last_ckpt)

        # Early stopping
        if early_stop(val_loss):
            print(f"  [EarlyStopping] No improvement for "
                  f"{CNN_PATIENCE} epochs. Stopping.")
            break

    print("\n" + "=" * 65)
    print("  EVALUATION on TEST SET")
    print("=" * 65)

    # Loading best checkpoint
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"  Test Loss: {test_loss:.4f}  |  Test Acc: {test_acc:.4f}")

    # Detailed report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    class_names = [idx_to_label[i] for i in range(num_classes)]
    report = full_classification_report(
        np.array(all_labels), np.array(all_preds), class_names)
    print(report)

    print(f"\n  Best checkpoint saved to: {best_ckpt}")
    return model, history


if __name__ == "__main__":
    train_word_model()
