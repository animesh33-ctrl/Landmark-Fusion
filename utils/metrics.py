import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)



def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray,
                      average: str = "weighted") -> float:
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray,
                   average: str = "weighted") -> float:
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray,
               average: str = "weighted") -> float:
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def full_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: list[str] | None = None) -> str:
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )


class AverageMeter:

    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0,
                 mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric: float) -> bool:
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, loss: float, path: str) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [Checkpoint] Loaded ← {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["loss"]
