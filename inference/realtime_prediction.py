import os
import sys
import time
import collections
import argparse
import cv2
import numpy as np
import torch
import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import (
    DEVICE, CHECKPOINT_DIR,
    IMG_SIZE, SEQUENCE_LENGTH, FEATURE_SIZE,
    WEBCAM_INDEX, PREDICTION_THRESHOLD,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS, TRANSFORMER_DIM_FF,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    SEQ_DROPOUT, CNN_DROPOUT, IMG_CHANNELS,
)
from src.preprocessing import get_val_transforms
from src.keypoint_extractor import extract_keypoints_from_frame
from src.models.cnn_model import SignLanguageCNN
from src.models.transformer_model import SignLanguageTransformer
from src.models.lstm_model import SignLanguageLSTM

def _load_label_map(path: str) -> dict:
    cache = os.path.join(CHECKPOINT_DIR, "keypoint_cache.npz")
    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        label_map = data["label_map"].item()
        return label_map  # {name: old_idx}
    return {}


def _detect_num_classes_from_ckpt(ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    for key in reversed(list(sd.keys())):
        if "weight" in key and sd[key].dim() == 2:
            return sd[key].shape[0]
    return 0


def _build_sentence_idx_to_label() -> dict:
    cache = os.path.join(CHECKPOINT_DIR, "keypoint_cache.npz")
    if not os.path.exists(cache):
        return {}

    data = np.load(cache, allow_pickle=True)
    label_map = data["label_map"].item()   # {name: old_idx}
    labels = data["labels"].tolist()

    inv_label_map = {v: k for k, v in label_map.items()}  # old_idx → name

    from collections import Counter
    counts = Counter(labels)
    keep = {cls for cls, cnt in counts.items() if cnt >= 3}
    unique_sorted = sorted(keep)
    remap = {old: new for new, old in enumerate(unique_sorted)}

    return {new: inv_label_map[old] for old, new in remap.items()}


def load_word_model(num_classes: int, ckpt_path: str | None = None):
    if ckpt_path is None:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best_cnn_word.pth")
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Word model checkpoint not found: {ckpt_path}")
        return None, None

    model = SignLanguageCNN(num_classes=num_classes,
                            in_channels=IMG_CHANNELS,
                            dropout=CNN_DROPOUT).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Word Model] Loaded from {ckpt_path}")

    from config.config import INDIAN_DATASET_PATH
    if os.path.exists(INDIAN_DATASET_PATH):
        classes = sorted([
            d for d in os.listdir(INDIAN_DATASET_PATH)
            if os.path.isdir(os.path.join(INDIAN_DATASET_PATH, d))
        ])
        idx_to_label = {i: c for i, c in enumerate(classes)}
    else:
        idx_to_label = {i: str(i) for i in range(num_classes)}

    return model, idx_to_label


def load_sentence_model(num_classes: int = 0,
                        model_type: str = "transformer",
                        ckpt_path: str | None = None):
    tag = model_type
    if ckpt_path is None:
        ckpt_path = os.path.join(CHECKPOINT_DIR,
                                  f"best_{tag}_sentence.pth")
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Sentence model checkpoint not found: {ckpt_path}")
        return None, None

    if num_classes == 0:
        num_classes = _detect_num_classes_from_ckpt(ckpt_path)
        print(f"  [Auto] Detected {num_classes} classes from checkpoint")

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
        ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Sentence Model] Loaded from {ckpt_path}")

    idx_to_label = _build_sentence_idx_to_label()
    return model, idx_to_label


def predict_word(model, frame_bgr: np.ndarray, transform,
                 idx_to_label: dict):

    from PIL import Image
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    label = idx_to_label.get(pred.item(), str(pred.item()))
    return label, conf.item()


def predict_sentence(model, keypoint_buffer: np.ndarray,
                     idx_to_label: dict):

    tensor = torch.tensor(keypoint_buffer, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).to(DEVICE)  # (1, T, F)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    label = idx_to_label.get(pred.item(), str(pred.item()))
    return label, conf.item()


def run_realtime(sentence_model_type: str = "transformer"):
    """Open webcam and run real‑time sign language recognition."""

    print("=" * 65)
    print("  REAL‑TIME SIGN LANGUAGE RECOGNITION")
    print("=" * 65)
    print("  Controls:  q = quit  |  m = toggle mode  |  c = clear")

    word_model, word_labels = None, {}
    sentence_model, sentence_labels = None, {}

    from config.config import INDIAN_DATASET_PATH
    if os.path.exists(INDIAN_DATASET_PATH):
        num_word_classes = len([
            d for d in os.listdir(INDIAN_DATASET_PATH)
            if os.path.isdir(os.path.join(INDIAN_DATASET_PATH, d))
        ])
        word_model, word_labels = load_word_model(num_word_classes)

    sentence_model, sentence_labels = load_sentence_model(
        num_classes=0, model_type=sentence_model_type)

    if word_model is None and sentence_model is None:
        print("[ERROR] No trained models found. Train first using main.py.")
        return

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    mode = "word" if word_model is not None else "sentence"
    keypoint_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    sentence_history: list[str] = []
    transform = get_val_transforms()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index {WEBCAM_INDEX}).")
        return

    print(f"  Mode: {mode.upper()}")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror
            display = frame.copy()
            h, w = frame.shape[:2]

            kp = extract_keypoints_from_frame(frame, holistic,
                                               include_pose=False)
            keypoint_buffer.append(kp)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display, results.left_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display, results.right_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS)

            label_text = ""
            confidence = 0.0

            if mode == "word" and word_model is not None:
                label_text, confidence = predict_word(
                    word_model, frame, transform, word_labels)

            elif mode == "sentence" and sentence_model is not None:
                if len(keypoint_buffer) == SEQUENCE_LENGTH:
                    buf = np.array(keypoint_buffer, dtype=np.float32)
                    label_text, confidence = predict_sentence(
                        sentence_model, buf, sentence_labels)

            cv2.putText(display, f"Mode: {mode.upper()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 200, 0), 2)

            if label_text and confidence >= PREDICTION_THRESHOLD:
                text = f"{label_text}  ({confidence:.0%})"
                cv2.putText(display, text,
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)

                if mode == "sentence":
                    if (not sentence_history or
                            sentence_history[-1] != label_text):
                        sentence_history.append(label_text)

            if sentence_history:
                hist_text = " | ".join(sentence_history[-5:])
                cv2.putText(display, hist_text,
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (200, 200, 255), 2)

            cv2.imshow("ISL Recognition", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mode = "sentence" if mode == "word" else "word"
                keypoint_buffer.clear()
                print(f"  Switched to {mode.upper()} mode")
            elif key == ord("c"):
                sentence_history.clear()
                keypoint_buffer.clear()
                print("  Cleared buffers")

    cap.release()
    cv2.destroyAllWindows()
    print("  Webcam closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["transformer", "lstm"])
    args = parser.parse_args()
    run_realtime(sentence_model_type=args.model)
