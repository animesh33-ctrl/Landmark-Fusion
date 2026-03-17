import os
import sys

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional
from tqdm import tqdm

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import (
    HAND_LANDMARKS, POSE_UPPER_LANDMARKS,
    FEATURE_SIZE, FEATURE_SIZE_WITH_POSE,
    SEQUENCE_LENGTH,
)


mp_holistic = mp.solutions.holistic

UPPER_BODY_INDICES = list(range(0, 11))  # 0‑10 in MediaPipe Pose


def _landmarks_to_array(landmarks, indices: Optional[list] = None) -> np.ndarray:

    if landmarks is None:
        n = len(indices) if indices else 21
        return np.zeros(n * 3, dtype=np.float32)
    lm = landmarks.landmark
    if indices:
        lm = [lm[i] for i in indices]
    return np.array([[p.x, p.y, p.z] for p in lm],
                    dtype=np.float32).flatten()


def extract_keypoints_from_frame(
        frame_bgr: np.ndarray,
        holistic,
        include_pose: bool = False) -> np.ndarray:
  
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = holistic.process(frame_rgb)

    left_hand = _landmarks_to_array(results.left_hand_landmarks)
    right_hand = _landmarks_to_array(results.right_hand_landmarks)

    if include_pose:
        pose = _landmarks_to_array(results.pose_landmarks,
                                   indices=UPPER_BODY_INDICES)
        return np.concatenate([left_hand, right_hand, pose])
    return np.concatenate([left_hand, right_hand])


def extract_keypoints_from_frame_folder(
        folder_path: str,
        sequence_length: int = SEQUENCE_LENGTH,
        include_pose: bool = False) -> Optional[np.ndarray]:

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])
    if len(frame_files) == 0:
        return None

    feat_size = FEATURE_SIZE_WITH_POSE if include_pose else FEATURE_SIZE

    all_keypoints = []
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=0,          # 0 = lite (faster), 1 = full
        min_detection_confidence=0.5,
    ) as holistic:
        for fname in frame_files:
            img = cv2.imread(os.path.join(folder_path, fname))
            if img is None:
                continue
            kp = extract_keypoints_from_frame(img, holistic,
                                              include_pose=include_pose)
            all_keypoints.append(kp)

    if len(all_keypoints) == 0:
        return None

    all_keypoints = np.array(all_keypoints, dtype=np.float32)

    n_frames = len(all_keypoints)
    if n_frames == sequence_length:
        return all_keypoints
    elif n_frames > sequence_length:
        indices = np.linspace(0, n_frames - 1, sequence_length, dtype=int)
        return all_keypoints[indices]
    else:
        pad = np.zeros((sequence_length - n_frames, feat_size),
                       dtype=np.float32)
        return np.vstack([all_keypoints, pad])


def extract_keypoints_from_video(
        video_path: str,
        sequence_length: int = SEQUENCE_LENGTH,
        include_pose: bool = False) -> Optional[np.ndarray]:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return None

    feat_size = FEATURE_SIZE_WITH_POSE if include_pose else FEATURE_SIZE
    all_keypoints = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            kp = extract_keypoints_from_frame(frame, holistic,
                                              include_pose=include_pose)
            all_keypoints.append(kp)

    cap.release()

    if len(all_keypoints) == 0:
        return None

    all_keypoints = np.array(all_keypoints, dtype=np.float32)
    n_frames = len(all_keypoints)

    if n_frames == sequence_length:
        return all_keypoints
    elif n_frames > sequence_length:
        indices = np.linspace(0, n_frames - 1, sequence_length, dtype=int)
        return all_keypoints[indices]
    else:
        pad = np.zeros((sequence_length - n_frames, feat_size),
                       dtype=np.float32)
        return np.vstack([all_keypoints, pad])


def bulk_extract_from_frame_folders(
        root_dir: str,
        sequence_length: int = SEQUENCE_LENGTH,
        include_pose: bool = False):

    sequences = []
    label_names = []

    class_dirs = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    label_map = {name: idx for idx, name in enumerate(class_dirs)}

    print(f"[Keypoints] Extracting from {len(class_dirs)} classes in "
          f"{root_dir} ...")

    for class_name in tqdm(class_dirs, desc="Classes"):
        class_path = os.path.join(root_dir, class_name)

        # Each class folder may contain sub‑folders (one per sample)
        sub_items = sorted(os.listdir(class_path))
        sub_dirs = [s for s in sub_items
                    if os.path.isdir(os.path.join(class_path, s))]

        if sub_dirs:
            for sample_dir in sub_dirs:
                sample_path = os.path.join(class_path, sample_dir)
                seq = extract_keypoints_from_frame_folder(
                    sample_path, sequence_length, include_pose)
                if seq is not None:
                    sequences.append(seq)
                    label_names.append(class_name)
        else:
            seq = extract_keypoints_from_frame_folder(
                class_path, sequence_length, include_pose)
            if seq is not None:
                sequences.append(seq)
                label_names.append(class_name)

    sequences = np.array(sequences, dtype=np.float32)
    labels = [label_map[n] for n in label_names]
    print(f"[Keypoints] Extracted {len(sequences)} sequences, "
          f"shape per sequence: {sequences[0].shape if len(sequences) > 0 else 'N/A'}")
    return sequences, labels, label_map
