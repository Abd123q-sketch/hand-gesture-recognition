"""
Preprocessing and hand-region extraction utilities.

Features:
- Basic hand detection using skin-color thresholding or full-frame (fallback)
- Resize & normalization
- Sequence generation for temporal models
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Tuple

import cv2
import numpy as np

from hand_gesture_recognition.utils.config import PreprocessingConfig
from hand_gesture_recognition.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class FrameWithLabel:
    frame: np.ndarray
    label: int


def detect_hand_region(frame: np.ndarray, use_full_frame: bool = False) -> np.ndarray:
    """
    Very simple hand-region approximation using skin-color thresholding.
    If use_full_frame is True, returns the full frame (useful as a baseline).
    """
    if use_full_frame:
        return frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.blur(mask, (5, 5))
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return frame

    # Largest contour as hand candidate
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    if w * h < 100:  # too small, fallback
        return frame
    return frame[y : y + h, x : x + w]


def preprocess_image(
    frame_bgr: np.ndarray,
    config: PreprocessingConfig,
    use_full_frame: bool = False,
) -> np.ndarray:
    """Detect hand region, resize, and normalize."""
    roi = detect_hand_region(frame_bgr, use_full_frame=use_full_frame)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (config.image_size, config.image_size))
    roi = roi.astype(np.float32) / 255.0 if config.normalize else roi.astype(
        np.float32
    )
    return roi


def build_sequences(
    frames_with_labels: Iterable[FrameWithLabel],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequences of frames for temporal models (CNN+LSTM).
    Uses a sliding window with stride 1.
    """
    window: Deque[np.ndarray] = deque(maxlen=sequence_length)
    labels: Deque[int] = deque(maxlen=sequence_length)

    sequences: List[np.ndarray] = []
    sequence_labels: List[int] = []

    for item in frames_with_labels:
        window.append(item.frame)
        labels.append(item.label)
        if len(window) == sequence_length:
            sequences.append(np.stack(list(window), axis=0))
            # majority label
            vals, counts = np.unique(np.array(labels), return_counts=True)
            sequence_labels.append(int(vals[np.argmax(counts)]))

    if not sequences:
        logger.warning("No sequences built (not enough frames).")
        return np.empty((0, sequence_length, 1, 1, 1)), np.empty((0,))

    return np.stack(sequences, axis=0), np.array(sequence_labels, dtype=np.int64)


def preprocess_folder_images(
    raw_root: str | Path,
    class_to_idx: dict,
    config: PreprocessingConfig,
    output_dir: str | Path,
) -> None:
    """
    Preprocess raw images and save NumPy arrays for faster training.
    """
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X: List[np.ndarray] = []
    y: List[int] = []

    for class_name, idx in class_to_idx.items():
        class_dir = raw_root / class_name
        if not class_dir.is_dir():
            logger.warning("Class directory missing: %s", class_dir)
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for img_path in class_dir.glob(ext):
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                roi = preprocess_image(
                    frame, config=config, use_full_frame=config.use_full_frame
                )
                X.append(roi)
                y.append(idx)

    if not X:
        logger.warning("No images found to preprocess in %s", raw_root)
        return

    X_arr = np.stack(X, axis=0)
    y_arr = np.array(y, dtype=np.int64)
    np.save(output_dir / "X.npy", X_arr)
    np.save(output_dir / "y.npy", y_arr)
    logger.info(
        "Saved preprocessed arrays to %s (X shape=%s, y shape=%s)",
        output_dir,
        X_arr.shape,
        y_arr.shape,
    )


