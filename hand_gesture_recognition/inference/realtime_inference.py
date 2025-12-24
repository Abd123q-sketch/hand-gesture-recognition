import argparse
from collections import deque, Counter
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch import nn

from hand_gesture_recognition.utils.config import InferenceConfig, PreprocessingConfig, DATASET_ROOT
from hand_gesture_recognition.utils.logger import get_logger
from hand_gesture_recognition.models.cnn_model import SimpleCNN
from hand_gesture_recognition.models.cnn_lstm_model import CNNLSTM

logger = get_logger(__name__)


def load_class_names(dataset_root: Path) -> List[str]:
    train_dir = dataset_root / "train"
    if not train_dir.exists():
        raise RuntimeError(f"Train dir not found at {train_dir}")
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def majority_vote(buffer: deque[str]) -> Optional[str]:
    if not buffer:
        return None
    c = Counter(buffer)
    return c.most_common(1)[0][0]


def run_static(weights: Path, dataset_root: Path, image_size: int, device: str, cfg: InferenceConfig):
    classes = load_class_names(dataset_root)
    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(weights, map_location=device)["model_state"])
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(cfg.webcam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    buffer = deque(maxlen=cfg.smoothing_window)

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cfg.flip_horizontal:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (image_size, image_size)).astype(np.float32) / 255.0
            x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = classes[pred_idx]
            conf = float(probs[pred_idx])

            label = "No hands detected"
            color = (0, 0, 255)  # Rouge par défaut
            
            if conf >= cfg.confidence_threshold:
                buffer.append(pred_label)
                label = majority_vote(buffer) or pred_label
                color = (0, 255, 0)  # Vert quand détecté

            cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Realtime Gesture (CNN)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def run_dynamic(weights: Path, dataset_root: Path, image_size: int, seq_len: int, device: str, cfg: InferenceConfig):
    classes = load_class_names(dataset_root)
    model = CNNLSTM(num_classes=len(classes))
    model.load_state_dict(torch.load(weights, map_location=device)["model_state"])
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(cfg.webcam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    frames: deque[np.ndarray] = deque(maxlen=seq_len)
    buffer = deque(maxlen=cfg.smoothing_window)

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cfg.flip_horizontal:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (image_size, image_size)).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))  # C,H,W

            label = "No hands detected"
            conf = 0.0
            color = (0, 0, 255)  # Rouge par défaut
            
            if len(frames) == seq_len:
                x = torch.stack(list(frames), dim=0).unsqueeze(0).to(device)  # 1,T,C,H,W
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_label = classes[pred_idx]
                pred_conf = float(probs[pred_idx])
                
                if pred_conf >= cfg.confidence_threshold:
                    buffer.append(pred_label)
                    label = majority_vote(buffer) or pred_label
                    conf = pred_conf
                    color = (0, 255, 0)  # Vert quand détecté

            cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Realtime Gesture (CNN+LSTM)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Realtime hand gesture inference")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--model", choices=["cnn", "cnn_lstm"], default="cnn")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=12)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--smoothing", type=int, default=InferenceConfig.smoothing_window)
    parser.add_argument("--conf", type=float, default=InferenceConfig.confidence_threshold)
    parser.add_argument("--webcam", type=int, default=InferenceConfig.webcam_index)
    parser.add_argument("--flip", action="store_true")
    args = parser.parse_args()

    cfg = InferenceConfig(
        smoothing_window=args.smoothing,
        confidence_threshold=args.conf,
        webcam_index=args.webcam,
        flip_horizontal=args.flip,
    )

    device = args.device
    weights = Path(args.weights)
    dataset_root = Path(args.dataset_root)

    if args.model == "cnn":
        run_static(weights, dataset_root, args.image_size, device, cfg)
    else:
        run_dynamic(weights, dataset_root, args.image_size, args.sequence_length, device, cfg)


if __name__ == "__main__":
    main()
