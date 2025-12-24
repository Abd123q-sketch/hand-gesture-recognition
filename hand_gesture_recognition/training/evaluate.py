import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from hand_gesture_recognition.utils.config import DATASET_ROOT
from hand_gesture_recognition.utils.logger import get_logger
from hand_gesture_recognition.data.dataset_loader import create_dataloaders_from_folders, SequenceFromFramesDataset
from hand_gesture_recognition.models.cnn_model import SimpleCNN
from hand_gesture_recognition.models.cnn_lstm_model import CNNLSTM


logger = get_logger(__name__)


def evaluate(model, loader, device: torch.device):
    model.eval()
    total_correct, total = 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            total_correct += (pred == y).sum().item()
            total += y.numel()
    return total_correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--dataset_root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--model_type", choices=["cnn", "cnn_lstm"], default="cnn")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=12)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_root = Path(args.dataset_root)

    # Build loaders
    if args.model_type == "cnn":
        _, _, test_loader, idx_to_class = create_dataloaders_from_folders(
            dataset_root, batch_size=64, image_size=(args.image_size, args.image_size)
        )
        num_classes = len(idx_to_class)
        model = SimpleCNN(num_classes=num_classes)
    else:
        test_split = dataset_root / "test"
        ds = SequenceFromFramesDataset(test_split, sequence_length=args.sequence_length, image_size=(args.image_size, args.image_size))
        test_loader = DataLoader(ds, batch_size=16, shuffle=False)
        # infer classes from dataset
        num_classes = len(ds.class_names)
        model = CNNLSTM(num_classes=num_classes)

    model.to(device)

    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    acc = evaluate(model, test_loader, device)
    logger.info("Test accuracy: %.4f", acc)


if __name__ == "__main__":
    main()
