import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from hand_gesture_recognition.utils.config import (
    DATASET_ROOT,
    MODELS_DIR,
    TrainingConfig,
    PreprocessingConfig,
)
from hand_gesture_recognition.utils.logger import get_logger
from hand_gesture_recognition.data.dataset_loader import (
    create_dataloaders_from_folders,
    SequenceFromFramesDataset,
)
from hand_gesture_recognition.models.cnn_model import SimpleCNN
from hand_gesture_recognition.models.cnn_lstm_model import CNNLSTM


logger = get_logger(__name__)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        else:
            raise RuntimeError("Unexpected batch format")
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), targets) * bs
        n += bs
    return total_loss / max(n, 1), total_acc / max(n, 1)


def evaluate(model, loader, criterion, device: torch.device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    if loader is None:
        return 0.0, 0.0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, targets) * bs
            n += bs
    return total_loss / max(n, 1), total_acc / max(n, 1)


def build_sequence_loaders(dataset_root: Path, batch_size: int, seq_len: int, image_size: int, num_workers: int):
    # Build train/val/test datasets for sequences
    def build_split(split: str):
        root = dataset_root / split
        if not root.exists():
            return None
        ds = SequenceFromFramesDataset(root, sequence_length=seq_len, image_size=(image_size, image_size))
        return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)

    return build_split("train"), build_split("val"), build_split("test"), None


def main():
    parser = argparse.ArgumentParser(description="Train hand gesture recognition models")
    parser.add_argument("--dataset_root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--model", type=str, choices=["cnn", "cnn_lstm"], default="cnn")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.num_epochs)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--num_workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--image_size", type=int, default=PreprocessingConfig.image_size)
    parser.add_argument("--sequence_length", type=int, default=TrainingConfig.sequence_length)
    parser.add_argument("--classes", type=str, nargs="*", default=None, help="Optional explicit class names in train folder order")
    parser.add_argument("--save_dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_root = Path(args.dataset_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if args.model == "cnn":
        train_loader, val_loader, test_loader, idx_to_class = create_dataloaders_from_folders(
            dataset_root,
            batch_size=args.batch_size,
            image_size=(args.image_size, args.image_size),
            num_workers=args.num_workers,
        )
    else:
        train_loader, val_loader, test_loader, idx_to_class = build_sequence_loaders(
            dataset_root, args.batch_size, args.sequence_length, args.image_size, args.num_workers
        )

    num_classes = len(idx_to_class) if idx_to_class is not None else None
    if num_classes is None:
        # derive from train set
        train_dir = dataset_root / "train"
        classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        num_classes = len(classes)

    # Model
    if args.model == "cnn":
        model = SimpleCNN(num_classes=num_classes)
    else:
        model = CNNLSTM(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_path = save_dir / f"{args.model}_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
        )

        if val_loader is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, best_path)
            logger.info("Saved new best model to %s", best_path)

    # Final test
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        logger.info("Test Loss %.4f Acc %.4f", test_loss, test_acc)

    # Save final
    final_path = save_dir / f"{args.model}_final.pt"
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs}, final_path)
    logger.info("Saved final model to %s", final_path)


if __name__ == "__main__":
    main()
