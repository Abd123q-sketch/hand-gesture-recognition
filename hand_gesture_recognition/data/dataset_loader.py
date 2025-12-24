"""
Dataset loading utilities for hand gesture recognition.

Supports:
- Folder-based image datasets (train/val/test with class subfolders)
- Optional Sign Language MNIST (if downloaded separately)
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ImageFolderDataset(Dataset):
    """
    Simple folder-based dataset:

    root/
      class_a/
        img1.jpg
      class_b/
        img2.jpg
    """

    def __init__(
        self,
        root_dir: str | Path,
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (128, 128),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.image_size = image_size

        self.samples: List[Tuple[Path, int]] = []
        self._scan()

    def _scan(self) -> None:
        for class_name, idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(img)
        # HWC -> CHW
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor


def create_dataloaders_from_folders(
    dataset_root: str | Path,
    batch_size: int,
    image_size: Tuple[int, int],
    num_workers: int = 0,
    transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[int, str]]:
    """
    Create train/val/test DataLoaders from a folder structure:

    dataset_root/
      train/
        open_palm/
        fist/
        ...
      val/
      test/
    """
    
    dataset_root = Path(dataset_root)

    train_root = dataset_root / "train"
    class_names = sorted(
        [d.name for d in train_root.iterdir() if d.is_dir()]
    )
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_ds = ImageFolderDataset(
        train_root, class_to_idx, transform=transform, image_size=image_size
    )

    val_root = dataset_root / "val"
    val_ds = None
    if val_root.exists():
        val_ds = ImageFolderDataset(
            val_root, class_to_idx, transform=transform, image_size=image_size
        )

    test_root = dataset_root / "test"
    test_ds = None
    if test_root.exists():
        test_ds = ImageFolderDataset(
            test_root, class_to_idx, transform=transform, image_size=image_size
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader, idx_to_class


class SequenceFromFramesDataset(Dataset):
    """
    Builds sequences from per-class image folders.
    Assumes dataset split folder contains subfolders per class with images.
    Creates non-overlapping sequences of length T within each class folder.
    """

    def __init__(
        self,
        root_dir: str | Path,
        sequence_length: int = 12,
        image_size: Tuple[int, int] = (128, 128),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size

        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples: List[Tuple[List[Path], int]] = []
        self._scan_sequences()

    def _scan_sequences(self) -> None:
        T = self.sequence_length
        for class_name, idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            imgs = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                imgs.extend(sorted(class_dir.glob(ext)))
            # Build non-overlapping sequences
            for i in range(0, max(0, len(imgs) - T + 1), T):
                seq_paths = imgs[i : i + T]
                if len(seq_paths) == T:
                    self.samples.append((seq_paths, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        seq_paths, label = self.samples[index]
        frames: List[torch.Tensor] = []
        for p in seq_paths:
            img = cv2.imread(str(p))
            if img is None:
                raise RuntimeError(f"Failed to read image: {p}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            img = img.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(img).permute(2, 0, 1))  # C,H,W
        x = torch.stack(frames, dim=0)  # T,C,H,W
        y = torch.tensor(label, dtype=torch.long)
        return x, y

