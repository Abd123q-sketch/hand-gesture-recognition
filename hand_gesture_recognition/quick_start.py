"""
Quick-start interactive script (FR) pour utiliser la solution rapidement.

Fonctions principales:
1) Capturer des données via webcam (par classes)
2) Split automatique de dataset/raw -> train/val/test
3) Entraîner CNN (statique)
4) Entraîner CNN+LSTM (séquences)
5) Évaluer un modèle
6) Inference temps réel

Usage:
  python -m hand_gesture_recognition.quick_start
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
import shutil

"""Allow running this file directly (python quick_start.py) by adding project root to sys.path."""
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_gesture_recognition.utils.config import DATASET_ROOT
from hand_gesture_recognition.data.data_capture import capture_dataset_for_gestures
from hand_gesture_recognition.utils.config import DataCaptureConfig


def run_module(module: str, args: List[str]) -> int:
    cmd = [sys.executable, "-m", module] + args
    return subprocess.call(cmd)


def input_list(prompt: str) -> List[str]:
    raw = input(prompt).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def split_raw_dataset(raw_dir: Path, out_root: Path, ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> None:
    """Split dataset/raw/<class> vers dataset/{train,val,test}/<class>.
    Les fichiers sont copiés (ou déplacés si vous voulez: remplacez copy par move).
    """
    train_r, val_r, test_r = ratios
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        print("Ratios doivent sommer à 1.0")
        return

    train_dir = out_root / "train"
    val_dir = out_root / "val"
    test_dir = out_root / "test"
    ensure_dirs(train_dir, val_dir, test_dir)

    classes = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not classes:
        print(f"Aucune classe trouvée dans {raw_dir}")
        return

    for cdir in classes:
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images.extend(sorted(cdir.glob(ext)))
        if not images:
            print(f"[Info] Pas d'images pour {cdir.name}")
            continue
        n = len(images)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        n_test = n - n_train - n_val

        splits = (
            (images[:n_train], train_dir / cdir.name),
            (images[n_train:n_train + n_val], val_dir / cdir.name),
            (images[n_train + n_val:], test_dir / cdir.name),
        )
        for files, dst in splits:
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dst / f.name)
        print(f"Classe {cdir.name}: {n_train} train, {n_val} val, {n_test} test")


MENU = """
================= Quick Start =================
1) Capturer des données (webcam)
2) Split dataset/raw -> train/val/test
3) Entraîner CNN (statique)
4) Entraîner CNN+LSTM (séquences)
5) Évaluer un modèle
6) Inference temps réel
0) Quitter
==============================================
Choix: """


def main():
    while True:
        try:
            choice = input(MENU).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if choice == "0":
            print("Bye.")
            return

        elif choice == "1":
            print("\n-- Capture données --")
            gestures = input_list("Entrez la liste des gestes (séparés par des virgules) [ex: open_palm,fist,victory]: ")
            if not gestures:
                gestures = ["open_palm", "fist", "victory", "thumbs_up"]
            samples = input("Samples par classe [200]: ").strip() or "200"
            webcam = input("Index webcam [0]: ").strip() or "0"
            flip = input("Flip horizontal ? (y/n) [y]: ").strip().lower() or "y"

            cfg = DataCaptureConfig(
                samples_per_class=int(samples),
                webcam_index=int(webcam),
                flip_horizontal=(flip == "y"),
            )
            raw = DATASET_ROOT / "raw"
            capture_dataset_for_gestures(DATASET_ROOT, gestures, cfg)
            print(f"OK. Images capturées dans {raw}\n")

        elif choice == "2":
            print("\n-- Split dataset/raw -> train/val/test --")
            raw = DATASET_ROOT / "raw"
            if not raw.exists():
                print(f"{raw} introuvable. Lancez d'abord la capture (1).\n")
                continue
            ratios_str = input("Ratios train,val,test [0.7,0.15,0.15]: ").strip() or "0.7,0.15,0.15"
            try:
                tr, vr, te = [float(x) for x in ratios_str.split(",")]
            except Exception:
                print("Ratios invalides. Ex: 0.7,0.15,0.15\n")
                continue
            split_raw_dataset(raw, DATASET_ROOT, (tr, vr, te))
            print("Split terminé.\n")

        elif choice == "3":
            print("\n-- Entraîner CNN (statique) --")
            image_size = input("Taille image [64]: ").strip() or "64"
            epochs = input("Epochs [10]: ").strip() or "10"
            bs = input("Batch size [64]: ").strip() or "64"
            args = [
                "--dataset_root", str(DATASET_ROOT),
                "--model", "cnn",
                "--image_size", image_size,
                "--epochs", epochs,
                "--batch_size", bs,
            ]
            run_module("hand_gesture_recognition.training.train", args)
            print("Entraînement CNN terminé.\n")

        elif choice == "4":
            print("\n-- Entraîner CNN+LSTM (séquences) --")
            image_size = input("Taille image [64]: ").strip() or "64"
            seq_len = input("Longueur séquence [12]: ").strip() or "12"
            epochs = input("Epochs [10]: ").strip() or "10"
            bs = input("Batch size [16]: ").strip() or "16"
            args = [
                "--dataset_root", str(DATASET_ROOT),
                "--model", "cnn_lstm",
                "--image_size", image_size,
                "--sequence_length", seq_len,
                "--epochs", epochs,
                "--batch_size", bs,
            ]
            run_module("hand_gesture_recognition.training.train", args)
            print("Entraînement CNN+LSTM terminé.\n")

        elif choice == "5":
            print("\n-- Évaluer un modèle --")
            model_type = input("Type [cnn|cnn_lstm] [cnn]: ").strip() or "cnn"
            weights = input("Chemin des poids (ex: saved_models/cnn_best.pt): ").strip()
            if not weights:
                print("Veuillez fournir un chemin de poids.\n")
                continue
            args = [
                "--dataset_root", str(DATASET_ROOT),
                "--model_type", model_type,
                "--weights", weights,
            ]
            run_module("hand_gesture_recognition.training.evaluate", args)
            print("Évaluation terminée.\n")

        elif choice == "6":
            print("\n-- Inference temps réel --")
            model_type = input("Type [cnn|cnn_lstm] [cnn]: ").strip() or "cnn"
            weights = input("Chemin des poids (ex: saved_models/cnn_best.pt): ").strip()
            if not weights:
                print("Veuillez fournir un chemin de poids.\n")
                continue
            img_size = input("Taille image [64]: ").strip() or "64"
            seq_len = input("Longueur séquence (si cnn_lstm) [12]: ").strip() or "12"
            args = [
                "--weights", weights,
                "--dataset_root", str(DATASET_ROOT),
                "--model", model_type,
                "--image_size", img_size,
            ]
            if model_type == "cnn_lstm":
                args += ["--sequence_length", seq_len]
            args += ["--flip"]
            run_module("hand_gesture_recognition.inference.realtime_inference", args)
            print("Inference terminée.\n")

        else:
            print("Choix invalide. Réessayez.\n")


if __name__ == "__main__":
    main()
