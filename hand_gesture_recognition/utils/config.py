"""
Central configuration for the Hand Gesture Recognition project.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "saved_models"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in (DATASET_ROOT, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataCaptureConfig:
    samples_per_class: int = 200
    image_size: Tuple[int, int] = (224, 224)
    webcam_index: int = 0
    flip_horizontal: bool = True


@dataclass
class PreprocessingConfig:
    image_size: int = 64
    normalize: bool = True
    use_full_frame: bool = False


@dataclass
class TrainingConfig:
    # Generic
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cuda"
    save_best_only: bool = True

    # Temporal
    sequence_length: int = 12


@dataclass
class InferenceConfig:
    smoothing_window: int = 7
    confidence_threshold: float = 0.6
    webcam_index: int = 0
    flip_horizontal: bool = True


def default_gestures() -> List[str]:
    """
    Default gesture set. Extend as needed.
    """
    return ["open_palm", "fist", "victory", "thumbs_up"]


