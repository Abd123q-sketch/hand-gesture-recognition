"""
Webcam-based data capture utilities.
Captures hand gesture images into a folder structure like:

dataset_root/
  raw/
    open_palm/
    fist/
    ...
"""
from pathlib import Path
from typing import List
import cv2
from hand_gesture_recognition.utils.config import DataCaptureConfig
from hand_gesture_recognition.utils.logger import get_logger


logger = get_logger(__name__)


def capture_gesture_class(
    gesture_name: str,
    output_dir: str | Path,
    config: DataCaptureConfig,
) -> None:
    """
    Capture a set of images for a single gesture class.
    Press 'c' to capture a frame, 'q' to quit early.
    """
    output_dir = Path(output_dir) / gesture_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(config.webcam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    logger.info(
        "Capturing gesture '%s' into %s (target=%d samples)",
        gesture_name,
        output_dir,
        config.samples_per_class,
    )

    count = 0
    while count < config.samples_per_class:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from webcam.")
            break

        # Optionally flip for mirror effect
        if config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Gesture: {gesture_name} ({count}/{config.samples_per_class})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            "Press 'c' to capture, 'q' to quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Capture Gesture", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            logger.info("Early quit requested.")
            break
        if key == ord("c"):
            resized = cv2.resize(frame, config.image_size)
            filename = output_dir / f"{gesture_name}_{count:04d}.jpg"
            cv2.imwrite(str(filename), resized)
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Finished capturing gesture '%s' (%d samples).", gesture_name, count)


def capture_dataset_for_gestures(
    dataset_root: str | Path,
    gesture_names: List[str],
    config: DataCaptureConfig,
) -> None:
    """
    Capture images for multiple gesture classes.
    """
    dataset_root = Path(dataset_root)
    raw_dir = dataset_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for gesture in gesture_names:
        capture_gesture_class(gesture, raw_dir, config)


