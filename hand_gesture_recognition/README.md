# Hand Gesture Recognition using Deep Learning and Computer Vision

## Project Overview
A modular, clean-architecture project to recognize hand gestures for HCI. It supports static image classification (CNN) and dynamic sequence recognition (CNN+LSTM), webcam dataset capture, preprocessing, full training/evaluation pipeline, and real-time inference.

## System Architecture
- **data/**: dataset loading, webcam capture, preprocessing utilities
- **models/**: `SimpleCNN` for static, `CNNLSTM` for sequences
- **training/**: train/evaluate scripts with checkpoints and GPU support
- **inference/**: real-time webcam inference with smoothing
- **utils/**: centralized config and logging
- **notebooks/**: experimentation space

ASCII diagram:

```
main.py
  ├─ training/train.py ──> models/{cnn,cnn_lstm}.py
  │        │             └─ data/dataset_loader.py
  │        └─ utils/{config,logger}.py
  ├─ training/evaluate.py
  └─ inference/realtime_inference.py
```

## Dataset Creation & Annotation
- Use `data/data_capture.py` to capture custom gestures via webcam.
- Folder structure (example):
```
dataset/
  train/
    open_palm/
    fist/
    victory/
  val/
  test/
```
- Images are automatically labeled by folder name. For sequences, frames are grouped into fixed-length segments per class.

Commands:
- Capture: `python -m hand_gesture_recognition.data.data_capture`
- Or import and call `capture_dataset_for_gestures(dataset_root, [classes], DataCaptureConfig())`.

## Preprocessing Pipeline
- Simple skin-color based ROI or full-frame fallback.
- Resize to square `image_size` and optional normalization.
- Sequence builder (sliding window) for temporal models.

## Model Architectures
- **CNN (Static)**: 3 conv blocks + BatchNorm + ReLU + MaxPool, AdaptiveAvgPool, Dropout, Linear to softmax logits.
- **CNN+LSTM (Dynamic)**: CNN backbone to per-frame features, LSTM over time, MLP classifier.

## Training Process
- CrossEntropy loss, Adam optimizer, configurable LR/weight decay.
- Tracks train/val loss and accuracy each epoch.
- Saves best model by validation accuracy and final checkpoint.

Example:
```
python -m hand_gesture_recognition.training.train \
  --dataset_root dataset \
  --model cnn \
  --image_size 64 \
  --epochs 15 \
  --batch_size 64
```
For sequences:
```
python -m hand_gesture_recognition.training.train \
  --dataset_root dataset \
  --model cnn_lstm \
  --image_size 64 \
  --sequence_length 12
```

## Real-Time Inference Workflow
- Grabs webcam frames, preprocesses, runs model, applies majority-vote smoothing over last N predictions, displays label and confidence.

Example:
```
python -m hand_gesture_recognition.inference.realtime_inference \
  --weights saved_models/cnn_best.pt \
  --dataset_root dataset \
  --model cnn --image_size 64 --smoothing 7 --conf 0.6 --flip
```

## How to Run the Project
1. Install dependencies: `pip install -r requirements.txt` (root or subproject).
2. Capture dataset (optional) or download static dataset.
3. Organize folders as specified.
4. Train a model (CNN or CNN+LSTM).
5. Evaluate with `training/evaluate.py`.
6. Run real-time inference.


## Configuration
All paths and hyperparameters are centralized in `utils/config.py`. Avoid hard-coded paths; pass overrides via CLI args.

