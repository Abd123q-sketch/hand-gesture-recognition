# Hand Gesture Recognition

A comprehensive PyTorch-based hand gesture recognition system supporting both static and dynamic gesture recognition through webcam-based real-time inference.

## Project Overview

This project implements a complete hand gesture recognition pipeline with the following capabilities:

- **Static Gesture Recognition**: Single-frame classification using CNN
- **Dynamic Gesture Recognition**: Sequence-based classification using CNN+LSTM
- **Real-time Inference**: Webcam-based live prediction with smoothing
- **Data Collection**: Built-in webcam capture for dataset creation
- **Modular Architecture**: Clean separation of concerns with reusable components

**Use Cases**:
- Human-Computer Interaction (HCI) applications
- Sign language recognition foundations
- Gesture-based control systems
- Computer vision research and education

## System Architecture

The system follows a modular architecture with clear separation between data processing, model training, and inference components:

```
Data Collection → Dataset Organization → Model Training → Real-time Inference
     ↓                ↓                    ↓                  ↓
Webcam Capture   Train/Val/Test Split   CNN/CNN-LSTM      Live Prediction
```

**Component Flow**:
1. **Data Module**: Handles capture, loading, and preprocessing
2. **Models Module**: Contains CNN and CNN-LSTM architectures
3. **Training Module**: Manages training loops and evaluation
4. **Inference Module**: Real-time prediction with smoothing
5. **Utils Module**: Configuration, logging, and utilities

## Project Structure

```
hand_gesture_recognition/
│
├── data/
│   ├── __init__.py
│   ├── data_capture.py          # Webcam-based dataset collection
│   └── dataset_loader.py        # Dataset loading and preprocessing
│
├── models/
│   ├── __init__.py
│   ├── cnn_model.py            # SimpleCNN for static gestures
│   └── cnn_lstm_model.py       # CNN+LSTM for dynamic gestures
│
├── training/
│   ├── __init__.py
│   ├── train.py                # Training pipeline
│   └── evaluate.py             # Model evaluation
│
├── inference/
│   ├── __init__.py
│   └── realtime_inference.py   # Real-time webcam inference
│
├── utils/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration
│   ├── logger.py               # Logging utilities
│   └── [additional utilities]  # Dataset management tools
│
├── main.py                     # CLI entry point
├── quick_start.py             # Interactive quick-start script
└── requirements.txt           # Dependencies
```

**Module Responsibilities**:
- **data/**: Dataset creation, loading, and preprocessing pipelines
- **models/**: Neural network architectures for different recognition tasks
- **training/**: Model training, validation, and evaluation workflows
- **inference/**: Real-time gesture recognition with webcam integration
- **utils/**: Configuration management, logging, and helper utilities

## Dataset & Data Collection

### Data Collection Process

The project includes a built-in webcam-based data collection system:

1. **Interactive Capture**: Use `quick_start.py` for guided data collection
2. **Manual Capture**: Direct use of `data_capture.py` for programmatic collection
3. **Class Organization**: Images automatically organized by gesture class
4. **Real-time Preview**: Live webcam feed with capture controls

**Dataset Structure**:
```
dataset/
├── raw/                    # Captured images (pre-split)
│   ├── fist/
│   ├── like/
│   └── okay/
├── train/                  # Training split
├── val/                    # Validation split
└── test/                   # Test split
```

### Supported Gesture Classes

Default gesture set includes:
- `fist` - Closed fist gesture
- `like` - Thumbs-up gesture  
- `okay` - OK hand gesture

Additional gestures can be added by creating corresponding folders during data collection.

### Dataset Splitting

The `quick_start.py` script provides automatic dataset splitting:
- **Default ratios**: 70% train, 15% validation, 15% test
- **Customizable ratios**: Configurable via interactive prompt
- **Non-overlapping splits**: Ensures no data leakage between sets

## Preprocessing Pipeline

### Image Preprocessing

1. **Color Space Conversion**: BGR → RGB for model compatibility
2. **Resizing**: Images resized to configurable dimensions (default: 64x64)
3. **Normalization**: Pixel values scaled to [0, 1] range
4. **Tensor Conversion**: HWC → CHW format for PyTorch compatibility

### Sequence Generation (CNN-LSTM)

For dynamic gesture recognition:
1. **Frame Sequencing**: Consecutive frames grouped into sequences
2. **Non-overlapping Windows**: Sequences created without overlap
3. **Fixed Length**: Configurable sequence length (default: 12 frames)
4. **Temporal Ordering**: Maintains temporal sequence integrity

### Data Augmentation

Currently supports basic preprocessing. Augmentation can be extended via custom transform functions in the dataset loader.

## Deep Learning Models

### SimpleCNN Architecture

**Designed for static gesture recognition from single frames.**

**Architecture Details**:
- **Input**: 3-channel RGB images (configurable size)
- **Convolutional Blocks**: 3 blocks with progressive feature extraction
  - Block 1: 3 → 32 channels
  - Block 2: 32 → 64 channels  
  - Block 3: 64 → 128 channels
  - Each block: Conv2D → BatchNorm → ReLU → MaxPool(2)
- **Adaptive Pooling**: AdaptiveAvgPool2d((4,4)) for spatial invariance
- **Classifier**: Fully connected layers with dropout
  - Linear(128×4×4 → 256 → num_classes)
  - Dropout(0.5) for regularization

**Use Case**: Best for static poses where single-frame information is sufficient.

### CNN-LSTM Architecture

**Designed for dynamic gesture recognition from frame sequences.**

**Architecture Components**:
- **CNN Backbone**: Same feature extractor as SimpleCNN
  - Outputs 256-dimensional feature vectors per frame
- **LSTM Layer**: Temporal modeling of feature sequences
  - Hidden dimension: 256 (configurable)
  - Bidirectional: True for enhanced temporal context
  - Layers: 2 (configurable)
- **Classifier**: Temporal feature integration and classification
  - Linear(hidden_dim×2 → 256 → num_classes)
  - Dropout(0.5) regularization

**Input Shape**: (Batch, Time, Channels, Height, Width)
**Output**: Class logits based on temporal sequence understanding

**Use Case**: Ideal for dynamic gestures requiring temporal context and motion patterns.

## Training Workflow

### Training Pipeline

The training system supports both model types with a unified interface:

**Training Loop**:
1. **Data Loading**: Automatic train/val/test DataLoader creation
2. **Model Initialization**: Dynamic model instantiation based on type
3. **Optimization**: Adam optimizer with configurable learning rate
4. **Loss Computation**: CrossEntropyLoss for multi-class classification
5. **Metrics Tracking**: Training and validation accuracy per epoch
6. **Model Saving**: Best model based on validation accuracy

**Key Features**:
- **Automatic GPU Detection**: Uses CUDA if available, falls back to CPU
- **Flexible Configuration**: All hyperparameters configurable via CLI
- **Early Stopping**: Saves best model based on validation performance
- **Comprehensive Logging**: Structured logging with progress tracking

### Training Metrics

**Monitored Metrics**:
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch
- Final test set evaluation
- Model checkpointing with best validation accuracy

**Model Saving Strategy**:
- **Best Model**: `{model_type}_best.pt` (highest validation accuracy)
- **Final Model**: `{model_type}_final.pt` (last epoch)
- **Checkpoint Format**: PyTorch state_dict with epoch information

## Real-Time Inference Pipeline

### Inference Architecture

The inference system provides real-time gesture recognition with the following features:

**Processing Pipeline**:
1. **Webcam Capture**: OpenCV-based video stream acquisition
2. **Frame Preprocessing**: Real-time resizing and normalization
3. **Model Prediction**: Batch inference with GPU acceleration
4. **Prediction Smoothing**: Majority voting over temporal window
5. **Visualization**: Real-time overlay with confidence scores

### Prediction Smoothing

**Stability Enhancement**:
- **Sliding Window**: Configurable smoothing window (default: 7 frames)
- **Majority Voting**: Most frequent prediction in window
- **Confidence Thresholding**: Minimum confidence for valid predictions
- **Temporal Consistency**: Reduces flickering and false positives

### Display Features

**Real-time Overlay**:
- **Predicted Label**: Current gesture prediction
- **Confidence Score**: Model confidence percentage
- **Color Coding**: Green for detected, red for no detection
- **Interactive Controls**: Press 'q' to quit inference

## Configuration & CLI Usage

### Configuration System

All hyperparameters are centralized in `utils/config.py`:

**Data Capture Config**:
```python
samples_per_class: int = 200
image_size: Tuple[int, int] = (224, 224)
webcam_index: int = 0
flip_horizontal: bool = True
```

**Training Config**:
```python
batch_size: int = 64
num_epochs: int = 20
learning_rate: float = 1e-3
sequence_length: int = 12  # For CNN-LSTM
```

**Inference Config**:
```python
smoothing_window: int = 7
confidence_threshold: float = 0.6
webcam_index: int = 0
```

### CLI Commands

**Main Entry Point** (`main.py`):

```bash
# Training
python -m hand_gesture_recognition.main train --extra --model cnn --epochs 20 --batch_size 64
python -m hand_gesture_recognition.main train --extra --model cnn_lstm --sequence_length 12

# Evaluation
python -m hand_gesture_recognition.main evaluate --extra --weights saved_models/cnn_best.pt

# Real-time Inference
python -m hand_gesture_recognition.main realtime --extra --weights saved_models/cnn_best.pt --model cnn
```

**Quick Start Interface** (`quick_start.py`):
```bash
python -m hand_gesture_recognition.quick_start
```

Interactive menu for:
1. Data capture via webcam
2. Dataset splitting
3. Model training (CNN/CNN-LSTM)
4. Model evaluation
5. Real-time inference

**Direct Module Execution**:

```bash
# Training modules
python -m hand_gesture_recognition.training.train --model cnn --epochs 20
python -m hand_gesture_recognition.training.train --model cnn_lstm --sequence_length 12

# Evaluation
python -m hand_gesture_recognition.training.evaluate --weights saved_models/cnn_best.pt

# Real-time inference
python -m hand_gesture_recognition.inference.realtime_inference --weights saved_models/cnn_best.pt --model cnn
```

## Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Hardware**: CPU required, GPU (CUDA) recommended for training

### Setup Instructions

1. **Clone Repository**:
```bash
git clone <repository-url>
cd mon_projet_deeplearning
```

2. **Create Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify Installation**:
```bash
python -m hand_gesture_recognition.main --help
```

**Note**: The requirements.txt includes TensorFlow, but the project primarily uses PyTorch. TensorFlow may be included for potential future extensions or compatibility.

### Hardware Requirements

**Minimum Requirements**:
- CPU: Any modern processor
- RAM: 4GB minimum, 8GB recommended
- Storage: 2GB free space for models and dataset

**Recommended Requirements**:
- GPU: NVIDIA GPU with CUDA support
- RAM: 16GB or more
- Storage: SSD for faster data loading

## Performance & Limitations

### Current Limitations

**Model Limitations**:
- **Lightweight Architecture**: Models designed for real-time performance, not maximum accuracy
- **Limited Context**: CNN processes single frames, CNN-LSTM uses short sequences
- **No Hand Segmentation**: Uses full frame rather than hand-only regions
- **Fixed Gesture Set**: Requires retraining for new gesture classes

**Data Limitations**:
- **Lighting Sensitivity**: Performance varies with lighting conditions
- **Background Dependency**: Model may learn background patterns
- **Limited Augmentation**: Basic preprocessing without extensive augmentation
- **Single User**: Models trained on specific user may not generalize well

**Hardware Constraints**:
- **CPU Inference**: Real-time performance may be limited on CPU
- **Memory Usage**: CNN-LSTM requires more memory for sequence processing
- **Webcam Quality**: Performance depends on webcam resolution and framerate

### Performance Expectations

**Training Performance**:
- **CNN**: ~1-5 minutes per epoch on modern GPU
- **CNN-LSTM**: ~2-10 minutes per epoch depending on sequence length
- **Dataset Size**: Performance scales with dataset size and complexity

**Inference Performance**:
- **CNN**: 30+ FPS on GPU, 10-20 FPS on CPU
- **CNN-LSTM**: 20+ FPS on GPU, 5-15 FPS on CPU
- **Latency**: ~50-100ms processing delay

## Possible Improvements

### Model Enhancements

**Advanced Architectures**:
- **MediaPipe Integration**: Use MediaPipe for hand detection and cropping
- **Transformer Models**: Vision transformers for spatial feature extraction
- **Attention Mechanisms**: Self-attention for temporal modeling
- **Ensemble Methods**: Combine multiple models for improved accuracy

**Data Improvements**:
- **Data Augmentation**: Rotation, scaling, color jitter, and synthetic data
- **Multi-user Datasets**: Collect data from multiple users for better generalization
- **Background Variation**: Train with diverse backgrounds for robustness
- **Hand Segmentation**: Pre-process to isolate hand regions

### Performance Optimizations

**Model Optimization**:
- **Model Quantization**: INT8 quantization for faster inference
- **ONNX Export**: Deploy to ONNX Runtime for cross-platform inference
- **TensorRT Integration**: NVIDIA TensorRT for optimized GPU inference
- **Mobile Deployment**: Convert models for mobile/edge deployment

**System Optimizations**:
- **Multi-threading**: Parallel data loading and preprocessing
- **Batch Inference**: Process multiple frames simultaneously
- **Model Caching**: Cache model predictions for repeated gestures
- **Adaptive Inference**: Dynamic sequence length based on motion detection

### Feature Extensions

**Advanced Features**:
- **Gesture Sequences**: Recognize sequences of gestures (e.g., swipe patterns)
- **Continuous Recognition**: Real-time continuous gesture recognition
- **Multi-hand Detection**: Support for multiple simultaneous hands
- **3D Gesture Recognition**: Depth camera integration for 3D gestures

**User Experience**:
- **GUI Interface**: Graphical user interface for easier interaction
- **Calibration System**: User-specific calibration for improved accuracy
- **Gesture Recording**: Record and playback gesture sequences
- **Custom Gesture Training**: Interface for training custom gestures

### Integration Possibilities

**External Integrations**:
- **Game Engines**: Unity/Unreal integration for game controls
- **Web Applications**: WebRTC integration for browser-based recognition
- **Mobile Apps**: React Native or native mobile applications
- **IoT Devices**: Integration with smart home devices and IoT platforms

**Research Extensions**:
- **Sign Language**: Extend to full sign language recognition
- **Emotion Recognition**: Combine gesture with facial expression analysis
- **Activity Recognition**: Full-body activity recognition
- **Healthcare Applications**: Rehabilitation and monitoring systems
