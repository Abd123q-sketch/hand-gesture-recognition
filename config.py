"""
Configuration file for Hand Gesture Recognition Project
"""
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
LANDMARKS_DATA_DIR = os.path.join(DATA_DIR, "landmarks")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  LANDMARKS_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Gesture classes
GESTURE_CLASSES = [
    "open_palm",      # Paume ouverte
    "closed_fist",    # Poing fermé
    "victory",        # Signe de victoire (peace)
    "thumbs_up",      # Pouce levé
    "okay",           # Signe OK
    "pointing",       # Pointage avec index
]

# Data collection settings
COLLECTION_SETTINGS = {
    "samples_per_class": 200,  # Nombre d'échantillons par classe
    "image_size": (224, 224),  # Taille des images capturées
    "webcam_index": 0,         # Index de la webcam
}

# Preprocessing settings
PREPROCESSING_SETTINGS = {
    "image_size": 32,          # Réduit de 64 à 32 pour accélérer (4x moins de pixels)
    "use_landmarks": False,    # Utiliser MediaPipe landmarks (False = utiliser images pour CNN+LSTM)
    "landmark_dim": 21 * 3,    # 21 points * 3 coordonnées (x, y, z)
    "normalize": True,         # Normaliser les valeurs entre 0 et 1
}

# Model settings - OPTIMISÉ POUR ENTRAÎNEMENT RAPIDE (< 1 minute)
MODEL_SETTINGS = {
    "sequence_length": 8,      # Réduit de 16 à 8 (2x plus rapide)
    "batch_size": 256,         # Augmenté de 64 à 256 (4x moins d'itérations)
    "epochs": 2,               # Réduit de 25 à 2 (12x plus rapide)
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "test_split": 0.1,
    "data_subsample": 0.3,     # Utiliser seulement 30% des données pour vitesse
}

# MediaPipe settings
MEDIAPIPE_SETTINGS = {
    "max_num_hands": 1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
}

# Inference settings
INFERENCE_SETTINGS = {
    "confidence_threshold": 0.6,  # Seuil de confiance minimum (augmenté pour plus de précision)
    "smoothing_window": 7,        # Nombre de frames pour lissage (augmenté pour plus de stabilité)
}

