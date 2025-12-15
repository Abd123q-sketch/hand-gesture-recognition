"""
Modèles de deep learning pour la reconnaissance de gestes de la main
Inclut CNN simple et CNN+LSTM pour la dimension temporelle
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_cnn_model(input_shape, num_classes):
    """
    Construit un modèle CNN simple pour la classification d'images
    
    Args:
        input_shape: Shape de l'input (height, width, channels)
        num_classes: Nombre de classes à classifier
        
    Returns:
        model: Modèle Keras compilé
    """
    model = keras.Sequential([
        # Première couche de convolution
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Deuxième couche de convolution
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Troisième couche de convolution
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Quatrième couche de convolution
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten et couches fully connected
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        
        # Couche de sortie
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.MODEL_SETTINGS["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_cnn_lstm_model(input_shape, num_classes, sequence_length=None):
    """
    Construit un modèle CNN+LSTM pour la classification de séquences temporelles
    
    Args:
        input_shape: Shape d'un frame (height, width, channels)
        num_classes: Nombre de classes à classifier
        sequence_length: Longueur des séquences (optionnel, depuis config)
        
    Returns:
        model: Modèle Keras compilé
    """
    if sequence_length is None:
        sequence_length = config.MODEL_SETTINGS["sequence_length"]
    
    # Input: séquence de frames
    input_layer = layers.Input(shape=(sequence_length,) + input_shape)
    
    # CNN équilibré pour bonne accuracy
    cnn = keras.Sequential([
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                              input_shape=(sequence_length,) + input_shape),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.BatchNormalization()),
        
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.BatchNormalization()),
        
        layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        
        layers.TimeDistributed(layers.Flatten()),
        layers.TimeDistributed(layers.Dense(64, activation='relu')),
        layers.TimeDistributed(layers.Dropout(0.3)),
    ])
    
    # Appliquer le CNN sur la séquence
    cnn_output = cnn(input_layer)
    
    # LSTM pour modélisation temporelle
    lstm_out = layers.LSTM(64, return_sequences=True, dropout=0.3)(cnn_output)
    lstm_out = layers.LSTM(32, dropout=0.3)(lstm_out)
    
    # Couches fully connected
    dense = layers.Dense(64, activation='relu')(lstm_out)
    dense = layers.Dropout(0.4)(dense)
    dense = layers.Dense(32, activation='relu')(dense)
    dense = layers.Dropout(0.3)(dense)
    
    # Couche de sortie
    output = layers.Dense(num_classes, activation='softmax')(dense)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.MODEL_SETTINGS["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_landmark_model(input_dim, num_classes):
    """
    Construit un modèle pour les landmarks MediaPipe (pas d'images)
    
    Args:
        input_dim: Dimension de l'input (nombre de features)
        num_classes: Nombre de classes
        
    Returns:
        model: Modèle Keras compilé
    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.MODEL_SETTINGS["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_landmark_lstm_model(input_dim, num_classes, sequence_length=None):
    """
    Construit un modèle LSTM pour les séquences de landmarks
    
    Args:
        input_dim: Dimension d'un frame de landmarks
        num_classes: Nombre de classes
        sequence_length: Longueur des séquences
        
    Returns:
        model: Modèle Keras compilé
    """
    if sequence_length is None:
        sequence_length = config.MODEL_SETTINGS["sequence_length"]
    
    model = keras.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, input_dim)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        
        layers.LSTM(32),
        layers.Dropout(0.2),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.MODEL_SETTINGS["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model(model_type, input_shape_or_dim, num_classes, **kwargs):
    """
    Factory function pour obtenir un modèle selon le type
    
    Args:
        model_type: Type de modèle ('cnn', 'cnn_lstm', 'landmark', 'landmark_lstm')
        input_shape_or_dim: Shape pour CNN ou dimension pour landmarks
        num_classes: Nombre de classes
        **kwargs: Arguments additionnels
        
    Returns:
        model: Modèle Keras compilé
    """
    if model_type == 'cnn':
        return build_cnn_model(input_shape_or_dim, num_classes)
    elif model_type == 'cnn_lstm':
        sequence_length = kwargs.get('sequence_length', None)
        return build_cnn_lstm_model(input_shape_or_dim, num_classes, sequence_length)
    elif model_type == 'landmark':
        return build_landmark_model(input_shape_or_dim, num_classes)
    elif model_type == 'landmark_lstm':
        sequence_length = kwargs.get('sequence_length', None)
        return build_landmark_lstm_model(input_shape_or_dim, num_classes, sequence_length)
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")


if __name__ == "__main__":
    # Test des modèles
    print("Test des modèles...")
    
    # Test CNN
    print("\n1. Test CNN")
    cnn_model = build_cnn_model((64, 64, 3), 6)
    cnn_model.summary()
    
    # Test CNN+LSTM
    print("\n2. Test CNN+LSTM")
    cnn_lstm_model = build_cnn_lstm_model((64, 64, 3), 6)
    cnn_lstm_model.summary()
    
    # Test Landmark
    print("\n3. Test Landmark")
    landmark_model = build_landmark_model(63, 6)  # 21 points * 3 coordonnées
    landmark_model.summary()
    
    # Test Landmark+LSTM
    print("\n4. Test Landmark+LSTM")
    landmark_lstm_model = build_landmark_lstm_model(63, 6)
    landmark_lstm_model.summary()

