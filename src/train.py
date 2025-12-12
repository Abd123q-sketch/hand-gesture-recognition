"""
Script d'entraînement pour les modèles de reconnaissance de gestes
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import os
import sys
from datetime import datetime
from tensorflow.keras import mixed_precision  # Mixed precision pour accélérer sur GPU

# Ajouter le répertoire parent au path pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.models import get_model
from src.preprocessing import HandPreprocessor

# Activer le mixed precision (gain de perf sur GPU, neutre sur CPU)
mixed_precision.set_global_policy("mixed_float16")

def load_data(data_dir=None, use_landmarks=None):
    """
    Charge les données d'entraînement
    
    Args:
        data_dir: Répertoire contenant les données
        use_landmarks: Utiliser les landmarks (None = auto-détecté)
        
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_mapping
    """
    if data_dir is None:
        # Auto-détecter le type de données
        landmarks_dir = Path(config.LANDMARKS_DATA_DIR)
        processed_dir = Path(config.PROCESSED_DATA_DIR)
        
        # Fallback vers dataset/ si les données n'existent pas dans data/
        dataset_landmarks_dir = Path(config.PROJECT_ROOT) / "dataset" / "landmarks"
        dataset_dir = Path(config.PROJECT_ROOT) / "dataset"
        
        if use_landmarks is None:
            # Vérifier d'abord dans data/, puis dans dataset/
            use_landmarks = (landmarks_dir.exists() and (landmarks_dir / "X_train.npy").exists()) or \
                           (dataset_landmarks_dir.exists() and (dataset_landmarks_dir / "X_train.npy").exists())
        
        # Choisir le répertoire avec fallback
        if use_landmarks:
            if landmarks_dir.exists() and (landmarks_dir / "X_train.npy").exists():
                data_dir = landmarks_dir
            elif dataset_landmarks_dir.exists() and (dataset_landmarks_dir / "X_train.npy").exists():
                data_dir = dataset_landmarks_dir
                print("⚠️  Utilisation des donnees depuis dataset/landmarks (fallback)")
            else:
                raise FileNotFoundError("Aucune donnee de landmarks trouvee dans data/landmarks ou dataset/landmarks")
        else:
            if processed_dir.exists() and (processed_dir / "X_train.npy").exists():
                data_dir = processed_dir
            elif dataset_dir.exists() and (dataset_dir / "X_train.npy").exists():
                data_dir = dataset_dir
                print("⚠️  Utilisation des donnees depuis dataset/ (fallback)")
            else:
                # Vérifier si des landmarks existent comme alternative
                if landmarks_dir.exists() and (landmarks_dir / "X_train.npy").exists():
                    raise FileNotFoundError(
                        f"❌ Aucune donnee d'images trouvee dans {processed_dir} ou {dataset_dir}.\n"
                        f"✅ Mais des landmarks existent dans {landmarks_dir}!\n\n"
                        f"💡 Solution: Utilisez les landmarks avec:\n"
                        f"   python src/train.py --model landmark --lstm\n\n"
                        f"Ou creez des donnees d'images avec:\n"
                        f"   python src/preprocessing.py"
                    )
                else:
                    raise FileNotFoundError(
                        f"❌ Aucune donnee trouvee!\n\n"
                        f"📝 Pour creer des donnees:\n"
                        f"   1. Collectez des images: python src/data_collection.py\n"
                        f"   2. Pretraitez-les: python src/preprocessing.py\n\n"
                        f"Ou utilisez les landmarks existants avec:\n"
                        f"   python src/train.py --model landmark"
                    )
    else:
        data_dir = Path(data_dir)
    
    print(f"Chargement des donnees depuis: {data_dir}")
    
    # Vérifier que les fichiers existent
    required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", 
                     "X_test.npy", "y_test.npy", "class_mapping.json"]
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing_files:
        raise FileNotFoundError(
            f"Fichiers manquants dans {data_dir}:\n" + 
            "\n".join(f"  - {f}" for f in missing_files) +
            f"\n\nExecutez d'abord: python src/preprocessing.py"
        )
    
    # Charger les données
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    # Charger le mapping des classes
    mapping_file = data_dir / "class_mapping.json"
    if not mapping_file.exists():
        # Créer un mapping par défaut basé sur les classes dans config
        print("⚠️  class_mapping.json non trouve, creation d'un mapping par defaut")
        class_mapping = {str(i): name for i, name in enumerate(config.GESTURE_CLASSES)}
        with open(mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        print(f"  Mapping cree: {class_mapping}")
    else:
        with open(mapping_file, 'r') as f:
            class_mapping = json.load(f)
    
    print(f"Donnees chargees:")
    print(f"  Train: {X_train.shape}, Labels: {y_train.shape}")
    print(f"  Val: {X_val.shape}, Labels: {y_val.shape}")
    print(f"  Test: {X_test.shape}, Labels: {y_test.shape}")
    print(f"  Classes: {class_mapping}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_mapping


def prepare_sequences(X_train, y_train, X_val, y_val, X_test, y_test, sequence_length):
    """
    Prépare les données sous forme de séquences pour les modèles LSTM
    """
    from src.preprocessing import HandPreprocessor
    
    preprocessor = HandPreprocessor()
    
    print("Creation des sequences temporelles...")
    
    X_train_seq, y_train_seq = preprocessor.create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = preprocessor.create_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = preprocessor.create_sequences(X_test, y_test, sequence_length)
    
    print(f"Sequences creees:")
    print(f"  Train: {X_train_seq.shape}")
    print(f"  Val: {X_val_seq.shape}")
    print(f"  Test: {X_test_seq.shape}")
    
    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq)


def train_model():
    """
    Entraîne un modèle CNN+LSTM pour la reconnaissance de gestes
    Utilise uniquement des images (pas de landmarks)
    """
    # Forcer l'utilisation des images (pas de landmarks)
    use_landmarks = False
    
    # Charger les données d'images
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_mapping = load_data(
        use_landmarks=use_landmarks
    )
    
    num_classes = len(class_mapping)
    
    # Vérifier que ce sont bien des images (4D: samples, height, width, channels)
    if len(X_train.shape) != 4:
        raise ValueError(
            f"❌ Les donnees doivent etre des images (4D), mais shape={X_train.shape}.\n"
            f"   Les landmarks ne sont pas supportes. Utilisez des images pretraitees.\n"
            f"   Executez: python src/preprocessing.py (avec use_landmarks=False dans config.py)"
        )
    
    # Préparer les données pour LSTM (toujours utilisé)
    sequence_length = config.MODEL_SETTINGS["sequence_length"]
    print(f"\nCreation des sequences temporelles (longueur: {sequence_length})...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_sequences(
        X_train, y_train, X_val, y_val, X_test, y_test, sequence_length
    )
    
    # Déterminer l'input shape pour CNN+LSTM
    input_shape = X_train.shape[2:]  # (height, width, channels) - sans la dimension séquence
    model_name = 'cnn_lstm'
    
    # Construire le modèle CNN+LSTM
    print(f"\nConstruction du modele: {model_name}")
    model = get_model('cnn_lstm', input_shape, num_classes,
                     sequence_length=sequence_length)
    
    model.summary()

    # Calculer les poids de classe pour compenser le déséquilibre
    from collections import Counter
    class_counts = Counter(y_train.flatten())
    total_samples = len(y_train)
    
    print(f"\nDistribution des classes dans le dataset d'entraînement:")
    class_weights = {}
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        weight = total_samples / (num_classes * count) if count > 0 else 1.0
        class_weights[class_id] = weight
        class_name = list(class_mapping.values())[class_id] if isinstance(class_mapping, dict) else f"Class_{class_id}"
        print(f"  {class_name:20s}: {count:4d} echantillons, poids: {weight:.3f}")
    
    # Normaliser les poids pour qu'ils soient autour de 1.0
    avg_weight = sum(class_weights.values()) / len(class_weights)
    class_weights = {k: v / avg_weight for k, v in class_weights.items()}
    
    print(f"\nUtilisation de class_weights pour compenser le desequilibre")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(config.MODELS_DIR) / f"{model_name}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(Path(config.LOGS_DIR) / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            histogram_freq=1
        )
    ]

    # Entraînement
    print("\n" + "="*60)
    print("DEBUT DE L'ENTRAINEMENT")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        batch_size=config.MODEL_SETTINGS["batch_size"],
        epochs=config.MODEL_SETTINGS["epochs"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,  # Utiliser les poids de classe pour compenser le déséquilibre
        verbose=1
    )
    
    # Évaluation sur le test set
    print("\n" + "="*60)
    print("EVALUATION SUR LE TEST SET")
    print("="*60)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Sauvegarder le modèle final
    model_path = Path(config.MODELS_DIR) / f"{model_name}_final.h5"
    model.save(str(model_path))
    print(f"\nModele sauvegarde: {model_path}")
    
    # Sauvegarder les métriques
    metrics = {
        'model_type': model_name,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'num_classes': num_classes,
        'class_mapping': class_mapping,
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    metrics_path = Path(config.MODELS_DIR) / f"{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metriques sauvegardees: {metrics_path}")
    
    return model, history, metrics

def main():
    """Fonction principale - Entraîne un modèle CNN+LSTM"""
    print("="*60)
    print("ENTRAINEMENT DU MODELE CNN+LSTM")
    print("="*60)
    print("Modele: CNN + LSTM (dimension temporelle)")
    print("Type de donnees: Images pretraitees")
    print("="*60 + "\n")
    
    train_model()


if __name__ == "__main__":
    main()
