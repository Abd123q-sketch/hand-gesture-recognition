"""
Module de prétraitement pour la reconnaissance de gestes de la main
Utilise OpenCV et MediaPipe pour extraire et traiter les données
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import os
import sys
from tqdm import tqdm

# Ajouter le répertoire parent au path pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class HandPreprocessor:
    """Classe pour prétraiter les images de gestes de la main"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=config.MEDIAPIPE_SETTINGS["max_num_hands"],
            min_detection_confidence=config.MEDIAPIPE_SETTINGS["min_detection_confidence"],
            min_tracking_confidence=config.MEDIAPIPE_SETTINGS["min_tracking_confidence"]
        )
        self.image_size = config.PREPROCESSING_SETTINGS["image_size"]
        self.use_landmarks = config.PREPROCESSING_SETTINGS["use_landmarks"]
        
    def extract_landmarks(self, image):
        """
        Extrait les landmarks MediaPipe d'une image
        
        Args:
            image: Image BGR (OpenCV format)
            
        Returns:
            landmarks: Array numpy de shape (21, 3) ou None si aucune main détectée
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
            ])
            return landmarks
        return None
    
    def preprocess_image(self, image_path, extract_landmarks_only=False):
        """
        Prétraite une image pour l'entraînement
        
        Args:
            image_path: Chemin vers l'image
            extract_landmarks_only: Si True, retourne seulement les landmarks
            
        Returns:
            processed_data: Image prétraitée ou landmarks selon le mode
        """
        # Lire l'image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Si on utilise seulement les landmarks
        if self.use_landmarks or extract_landmarks_only:
            landmarks = self.extract_landmarks(image)
            if landmarks is None:
                return None
            
            # Normaliser les landmarks (optionnel)
            if config.PREPROCESSING_SETTINGS["normalize"]:
                # Normaliser les coordonnées x, y, z séparément
                landmarks[:, 0] = (landmarks[:, 0] - landmarks[:, 0].min()) / (
                    landmarks[:, 0].max() - landmarks[:, 0].min() + 1e-8)
                landmarks[:, 1] = (landmarks[:, 1] - landmarks[:, 1].min()) / (
                    landmarks[:, 1].max() - landmarks[:, 1].min() + 1e-8)
                landmarks[:, 2] = (landmarks[:, 2] - landmarks[:, 2].min()) / (
                    landmarks[:, 2].max() - landmarks[:, 2].min() + 1e-8)
            
            return landmarks.flatten()  # Retourner un vecteur plat
        
        # Sinon, prétraiter l'image
        # Redimensionner
        processed = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convertir en niveaux de gris (optionnel, peut être amélioré)
        # processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Normaliser
        if config.PREPROCESSING_SETTINGS["normalize"]:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def process_dataset(self, raw_data_dir=None, output_dir=None):
        """
        Traite tout le dataset brut et génère les données prétraitées
        
        Args:
            raw_data_dir: Répertoire contenant les images brutes
            output_dir: Répertoire de sortie pour les données prétraitées
        """
        if raw_data_dir is None:
            raw_data_dir = Path(config.RAW_DATA_DIR)
        else:
            raw_data_dir = Path(raw_data_dir)
            
        if output_dir is None:
            if self.use_landmarks:
                output_dir = Path(config.LANDMARKS_DATA_DIR)
            else:
                output_dir = Path(config.PROCESSED_DATA_DIR)
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parcourir toutes les classes
        classes = sorted([d.name for d in raw_data_dir.iterdir() if d.is_dir()])
        print(f"Classes trouvees: {classes}")
        
        all_data = []
        all_labels = []
        
        # Compter d'abord pour détecter le déséquilibre
        class_counts = {}
        for class_name in classes:
            class_dir = raw_data_dir / class_name
            image_files = [f for f in class_dir.iterdir() 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            class_counts[class_name] = len(image_files)
        
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 1.5:
            print(f"\n⚠️  DESEQUILIBRE DETECTE!")
            print(f"   Ratio max/min: {imbalance_ratio:.2f}x")
            print(f"   Minimum: {min_count}, Maximum: {max_count}")
            print(f"   Recommandation: Equilibrez le dataset avec:")
            print(f"   python utils/balance_dataset.py")
            print()
        
        for class_idx, class_name in enumerate(classes):
            class_dir = raw_data_dir / class_name
            image_files = sorted([f for f in class_dir.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            
            print(f"\nTraitement de la classe '{class_name}' ({len(image_files)} images)...")
            
            class_data = []
            valid_count = 0
            
            for img_path in tqdm(image_files, desc=f"Classe {class_name}"):
                processed = self.preprocess_image(img_path)
                
                if processed is not None:
                    class_data.append(processed)
                    valid_count += 1
                else:
                    print(f"  Avertissement: Aucune main detectee dans {img_path.name}")
            
            print(f"  Images valides: {valid_count}/{len(image_files)}")
            
            if class_data:
                all_data.extend(class_data)
                all_labels.extend([class_idx] * len(class_data))
        
        # Convertir en arrays numpy
        if self.use_landmarks:
            all_data = np.array(all_data, dtype=np.float32)
        else:
            all_data = np.array(all_data, dtype=np.float32)
        
        all_labels = np.array(all_labels, dtype=np.int32)
        
        # Sauvegarder
        np.save(output_dir / "X.npy", all_data)
        np.save(output_dir / "y.npy", all_labels)
        
        # Sauvegarder le mapping des classes
        import json
        class_mapping = {idx: name for idx, name in enumerate(classes)}
        with open(output_dir / "class_mapping.json", 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Pretraitement termine!")
        print(f"Donnees sauvegardees dans: {output_dir}")
        print(f"Nombre total d'echantillons: {len(all_data)}")
        print(f"Shape des donnees: {all_data.shape}")
        print(f"Shape des labels: {all_labels.shape}")
        print(f"{'='*60}")
        
        return all_data, all_labels, class_mapping
    
    def create_sequences(self, data, labels, sequence_length=None):
        """
        Crée des séquences temporelles pour les modèles LSTM
        
        Args:
            data: Array de données (N, features)
            labels: Array de labels (N,)
            sequence_length: Longueur des séquences
            
        Returns:
            sequences: Array de séquences (N-seq_length+1, seq_length, features)
            sequence_labels: Labels correspondants
        """
        if sequence_length is None:
            sequence_length = config.MODEL_SETTINGS["sequence_length"]
        
        sequences = []
        sequence_labels = []
        
        # Grouper par classe pour créer des séquences cohérentes
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            
            # Créer des séquences pour cette classe
            for i in range(len(class_indices) - sequence_length + 1):
                seq = data[class_indices[i:i+sequence_length]]
                sequences.append(seq)
                sequence_labels.append(label)
        
        return np.array(sequences, dtype=np.float32), np.array(sequence_labels, dtype=np.int32)


def split_dataset(data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Divise le dataset en train/validation/test
    
    Args:
        data: Données
        labels: Labels
        train_ratio: Proportion pour l'entraînement
        val_ratio: Proportion pour la validation
        test_ratio: Proportion pour le test
        
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Diviser d'abord en train et (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=(val_ratio + test_ratio), 
        stratify=labels, random_state=42
    )
    
    # Diviser (val+test) en val et test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size),
        stratify=y_temp, random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    """Fonction principale pour le prétraitement"""
    preprocessor = HandPreprocessor()
    
    print("="*60)
    print("PRETRAITEMENT DU DATASET")
    print("="*60)
    print(f"Mode: {'Landmarks' if preprocessor.use_landmarks else 'Images'}")
    print(f"Taille d'image: {preprocessor.image_size}x{preprocessor.image_size}")
    print("="*60 + "\n")
    
    data, labels, class_mapping = preprocessor.process_dataset()
    
    print("\nDivision du dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
        data, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Sauvegarder les splits
    output_dir = Path(config.LANDMARKS_DATA_DIR if preprocessor.use_landmarks 
                     else config.PROCESSED_DATA_DIR)
    
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    
    print(f"\nSplits sauvegardes dans: {output_dir}")
    print(f"Train: {len(X_train)} echantillons")
    print(f"Validation: {len(X_val)} echantillons")
    print(f"Test: {len(X_test)} echantillons")


if __name__ == "__main__":
    main()

