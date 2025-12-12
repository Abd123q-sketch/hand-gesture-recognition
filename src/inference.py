"""
Script d'inférence en temps réel pour la reconnaissance de gestes
Utilise la webcam pour capturer et classifier les gestes en temps réel
"""
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp
from pathlib import Path
import json
import sys

# Ajouter le répertoire parent au path pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.preprocessing import HandPreprocessor


class GestureRecognizer:
    """Classe pour la reconnaissance de gestes en temps réel avec CNN+LSTM"""
    
    def __init__(self, model_path, use_full_frame=True):
        """
        Initialise le reconnaisseur de gestes CNN+LSTM
        
        Args:
            model_path: Chemin vers le modèle sauvegardé (.h5)
            use_full_frame: Si True, utilise le frame complet (recommandé pour correspondre à l'entraînement)
        """
        # Charger le modèle
        print(f"Chargement du modele CNN+LSTM: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path))
        
        # Le modèle est toujours CNN+LSTM avec images
        self.use_landmarks = False
        self.use_lstm = True
        self.use_full_frame = use_full_frame  # Utiliser le frame complet ou la bounding box
        
        # Charger le mapping des classes depuis data/processed ou dataset/
        processed_dir = Path(config.PROCESSED_DATA_DIR)
        dataset_dir = Path(config.PROJECT_ROOT) / "dataset"
        
        # Charger le mapping des classes
        if processed_dir.exists() and (processed_dir / "class_mapping.json").exists():
            mapping_path = processed_dir / "class_mapping.json"
            with open(mapping_path, 'r') as f:
                class_mapping = json.load(f)
        elif dataset_dir.exists() and (dataset_dir / "class_mapping.json").exists():
            mapping_path = dataset_dir / "class_mapping.json"
            with open(mapping_path, 'r') as f:
                class_mapping = json.load(f)
        else:
            # Créer un mapping par défaut
            print("⚠️  class_mapping.json non trouve, utilisation du mapping par defaut")
            class_mapping = {str(i): name for i, name in enumerate(config.GESTURE_CLASSES)}
        
        self.class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
        print(f"Classes chargees ({len(self.class_names)}): {self.class_names}")
        
        # Paramètres pour CNN+LSTM
        self.sequence_length = config.MODEL_SETTINGS["sequence_length"]
        self.image_size = config.PREPROCESSING_SETTINGS["image_size"]
        # Augmenter le seuil de confiance pour être plus strict
        self.confidence_threshold = max(config.INFERENCE_SETTINGS["confidence_threshold"], 0.6)
        self.smoothing_window = config.INFERENCE_SETTINGS["smoothing_window"]
        
        # Historique des probabilités pour un meilleur lissage
        self.probability_history = deque(maxlen=self.smoothing_window)
        
        # Queues pour les séquences temporelles et le lissage
        self.frames_queue = deque(maxlen=self.sequence_length)
        self.prediction_history = deque(maxlen=self.smoothing_window)
        self.probability_history = deque(maxlen=self.smoothing_window)
        
        # Initialiser MediaPipe Hands pour détecter la main
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MEDIAPIPE_SETTINGS["max_num_hands"],
            min_detection_confidence=config.MEDIAPIPE_SETTINGS["min_detection_confidence"],
            min_tracking_confidence=config.MEDIAPIPE_SETTINGS["min_tracking_confidence"]
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        print(f"\nConfiguration:")
        print(f"  - Sequence length: {self.sequence_length}")
        print(f"  - Image size: {self.image_size}x{self.image_size}")
        print(f"  - Confidence threshold: {self.confidence_threshold}")
        print(f"  - Smoothing window: {self.smoothing_window}")
        print(f"  - Preprocessing: {'Frame complet' if self.use_full_frame else 'Bounding box'}")
        
    def preprocess_frame(self, frame):
        """
        Prétraite un frame pour la prédiction CNN+LSTM
        Utilise le frame complet pour correspondre à l'entraînement
        
        Args:
            frame: Frame BGR de la webcam
            
        Returns:
            processed: Image prétraitée (height, width, channels) ou None si aucune main détectée
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        if self.use_full_frame:
            # OPTION 1: Utiliser le frame complet (recommandé - correspond à l'entraînement)
            # Les données d'entraînement utilisent cv2.imread() directement sans extraction de ROI
            processed = cv2.resize(frame, (self.image_size, self.image_size))
        else:
            # OPTION 2: Utiliser une bounding box agrandie autour de la main
            hand = results.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            x_coords = [lm.x for lm in hand.landmark]
            y_coords = [lm.y for lm in hand.landmark]
            
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)
            
            # Créer une bounding box carrée avec beaucoup de padding
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            size = max(x_max - x_min, y_max - y_min)
            size = int(size * 2.5)  # Padding important pour inclure le contexte
            
            x_min = max(0, center_x - size // 2)
            y_min = max(0, center_y - size // 2)
            x_max = min(w, center_x + size // 2)
            y_max = min(h, center_y + size // 2)
            
            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size == 0:
                return None
            processed = cv2.resize(hand_roi, (self.image_size, self.image_size))
        
        # Normaliser entre 0 et 1 (identique au preprocessing)
        if config.PREPROCESSING_SETTINGS["normalize"]:
            processed = processed.astype(np.float32) / 255.0
        else:
            processed = processed.astype(np.float32)
        
        return processed
    
    def predict(self, frame):
        """
        Prédit le geste dans un frame avec CNN+LSTM
        
        Args:
            frame: Frame BGR
            
        Returns:
            (class_name, confidence, probabilities) ou (None, 0, None)
        """
        processed = self.preprocess_frame(frame)
        
        if processed is None:
            self.frames_queue.clear()
            self.probability_history.clear()
            return None, 0.0, None
        
        # Ajouter le frame prétraité à la queue de séquences
        self.frames_queue.append(processed)
        
        # Attendre d'avoir une séquence complète pour CNN+LSTM
        if len(self.frames_queue) < self.sequence_length:
            return None, 0.0, None
        
        # Préparer la séquence pour le modèle CNN+LSTM
        # Format attendu: (batch, sequence_length, height, width, channels)
        seq = np.array(list(self.frames_queue), dtype=np.float32)
        seq = np.expand_dims(seq, axis=0)  # Ajouter la dimension batch
        
        # Vérifier la shape
        expected_shape = (1, self.sequence_length, self.image_size, self.image_size, 3)
        if seq.shape != expected_shape:
            print(f"⚠️  Shape incorrecte: {seq.shape}, attendu: {expected_shape}")
            return None, 0.0, None
        
        # Prédiction avec le modèle CNN+LSTM
        pred_proba = self.model.predict(seq, verbose=0)
        probabilities_array = pred_proba[0]
        
        # Ajouter les probabilités à l'historique pour lissage
        self.probability_history.append(probabilities_array)
        
        # Calculer les probabilités moyennes sur les dernières prédictions
        if len(self.probability_history) >= 3:
            # Moyenne des probabilités sur la fenêtre de lissage
            smoothed_probs = np.mean(list(self.probability_history), axis=0)
        else:
            smoothed_probs = probabilities_array
        
        # Trouver la classe avec la probabilité moyenne la plus élevée
        smoothed_class_id = np.argmax(smoothed_probs)
        smoothed_confidence = float(smoothed_probs[smoothed_class_id])
        
        # Vérifier que la confiance est suffisante
        if smoothed_confidence < self.confidence_threshold:
            # Si la confiance est trop faible, ne pas faire de prédiction
            return None, 0.0, None
        
        # Ajouter à l'historique des classes pour vérification de cohérence
        self.prediction_history.append(smoothed_class_id)
        
        # Vérifier la cohérence: si les dernières prédictions sont différentes, être plus strict
        if len(self.prediction_history) >= 5:
            from collections import Counter
            most_common_class, count = Counter(self.prediction_history).most_common(1)[0]
            consistency = count / len(self.prediction_history)
            
            # Si les prédictions ne sont pas cohérentes, être plus strict
            if consistency < 0.6:
                # Les prédictions varient trop, augmenter le seuil
                if smoothed_confidence < 0.7:
                    return None, 0.0, None
        
        class_name = self.class_names[smoothed_class_id]
        probabilities = {name: float(prob) for name, prob in 
                        zip(self.class_names, smoothed_probs)}
        
        return class_name, smoothed_confidence, probabilities
    
    def run(self):
        """Lance la reconnaissance en temps réel"""
        cap = cv2.VideoCapture(config.COLLECTION_SETTINGS["webcam_index"])
        
        if not cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la webcam")
        
        print("\n" + "="*60)
        print("RECONNAISSANCE DE GESTES EN TEMPS REEL - CNN+LSTM")
        print("="*60)
        print("Conseils pour une meilleure precision:")
        print("  - Maintenez votre geste stable pendant 1-2 secondes")
        print("  - Assurez-vous d'avoir un bon eclairage")
        print("  - Gardez votre main bien visible dans le cadre")
        print("  - Attendez que la sequence soit complete avant de bouger")
        print("="*60)
        print("Appuyez sur 'q' pour quitter")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Miroir horizontal
            frame = cv2.flip(frame, 1)
            
            # Détecter les mains et dessiner les landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Prédiction
                class_name, confidence, probabilities = self.predict(frame)
                
                if class_name:
                    # Afficher la prédiction avec indication de confiance
                    if confidence >= 0.8:
                        color = (0, 255, 0)  # Vert = très confiant
                        confidence_text = "HIGH"
                    elif confidence >= self.confidence_threshold:
                        color = (0, 200, 255)  # Orange = confiant
                        confidence_text = "MED"
                    else:
                        color = (0, 165, 255)  # Orange clair = faible confiance
                        confidence_text = "LOW"
                    
                    text = f"{class_name} ({confidence*100:.1f}%) [{confidence_text}]"
                    
                    cv2.putText(frame, text, (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Afficher un avertissement si la confiance est faible
                    if confidence < self.confidence_threshold:
                        cv2.putText(frame, "Low confidence - Hold gesture steady", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
                    # Afficher toutes les probabilités (top 3 seulement pour plus de clarté)
                    y_offset = 100 if confidence < self.confidence_threshold else 80
                    top_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
                    for name, prob in top_probs:
                        bar_width = int(prob * 200)
                        if name == class_name:
                            bar_color = (0, 255, 0) if confidence >= 0.8 else (0, 200, 255)
                        else:
                            bar_color = (100, 100, 100)
                        cv2.rectangle(frame, (10, y_offset), 
                                     (10 + bar_width, y_offset + 15), bar_color, -1)
                        cv2.putText(frame, f"{name[:12]}: {prob*100:.0f}%",
                                   (220, y_offset + 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        y_offset += 20
                else:
                    # Afficher le progrès de collecte de frames
                    progress = len(self.frames_queue) / self.sequence_length
                    progress_text = f"Collecting frames... {len(self.frames_queue)}/{self.sequence_length}"
                    cv2.putText(frame, progress_text, (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    # Barre de progression
                    bar_width = int(progress * 200)
                    cv2.rectangle(frame, (10, 60), (10 + bar_width, 75), (255, 255, 0), -1)
            else:
                self.frames_queue.clear()
                self.prediction_history.clear()
                self.probability_history.clear()
                cv2.putText(frame, "No hand detected", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Afficher le nombre de frames dans la queue
            cv2.putText(frame, f"Queue: {len(self.frames_queue)}/{self.sequence_length}",
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Hand Gesture Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Fonction principale - Reconnaissance CNN+LSTM en temps réel"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reconnaissance de gestes en temps reel avec CNN+LSTM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python src/inference.py --model models/cnn_lstm_best.h5
  python src/inference.py --model models/cnn_lstm_final.h5
        """
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Chemin vers le modele CNN+LSTM sauvegarde (.h5)')
    parser.add_argument('--use-bbox', action='store_true',
                       help='Utiliser une bounding box autour de la main (defaut: frame complet)')
    
    args = parser.parse_args()
    
    # Vérifier que le fichier existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Erreur: Le modele {model_path} n'existe pas!")
        print(f"   Assurez-vous d'avoir entraine un modele avec: python src/train.py")
        return
    
    print("="*60)
    print("RECONNAISSANCE DE GESTES - CNN+LSTM")
    print("="*60)
    
    try:
        recognizer = GestureRecognizer(str(model_path), use_full_frame=not args.use_bbox)
        recognizer.run()
    except KeyboardInterrupt:
        print("\n\nInterruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

