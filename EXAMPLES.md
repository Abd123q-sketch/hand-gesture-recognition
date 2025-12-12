# Exemples d'Utilisation - Hand Gesture Recognition

Ce document fournit des exemples concrets d'utilisation du projet.

## 📚 Table des Matières

1. [Exemple Complet de Workflow](#exemple-complet-de-workflow)
2. [Collecte de Données](#collecte-de-données)
3. [Prétraitement](#prétraitement)
4. [Entraînement](#entraînement)
5. [Évaluation](#évaluation)
6. [Inférence Temps Réel](#inférence-temps-réel)
7. [Cas d'Usage Avancés](#cas-dusage-avancés)

---

## 🔄 Exemple Complet de Workflow

### Scénario : Créer un système de reconnaissance de 3 gestes

```bash
# 1. Installation
pip install -r requirements.txt

# 2. Collecte pour "open_palm"
python src/data_collection.py
# Choisir option 2, classe: open_palm, 150 échantillons

# 3. Collecte pour "closed_fist"
python src/data_collection.py
# Choisir option 2, classe: closed_fist, 150 échantillons

# 4. Collecte pour "victory"
python src/data_collection.py
# Choisir option 2, classe: victory, 150 échantillons

# 5. Prétraitement
python src/preprocessing.py

# 6. Entraînement CNN
python src/train.py --model cnn

# 7. Évaluation
python src/evaluate.py --model models/cnn_best.h5

# 8. Test temps réel
python src/inference.py --model models/cnn_best.h5
```

---

## 📸 Collecte de Données

### Exemple 1 : Collecte interactive pour une classe

```python
from src.data_collection import DataCollector

collector = DataCollector()
collector.initialize_camera()

# Collecter 200 échantillons pour "thumbs_up"
collector.collect_samples("thumbs_up", num_samples=200)

collector.release()
```

### Exemple 2 : Collecte programmatique pour toutes les classes

```python
from src.data_collection import DataCollector

collector = DataCollector()
collector.initialize_camera()

# Collecter pour toutes les classes définies dans config.py
results = collector.collect_all_classes()

print("Résultats:", results)
# Output: {'open_palm': 200, 'closed_fist': 200, ...}

collector.release()
```

### Exemple 3 : Collecte avec paramètres personnalisés

Modifier `config.py` :

```python
COLLECTION_SETTINGS = {
    "samples_per_class": 300,      # Plus d'échantillons
    "image_size": (320, 320),      # Images plus grandes
    "webcam_index": 1,             # Utiliser une autre webcam
}
```

---

## 🔧 Prétraitement

### Exemple 1 : Prétraitement standard

```bash
python src/preprocessing.py
```

### Exemple 2 : Prétraitement avec mode landmarks

Modifier `config.py` :

```python
PREPROCESSING_SETTINGS = {
    "image_size": 64,
    "use_landmarks": True,      # Utiliser les landmarks MediaPipe
    "normalize": True,
}
```

Puis exécuter :

```bash
python src/preprocessing.py
```

### Exemple 3 : Prétraitement personnalisé

```python
from src.preprocessing import HandPreprocessor, split_dataset
import numpy as np

preprocessor = HandPreprocessor()

# Traiter le dataset
data, labels, class_mapping = preprocessor.process_dataset(
    raw_data_dir="data/raw",
    output_dir="data/custom_processed"
)

# Diviser avec des ratios personnalisés
(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
    data, labels,
    train_ratio=0.8,  # 80% train
    val_ratio=0.1,    # 10% val
    test_ratio=0.1    # 10% test
)

# Sauvegarder
np.save("data/custom_processed/X_train.npy", X_train)
np.save("data/custom_processed/y_train.npy", y_train)
# ... etc
```

---

## 🎯 Entraînement

### Exemple 1 : Entraînement CNN simple

```bash
python src/train.py --model cnn
```

### Exemple 2 : Entraînement CNN+LSTM

```bash
python src/train.py --model cnn --lstm
```

### Exemple 3 : Entraînement avec paramètres personnalisés

Modifier `config.py` :

```python
MODEL_SETTINGS = {
    "sequence_length": 30,     # Séquences plus longues
    "batch_size": 16,          # Batch plus petit
    "epochs": 100,             # Plus d'époques
    "learning_rate": 0.0001,   # Learning rate plus faible
}
```

Puis :

```bash
python src/train.py --model cnn_lstm --lstm
```

### Exemple 4 : Entraînement programmatique

```python
from src.train import train_model

# Entraîner un modèle CNN
model, history, metrics = train_model(
    model_type='cnn',
    use_landmarks=False,
    use_lstm=False
)

print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
```

---

## 📊 Évaluation

### Exemple 1 : Évaluation standard

```bash
python src/evaluate.py --model models/cnn_best.h5
```

### Exemple 2 : Évaluation avec répertoire personnalisé

```bash
python src/evaluate.py \
    --model models/cnn_lstm_best.h5 \
    --data_dir data/landmarks \
    --output_dir logs/my_evaluation
```

### Exemple 3 : Évaluation programmatique

```python
from src.evaluate import load_model_and_data, evaluate_model

# Charger le modèle et les données
model, (X_test, y_test), class_mapping = load_model_and_data(
    "models/cnn_best.h5"
)

# Évaluer
results = evaluate_model(
    model, X_test, y_test, class_mapping,
    save_dir="logs/custom_eval"
)

print(f"Accuracy: {results['test_accuracy']:.4f}")
```

---

## 🎥 Inférence Temps Réel

### Exemple 1 : Inférence standard

```bash
python src/inference.py --model models/cnn_best.h5
```

### Exemple 2 : Inférence avec modèle LSTM

```bash
python src/inference.py --model models/cnn_lstm_best.h5 --lstm
```

### Exemple 3 : Inférence avec modèle Landmarks

```bash
python src/inference.py --model models/landmark_best.h5 --landmarks True
```

### Exemple 4 : Inférence programmatique

```python
from src.inference import GestureRecognizer

# Initialiser le reconnaisseur
recognizer = GestureRecognizer(
    model_path="models/cnn_best.h5",
    use_landmarks=False,
    use_lstm=False
)

# Lancer la reconnaissance
recognizer.run()
```

### Exemple 5 : Inférence sur une image unique

```python
import cv2
from src.inference import GestureRecognizer
import numpy as np

recognizer = GestureRecognizer("models/cnn_best.h5")

# Charger une image
image = cv2.imread("test_image.jpg")

# Prédire
class_name, confidence, probabilities = recognizer.predict(image)

print(f"Geste: {class_name}")
print(f"Confiance: {confidence:.2%}")
print(f"Probabilités: {probabilities}")
```

---

## 🚀 Cas d'Usage Avancés

### Cas 1 : Comparaison de Modèles

```bash
# Entraîner plusieurs modèles
python src/train.py --model cnn
python src/train.py --model cnn --lstm
python src/train.py --model landmark
python src/train.py --model landmark --lstm

# Évaluer chacun
python src/evaluate.py --model models/cnn_best.h5 --output_dir logs/eval_cnn
python src/evaluate.py --model models/cnn_lstm_best.h5 --output_dir logs/eval_cnn_lstm
python src/evaluate.py --model models/landmark_best.h5 --output_dir logs/eval_landmark
python src/evaluate.py --model models/landmark_lstm_best.h5 --output_dir logs/eval_landmark_lstm

# Comparer les résultats dans logs/
```

### Cas 2 : Fine-tuning avec Plus de Données

```python
# 1. Collecter plus de données pour une classe spécifique
# 2. Ajouter les nouvelles images dans data/raw/[class_name]/
# 3. Ré-exécuter le prétraitement
python src/preprocessing.py

# 4. Ré-entraîner en chargeant les poids précédents
from src.models import build_cnn_model
from src.train import load_data
import tensorflow as tf

# Charger les données
(X_train, y_train), (X_val, y_val), (X_test, y_test), class_mapping = load_data()

# Créer le modèle
model = build_cnn_model((64, 64, 3), len(class_mapping))

# Charger les poids précédents (si disponibles)
try:
    model.load_weights("models/cnn_best.h5")
    print("Poids précédents chargés")
except:
    print("Nouveau modèle")

# Continuer l'entraînement...
```

### Cas 3 : Export pour Production

```python
import tensorflow as tf

# Charger le modèle
model = tf.keras.models.load_model("models/cnn_best.h5")

# Convertir en TensorFlow Lite (pour mobile)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("models/cnn.tflite", "wb") as f:
    f.write(tflite_model)

print("Modèle TFLite sauvegardé!")
```

### Cas 4 : Intégration dans une Application

```python
from src.inference import GestureRecognizer
import cv2

class GestureApp:
    def __init__(self, model_path):
        self.recognizer = GestureRecognizer(model_path)
        self.gesture_history = []
    
    def process_frame(self, frame):
        """Traite un frame et retourne le geste détecté"""
        class_name, confidence, probs = self.recognizer.predict(frame)
        
        if confidence > 0.7:  # Seuil élevé
            self.gesture_history.append(class_name)
            return class_name, confidence
        
        return None, 0.0
    
    def get_most_common_gesture(self, window=10):
        """Retourne le geste le plus fréquent dans la fenêtre"""
        if len(self.gesture_history) < window:
            return None
        
        recent = self.gesture_history[-window:]
        from collections import Counter
        return Counter(recent).most_common(1)[0][0]

# Utilisation
app = GestureApp("models/cnn_best.h5")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gesture, conf = app.process_frame(frame)
    
    if gesture:
        print(f"Geste détecté: {gesture} ({conf:.2%})")
        # Action basée sur le geste
        if gesture == "thumbs_up":
            print("👍 Action: Like!")
        elif gesture == "open_palm":
            print("✋ Action: Stop!")
    
    cv2.imshow("App", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 💡 Conseils et Astuces

### Améliorer les Performances

1. **Plus de données** : Collectez au moins 200-300 échantillons par classe
2. **Variété** : Variez les angles, éclairages, et positions
3. **Équilibrage** : Assurez-vous que toutes les classes ont le même nombre d'échantillons
4. **Prétraitement** : Expérimentez avec différents modes (landmarks vs images)

### Débogage

1. **Visualiser les données** :
   ```bash
   python utils/visualize_data.py --stats
   python utils/visualize_data.py --class open_palm
   ```

2. **Vérifier le prétraitement** :
   ```python
   import numpy as np
   data = np.load("data/landmarks/X_train.npy")
   print(f"Shape: {data.shape}")
   print(f"Min: {data.min()}, Max: {data.max()}")
   ```

3. **Monitorer l'entraînement** :
   ```bash
   tensorboard --logdir logs/
   ```

---

## 📝 Notes Importantes

- Les modèles LSTM nécessitent des séquences complètes avant de faire des prédictions
- Le mode landmarks est plus rapide mais peut être moins précis que les images complètes
- Assurez-vous que le prétraitement est identique entre entraînement et inférence
- Les performances dépendent fortement de la qualité et quantité des données

---

Pour plus d'informations, consultez le [README.md](README.md) principal.

