# src/eval_test.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# chemins (adapter si besoin)
BASE = os.path.join(os.path.dirname(__file__), "..", "dataset")
X_test_path = os.path.join(BASE, "test", "X.npy")
y_test_path = os.path.join(BASE, "test", "y.npy")

# charger
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

# vérifier shapes
print("X_test shape before reshape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Adapter la forme selon ton modèle (si tu as utilisé seq len =1)
# -> résultat attendu : (N, 1, 64, 64, 3) ou (N, 1, 21, 3) selon l'entraînement
# Si ton modèle attend (None,1,21,3) utilise la ligne suivante ; sinon commente et adapte.
try:
    # tentative reshape sûre : si déjà (N,1,...) cette ligne ne changera rien
    X_test = np.expand_dims(X_test, axis=1)
except Exception:
    pass

print("X_test shape after reshape:", X_test.shape)

# charger modèle (change le nom si nécessaire)
model_path = "C:\\Users\\Hp\\Desktop\\mon_projet_deeplearning\\src\\hand_gesture_cnn_lstm.h5"

model = load_model(model_path)
print("Modèle chargé avec succès !")


# prédictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = y_pred_proba.argmax(axis=1)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# afficher quelques prédictions brutes (5 premiers)
print("\n=== Exemples : premières prédictions brutes (5) ===")
for i in range(min(5, len(X_test))):
    print(f"idx {i} true={y_test[i]} pred={y_pred[i]} probs={np.round(y_pred_proba[i],3)}")
