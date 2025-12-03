import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import os

# --- Paramètres ---
model_path = "hand_gesture_cnn_lstm.h5"
gestes = ["first", "like", "okay", "open_hand", "peace", "thumbs_up"]

dataset_dir = "../dataset"  # chemin vers ton dossier dataset

# --- Charger modèle ---
model = load_model(model_path)
print(f"Modèle chargé : {model_path}")

# --- Charger dataset depuis dataset_dir ---
X_test_path = os.path.join(dataset_dir, "X_test.npy")
y_test_path = os.path.join(dataset_dir, "y_test.npy")

X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

# --- Convertir y_test one-hot en labels si nécessaire ---
if y_test.ndim > 1 and y_test.shape[1] > 1:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test

print(f"Nombre de séquences dans le test set : {X_test.shape[0]}")

# --- Prédiction sur le test set ---
y_pred_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- Déterminer les classes réellement présentes ---
classes_presentes = np.unique(y_test_labels)
target_names_presentes = [gestes[i] for i in classes_presentes]

# --- Rapport de classification ---
report = classification_report(y_test_labels, y_pred, target_names=target_names_presentes)
print("\n=== Classification Report ===")
print(report)

# --- Confiance moyenne par classe ---
for i in classes_presentes:
    class_probs = y_pred_probs[y_test_labels == i, i]
    print(f"{gestes[i]} - confiance moyenne : {np.mean(class_probs):.2f}")
