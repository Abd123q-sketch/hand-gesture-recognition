import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===== 1️⃣ Définir le dossier où se trouvent les fichiers =====
data_dir = 'C:\\Users\\Hp\\Desktop\\mon_projet_deeplearning\\src'  # chemin classique avec doubles backslashes

X_path = os.path.join(data_dir, 'X.npy')
y_path = os.path.join(data_dir, 'y.npy')

# ===== 2️⃣ Vérifier si les fichiers existent =====
if not os.path.exists(X_path) or not os.path.exists(y_path):
    raise FileNotFoundError("Vérifie que X.npy et y.npy existent dans " + data_dir)

# ===== 3️⃣ Charger les fichiers =====
X = np.load(X_path)
y = np.load(y_path)

print("Shape X avant reshape :", X.shape)
print("Shape y avant one-hot :", y.shape)

# ===== 4️⃣ Préparer les données =====
# Reshape X pour LSTM : (num_sequences, timesteps, 21*3)
num_sequences, timesteps, points, coords = X.shape
X = X.reshape((num_sequences, timesteps, points * coords))
print("Shape X après reshape :", X.shape)

# Normalisation
X = X / np.max(X)

# One-hot encoder les labels
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)
print("Shape y après one-hot :", y.shape)

# ===== 5️⃣ Construire le modèle LSTM =====
model = Sequential([
    LSTM(64, input_shape=(timesteps, points*coords)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===== 6️⃣ Vérifier le modèle =====
model.summary()
