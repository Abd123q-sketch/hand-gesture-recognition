import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Config
RAW_PATH = "raw"       # dossier contenant les sous-dossiers pour chaque geste
IMG_SIZE = 64          # taille des images pour CNN
SEQ_LENGTH = 21        # nombre d’images par séquence
GESTURES = os.listdir(RAW_PATH)
NUM_CLASSES = len(GESTURES)

X, y = [], []

for label, gesture in enumerate(GESTURES):
    gesture_path = os.path.join(RAW_PATH, gesture)
    # lister les images dans ce dossier
    image_files = sorted(os.listdir(gesture_path))  
    # grouper par séquences de SEQ_LENGTH
    for i in range(0, len(image_files) - SEQ_LENGTH + 1, SEQ_LENGTH):
        seq_images = []
        for j in range(SEQ_LENGTH):
            img_path = os.path.join(gesture_path, image_files[i + j])
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            seq_images.append(img)
        
        if len(seq_images) == SEQ_LENGTH:
            X.append(seq_images)
            y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)
y = to_categorical(y, num_classes=NUM_CLASSES)

# Split train / val / test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.36, random_state=42, stratify=y_temp)  # ~10% test

# Sauvegarde
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Dataset préparé !")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)
