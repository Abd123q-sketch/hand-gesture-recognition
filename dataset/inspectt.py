import numpy as np
import matplotlib.pyplot as plt
import os

# Chemins vers tes datasets préparés
train_path = "X_train.npy"
val_path   = "X_val.npy"
test_path  = "X_test.npy"
y_train_path = "y_train.npy"
y_val_path   = "y_val.npy"
y_test_path  = "y_test.npy"

# Chargement des datasets
X_train = np.load(train_path)
y_train = np.load(y_train_path)
X_val   = np.load(val_path)
y_val   = np.load(y_val_path)
X_test  = np.load(test_path)
y_test  = np.load(y_test_path)

print("=== Infos Dataset ===")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val  : {X_val.shape}, y_val  : {y_val.shape}")
print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

# Afficher un exemple de séquence
seq_index = 0  # index de la séquence à visualiser
seq = X_train[seq_index]
label = y_train[seq_index]

print(f"\nExemple de séquence {seq_index} (taille {seq.shape}):")
print(f"Label (one-hot): {label}")
print(f"Label (indice): {np.argmax(label)}")

# Affichage des images de la séquence
plt.figure(figsize=(15,3))
for i, img in enumerate(seq):
    plt.subplot(1, len(seq), i+1)
    plt.imshow(img.astype(np.uint8))
    plt.axis('off')
plt.suptitle(f"Séquence {seq_index} - Classe {np.argmax(label)}")
plt.show()
