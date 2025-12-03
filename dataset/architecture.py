import numpy as np

# Chemins vers les fichiers
train_X_path = 'train/X.npy'
train_y_path = 'train/y.npy'
val_X_path = 'val/X.npy'
val_y_path = 'val/y.npy'
test_X_path = 'test/X.npy'
test_y_path = 'test/y.npy'

# Charger les datasets
X_train = np.load(train_X_path)
y_train = np.load(train_y_path)
X_val = np.load(val_X_path)
y_val = np.load(val_y_path)
X_test = np.load(test_X_path)
y_test = np.load(test_y_path)

# Afficher infos
print("=== Dataset info ===")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Afficher un exemple pour inspection
print("\nExemple X_train[0]:")
print(X_train[0])
print("Label:", y_train[0])
