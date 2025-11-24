import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

print("X shape :", X.shape)
print("y shape :", y.shape)
print("Valeurs uniques dans y :", np.unique(y))


