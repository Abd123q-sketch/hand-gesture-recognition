import numpy as np
import os

folders = ["train", "val", "test"]

for folder in folders:
    print("\n============================")
    print(f"📂 Vérification dossier : {folder}")
    print("============================")

    X_path = os.path.join(folder, "X.npy")
    y_path = os.path.join(folder, "y.npy")

    if not os.path.exists(X_path):
        print(f"❌ Fichier introuvable : {X_path}")
        continue

    # Charger X et y
    X = np.load(X_path)
    y = np.load(y_path)

    print("✔ Chargé :", X_path)
    print("✔ Chargé :", y_path)

    print("→ X shape :", X.shape)
    print("→ y shape :", y.shape)

    # Vérification NaN
    print("🔍 NaN dans X :", np.isnan(X).any())

    # Vérification valeurs infinies
    print("🔍 Inf dans X :", np.isinf(X).any())

    # Vérification normalisation X/Y
    x_vals = X[:, :, 0]
    y_vals = X[:, :, 1]

    print("→ Min X :", x_vals.min(), " | Max X :", x_vals.max())
    print("→ Min Y :", y_vals.min(), " | Max Y :", y_vals.max())

    # Vérification structure 21 points × 3 coords
    if X.shape[1:] == (21, 3):
        print("✔ Structure correcte (21 landmarks, 3 coordonnées)")
    else:
        print("❌ Structure incorrecte :", X.shape[1:])

print("\n🎉 Vérification dataset terminée.")
