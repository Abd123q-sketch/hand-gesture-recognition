import numpy as np
import pandas as pd

def show_sample(folder):
    print("\n===============================")
    print("===== SAMPLE FROM", folder.upper(), "=====")
    print("===============================\n")

    # Charger X et y du dossier
    X = np.load(f"{folder}/X.npy")
    y = np.load(f"{folder}/y.npy")

    # Prendre un échantillon
    sample = X[0]   # (21, 3)

    # Construire tableau
    df = pd.DataFrame(sample, columns=["x", "y", "z"])
    df.insert(0, "landmark", range(21))

    # Afficher proprement
    print(df.to_string(index=False))
    print("\nClasse associée :", y[0])


# --- Exécuter pour train + val + test ---
for folder in ["train", "val", "test"]:
    show_sample(folder)
