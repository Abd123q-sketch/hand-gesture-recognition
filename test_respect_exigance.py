import os
import numpy as np

print("\n====== VALIDATION DU PROJET — RESPECT DES EXIGENCES ======\n")

BASE_PATH = "C:\\Users\\Hp\\Desktop\\mon_projet_deeplearning\\dataset"

required_splits = ["train", "val", "test"]
valid = True

# ---------------------------------------------------------
# 1️⃣ Vérification structure dataset
# ---------------------------------------------------------
print("📁 Vérification de la structure du dataset :")
for split in required_splits:
    split_path = os.path.join(BASE_PATH, split)
    if not os.path.exists(split_path):
        print(f"❌ Dossier manquant : {split_path}")
        valid = False
    else:
        print(f"✔ {split_path} OK")

# ---------------------------------------------------------
# 2️⃣ Vérification présence des fichiers X.npy et y.npy DANS CHAQUE DOSSIER
# ---------------------------------------------------------
print("\n🧪 Vérification des fichiers X.npy / y.npy :")

for split in required_splits:
    x_path = os.path.join(BASE_PATH, split, "X.npy")
    y_path = os.path.join(BASE_PATH, split, "y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"❌ Fichiers manquants pour {split} : {x_path} ou {y_path}")
        valid = False
    else:
        print(f"✔ Fichiers OK pour {split}")

# ---------------------------------------------------------
# 3️⃣ Vérification contenu + dimensions (train)
# ---------------------------------------------------------
print("\n📐 Vérification du contenu et dimensions :")

try:
    X_train = np.load(os.path.join(BASE_PATH, "train", "X.npy"))
    y_train = np.load(os.path.join(BASE_PATH, "train", "y.npy"))

    print(f"✔ Dimensions X_train : {X_train.shape}")
    print(f"✔ Dimensions y_train : {y_train.shape}")

    if X_train.shape[1:] == (21, 3):
        print("✔ Landmarks MediaPipe corrects (21, 3)")
    else:
        print("❌ Format incorrect des landmarks")
        valid = False

    if len(X_train.shape) == 3:
        print("✔ Format compatible CNN + LSTM")
    else:
        print("❌ Format incompatible pour CNN + LSTM")
        valid = False

except Exception as e:
    print("❌ Erreur lors du chargement :", e)
    valid = False

# ---------------------------------------------------------
# 4️⃣ Résultat final
# ---------------------------------------------------------
print("\n===================== RESULTAT =====================")

if valid:
    print("🎉✔ Toutes les exigences du projet sont validées !")
    print("🚀 Le dataset est propre, structuré et compatible CNN+LSTM.")
    print("➡️ Tu peux passer à la PHASE 4 : Modélisation.")
else:
    print("❌ Certaines exigences ne sont PAS respectées.")
    print("⚠️ Corrige les erreurs avant de commencer la modélisation.")
