import os
import numpy as np

base_path = "C:\\Users\\Hp\\Desktop\\mon_projet_deeplearning\\dataset\\val"   

X = []
y = []

# Récupération des noms de dossiers
class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

# Création d’un mapping nom → id
class_to_id = {name: idx for idx, name in enumerate(class_names)}

print("Mapping classes → IDs :", class_to_id)

# Chargement des données
for class_name in class_names:
    class_dir = os.path.join(base_path, class_name)

    for file_name in os.listdir(class_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(class_dir, file_name)

            data = np.load(file_path)

            X.append(data)
            y.append(class_to_id[class_name])

X = np.array(X)
y = np.array(y)

print("X shape :", X.shape)
print("y shape :", y.shape)

np.save("X.npy", X)
np.save("y.npy", y)

print("✔ X.npy et y.npy générés avec succès !")
