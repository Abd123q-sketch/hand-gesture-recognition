import os

base_path = "../dataset"
subfolders = ["train", "val", "test"]

for folder in subfolders:
    folder_path = os.path.join(base_path, folder)
    print(f"\n=== Contenu du dossier {folder} ===")
    extensions = set()
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        extensions.add(ext)
    print("Extensions trouvées :", extensions)
