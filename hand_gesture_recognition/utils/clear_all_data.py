"""
Script pour vider complètement toutes les données du dataset.
Supprime TOUTES les images sans créer de backup.
"""

import shutil
from pathlib import Path

# Configuration
DATASET_PATH = Path(__file__).resolve().parents[2] / "dataset"

def clear_all_data():
    # Dossiers à vider
    folders_to_clear = ['raw', 'train', 'val', 'test']
    
    print("VIDAGE COMPLET DU DATASET")
    print("ATTENTION: Cette action supprime TOUTES les données sans backup!")
    
    for folder in folders_to_clear:
        folder_path = DATASET_PATH / folder
        if folder_path.exists():
            print(f"\nVidage du dossier {folder}...")
            # Supprimer tout le contenu du dossier
            for item in folder_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            print(f"Dossier {folder} vidé.")
        else:
            print(f"\nCréation du dossier {folder}...")
            folder_path.mkdir(exist_ok=True)
    
    print("\nToutes les données ont été supprimées.")
    print("Le dataset est maintenant vide.")

if __name__ == "__main__":
    clear_all_data()
