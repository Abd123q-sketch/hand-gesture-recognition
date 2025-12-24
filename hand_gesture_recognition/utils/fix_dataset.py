"""
Script pour corriger les noms de classes et réorganiser le dataset.
Corrige les erreurs de capture et organise les gestes correctement.
"""

import shutil
from pathlib import Path

# Configuration
DATASET_PATH = Path(__file__).resolve().parents[2] / "dataset"
RAW_PATH = DATASET_PATH / "raw"
BACKUP_PATH = DATASET_PATH / "fix_backup"

# Mapping des corrections de noms
CORRECTIONS = {
    'okay thumbs_up': 'thumbs_up'  # Renommer "okay thumbs_up" en "thumbs_up"
}

# Gestes à conserver
GESTURES_TO_KEEP = [
    'open_palm',
    'fist', 
    'victory',
    'pointing',
    'okay',
    'thumbs_up'
]

def fix_dataset():
    # Créer un backup avant de commencer
    if not BACKUP_PATH.exists():
        shutil.copytree(DATASET_PATH, BACKUP_PATH)
        print(f"Backup créé dans: {BACKUP_PATH}")
    
    print("Correction des noms de classes...")
    
    # Étape 1: Renommer les dossiers incorrects
    for old_name, new_name in CORRECTIONS.items():
        old_path = RAW_PATH / old_name
        new_path = RAW_PATH / new_name
        
        if old_path.exists():
            if new_path.exists():
                print(f"Attention: Le dossier {new_name} existe déjà, fusion des données...")
                # Fusionner les dossiers
                for img_file in old_path.glob('*'):
                    if img_file.is_file():
                        shutil.copy2(img_file, new_path / img_file.name)
                shutil.rmtree(old_path)
                print(f"Fusionné: {old_name} -> {new_name}")
            else:
                shutil.move(str(old_path), str(new_path))
                print(f"Renommé: {old_name} -> {new_name}")
    
    # Étape 2: Supprimer les dossiers non désirés
    all_folders = [d for d in RAW_PATH.iterdir() if d.is_dir()]
    to_remove = [d for d in all_folders if d.name not in GESTURES_TO_KEEP]
    
    if to_remove:
        print("\nDossiers qui seront supprimés:")
        for d in to_remove:
            print(f"- {d.name}")
        
        confirm = input("\nVoulez-vous continuer ? (o/n): ")
        if confirm.lower() == 'o':
            for d in to_remove:
                shutil.rmtree(d)
                print(f"Supprimé: {d.name}")
        else:
            print("Annulé.")
            return
    
    # Étape 3: Afficher le résultat
    print("\nDataset corrigé:")
    for gesture in GESTURES_TO_KEEP:
        gesture_path = RAW_PATH / gesture
        if gesture_path.exists():
            count = len([f for f in gesture_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            print(f"- {gesture}: {count} images")
        else:
            print(f"- {gesture}: dossier manquant")
    
    print("\nCorrection terminée!")
    print("Prochaines étapes:")
    print("1. Lance 'python -m hand_gesture_recognition.quick_start'")
    print("2. Choisis l'option 2 pour splitter le dataset")
    print("3. Choisis l'option 3 pour réentraîner le modèle")

if __name__ == "__main__":
    fix_dataset()
