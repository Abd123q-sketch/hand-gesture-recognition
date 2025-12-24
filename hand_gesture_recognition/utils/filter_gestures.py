"""
Script pour filtrer et organiser les gestes du dataset.
Ne conserve que les dossiers de gestes spécifiés.
"""

import shutil
from pathlib import Path

# Configuration
DATASET_PATH = Path(__file__).resolve().parents[2] / "dataset"
RAW_PATH = DATASET_PATH / "raw"
BACKUP_PATH = DATASET_PATH / "raw_backup"

# Liste des gestes à conserver (noms des dossiers)
GESTURES_TO_KEEP = {
    'open_palm': '🖐️',
    'fist': '✊',
    'victory': '✌️',
    'thumbs_up': '👍',
    'okay': '👌',
    'pointing': '👉'
}

def filter_gestures():
    # Créer un backup avant de commencer
    if not BACKUP_PATH.exists():
        shutil.copytree(RAW_PATH, BACKUP_PATH)
        print(f"Backup créé dans: {BACKUP_PATH}")
    
    # Lister tous les dossiers de gestes
    all_gestures = [d for d in RAW_PATH.iterdir() if d.is_dir()]
    
    # Identifier les dossiers à supprimer
    to_remove = [d for d in all_gestures if d.name not in GESTURES_TO_KEEP]
    
    if not to_remove:
        print("Aucun dossier à supprimer. Les gestes sont déjà filtrés.")
        return
    
    # Afficher les actions à effectuer
    print("Dossiers à conserver:")
    for name, emoji in GESTURES_TO_KEEP.items():
        print(f"- {emoji} {name}")
    
    print("\nDossiers qui seront supprimés:")
    for d in to_remove:
        print(f"- {d.name}")
    
    # Demander confirmation
    confirm = input("\nVoulez-vous continuer ? (o/n): ")
    if confirm.lower() != 'o':
        print("Annulé.")
        return
    
    # Supprimer les dossiers non désirés
    for d in to_remove:
        try:
            shutil.rmtree(d)
            print(f"Supprimé: {d.name}")
        except Exception as e:
            print(f"Erreur lors de la suppression de {d.name}: {e}")
    
    print("\nNettoyage terminé. Les gestes ont été filtrés avec succès.")
    print(f"Un backup complet est disponible dans: {BACKUP_PATH}")

if __name__ == "__main__":
    filter_gestures()
