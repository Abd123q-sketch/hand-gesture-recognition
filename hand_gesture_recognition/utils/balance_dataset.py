"""
Script pour équilibrer un dataset d'images par classe.
Crée des ensembles train/val/test équilibrés à partir du dossier raw.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Configuration
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_class_sizes(dataset_path: Path) -> Dict[str, int]:
    """Retourne le nombre d'images par classe dans le dossier raw."""
    raw_path = dataset_path / "raw"
    if not raw_path.exists():
        raise FileNotFoundError(f"Dossier 'raw' introuvable dans {dataset_path}")
    
    class_sizes = {}
    for class_dir in raw_path.iterdir():
        if class_dir.is_dir():
            num_images = len([f for f in class_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            if num_images > 0:  # Ne considérer que les dossiers avec des images
                class_sizes[class_dir.name] = num_images
    
    return class_sizes


def create_splits(dataset_path: Path, min_samples: int):
    """Crée des ensembles train/val/test équilibrés."""
    raw_path = dataset_path / "raw"
    
    # Créer les dossiers s'ils n'existent pas
    for split in ["train", "val", "test"]:
        (dataset_path / split).mkdir(exist_ok=True)
    
    print(f"\nCréation des ensembles avec {min_samples} échantillons par classe...")
    
    for class_name in os.listdir(raw_path):
        class_path = raw_path / class_name
        if not class_path.is_dir():
            continue
            
        print(f"\nTraitement de la classe: {class_name}")
        
        # Lister toutes les images
        images = [f for f in class_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not images:
            print(f"  Aucune image trouvée pour la classe {class_name}")
            continue
            
        # Prendre un sous-ensemble aléatoire si nécessaire
        if len(images) > min_samples:
            images = random.sample(images, min_samples)
        
        # Calculer les tailles de chaque split
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        # Mélanger les images
        random.shuffle(images)
        
        # Répartir en train/val/test
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # Copier les images dans les dossiers correspondants
        for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest_dir = dataset_path / split / class_name
            dest_dir.mkdir(exist_ok=True)
            
            # Vider le dossier de destination s'il existe déjà
            for f in dest_dir.glob('*'):
                f.unlink()
            
            # Copier les images
            for img_path in imgs:
                shutil.copy2(img_path, dest_dir / img_path.name)
            
            print(f"  {split}: {len(imgs)} images")


def main():
    # Chemin vers le dossier du dataset
    dataset_path = Path(__file__).resolve().parents[2] / "dataset"
    
    if not dataset_path.exists():
        print(f"Erreur: Le dossier du dataset n'existe pas: {dataset_path}")
        return
    
    # Vérifier que le dossier raw existe
    raw_path = dataset_path / "raw"
    if not raw_path.exists():
        print(f"Erreur: Le dossier 'raw' est introuvable dans {dataset_path}")
        return
    
    # Afficher les tailles actuelles des classes
    try:
        class_sizes = get_class_sizes(dataset_path)
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        return
    
    if not class_sizes:
        print("Aucune classe trouvée dans le dossier 'raw'.")
        return
    
    print("Nombre d'images par classe dans le dossier 'raw':")
    for class_name, size in sorted(class_sizes.items()):
        print(f"- {class_name}: {size} images")
    
    # Déterminer le nombre minimum d'échantillons
    min_samples = min(class_sizes.values())
    print(f"\nNombre minimum d'échantillons par classe: {min_samples}")
    
    # Calculer le nombre d'images par split
    n_train = int(min_samples * TRAIN_RATIO)
    n_val = int(min_samples * VAL_RATIO)
    n_test = min_samples - n_train - n_val
    
    print(f"\nRépartition proposée par classe:")
    print(f"- Train: {n_train} images ({TRAIN_RATIO*100:.0f}%)")
    print(f"- Val:   {n_val} images ({VAL_RATIO*100:.0f}%)")
    print(f"- Test:  {n_test} images ({TEST_RATIO*100:.0f}%)")
    
    # Demander confirmation
    confirm = input("\nVoulez-vous créer les ensembles train/val/test ? (o/n): ")
    if confirm.lower() != 'o':
        print("Annulé.")
        return
    
    # Créer les ensembles équilibrés
    create_splits(dataset_path, min_samples)
    
    print("\nOpération terminée avec succès !")


if __name__ == "__main__":
    main()
