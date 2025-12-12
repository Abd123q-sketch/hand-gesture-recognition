"""
Utilitaire pour visualiser les données collectées
"""
import cv2
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au path pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def visualize_samples(class_name, num_samples=9):
    """
    Visualise des échantillons d'une classe
    
    Args:
        class_name: Nom de la classe
        num_samples: Nombre d'échantillons à afficher
    """
    class_dir = Path(config.RAW_DATA_DIR) / class_name
    if not class_dir.exists():
        print(f"Classe '{class_name}' non trouvee dans {config.RAW_DATA_DIR}")
        return
    
    image_files = sorted([f for f in class_dir.iterdir() 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if len(image_files) == 0:
        print(f"Aucune image trouvee pour la classe '{class_name}'")
        return
    
    # Sélectionner des échantillons uniformément répartis
    indices = np.linspace(0, len(image_files) - 1, num_samples, dtype=int)
    selected_files = [image_files[i] for i in indices]
    
    # Créer une grille
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, img_path in enumerate(selected_files):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{img_path.name}")
        axes[idx].axis('off')
    
    # Masquer les axes inutilisés
    for idx in range(len(selected_files), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Echantillons de la classe: {class_name}", fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_dataset_stats():
    """Affiche les statistiques du dataset"""
    raw_dir = Path(config.RAW_DATA_DIR)
    
    if not raw_dir.exists():
        print(f"Repertoire {raw_dir} non trouve")
        return
    
    classes = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    
    if len(classes) == 0:
        print("Aucune classe trouvee")
        return
    
    stats = {}
    for class_name in classes:
        class_dir = raw_dir / class_name
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        stats[class_name] = len(image_files)
    
    # Graphique en barres
    plt.figure(figsize=(12, 6))
    plt.bar(stats.keys(), stats.values(), color='steelblue', alpha=0.7)
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'echantillons')
    plt.title('Distribution des Echantillons par Classe')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for class_name, count in stats.items():
        plt.text(class_name, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les statistiques textuelles
    print("\n" + "="*60)
    print("STATISTIQUES DU DATASET")
    print("="*60)
    total = sum(stats.values())
    print(f"Nombre total d'echantillons: {total}")
    print(f"Nombre de classes: {len(classes)}")
    print(f"Moyenne par classe: {total/len(classes):.1f}")
    print("\nPar classe:")
    for class_name, count in stats.items():
        percentage = (count / total) * 100
        print(f"  {class_name:20s}: {count:4d} ({percentage:5.1f}%)")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualiser les donnees collectees')
    parser.add_argument('--class', type=str, default=None,
                       help='Nom de la classe a visualiser')
    parser.add_argument('--stats', action='store_true',
                       help='Afficher les statistiques du dataset')
    parser.add_argument('--samples', type=int, default=9,
                       help='Nombre d\'echantillons a afficher')
    
    args = parser.parse_args()
    
    if args.stats:
        visualize_dataset_stats()
    elif args.__dict__.get('class'):
        visualize_samples(args.__dict__['class'], args.samples)
    else:
        print("Utilisation:")
        print("  python utils/visualize_data.py --stats")
        print("  python utils/visualize_data.py --class open_palm --samples 9")

