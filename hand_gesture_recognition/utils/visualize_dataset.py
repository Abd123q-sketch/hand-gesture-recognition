"""
Script pour visualiser la structure actuelle du dataset.
Affiche le nombre d'images par dossier et par classe.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DATASET_PATH = Path(__file__).resolve().parents[2] / "dataset"

def count_images(folder_path):
    """Compte le nombre d'images dans un dossier."""
    if not folder_path.exists():
        return 0
    return len([f for f in folder_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

def visualize_dataset():
    print("=== VISUALISATION DU DATASET ===\n")
    
    # Analyser tous les dossiers principaux
    main_folders = ['raw', 'train', 'val', 'test']
    dataset_stats = {}
    
    for folder in main_folders:
        folder_path = DATASET_PATH / folder
        if folder_path.exists():
            print(f"--- Dossier {folder.upper()} ---")
            
            # Compter les images par classe
            classes = {}
            total_images = 0
            
            for class_dir in folder_path.iterdir():
                if class_dir.is_dir():
                    count = count_images(class_dir)
                    classes[class_dir.name] = count
                    total_images += count
                    print(f"  {class_dir.name}: {count} images")
            
            print(f"  Total: {total_images} images\n")
            dataset_stats[folder] = classes
        else:
            print(f"--- Dossier {folder.upper()}: NON PRÉSENT ---\n")
    
    # Créer un graphique de comparaison
    if dataset_stats:
        create_comparison_chart(dataset_stats)
    
    # Résumé
    print("=== RÉSUMÉ ===")
    for folder, classes in dataset_stats.items():
        if classes:
            print(f"{folder.upper()}: {len(classes)} classes, {sum(classes.values())} images totales")

def create_comparison_chart(dataset_stats):
    """Crée un graphique comparatif des datasets."""
    try:
        # Préparer les données
        all_classes = set()
        for classes in dataset_stats.values():
            all_classes.update(classes.keys())
        
        if not all_classes:
            return
        
        all_classes = sorted(list(all_classes))
        folders = list(dataset_stats.keys())
        
        # Créer la matrice de données
        data = []
        for folder in folders:
            row = []
            for class_name in all_classes:
                count = dataset_stats[folder].get(class_name, 0)
                row.append(count)
            data.append(row)
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(all_classes))
        width = 0.8 / len(folders)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (folder, color) in enumerate(zip(folders, colors)):
            ax.bar(x + i * width, data[i], width, label=folder.upper(), color=color)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Nombre d\'images')
        ax.set_title('Distribution des images par classe et par dossier')
        ax.set_xticks(x + width * (len(folders) - 1) / 2)
        ax.set_xticklabels(all_classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(DATASET_PATH / 'dataset_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\nGraphique sauvegardé dans: {DATASET_PATH / 'dataset_visualization.png'}")
        
        # Afficher le graphique si matplotlib est disponible
        try:
            plt.show()
        except:
            print("(Impossible d'afficher le graphique, mais il a été sauvegardé)")
            
    except Exception as e:
        print(f"Erreur lors de la création du graphique: {e}")

if __name__ == "__main__":
    visualize_dataset()
