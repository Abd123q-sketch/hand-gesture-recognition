"""
Script pour équilibrer le dataset en réduisant les classes sur-représentées
"""
import shutil
from pathlib import Path
import sys
import random

# Ajouter le répertoire parent au path pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def balance_dataset(target_samples=None, strategy='downsample'):
    """
    Équilibre le dataset en ajustant le nombre d'échantillons par classe
    
    Args:
        target_samples: Nombre cible d'échantillons par classe (None = utiliser le minimum)
        strategy: 'downsample' (réduire) ou 'upsample' (augmenter avec duplication)
    """
    raw_dir = Path(config.RAW_DATA_DIR)
    
    if not raw_dir.exists():
        print(f"❌ Repertoire {raw_dir} non trouve")
        return
    
    classes = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    
    if len(classes) == 0:
        print("❌ Aucune classe trouvee")
        return
    
    # Compter les échantillons par classe
    stats = {}
    for class_name in classes:
        class_dir = raw_dir / class_name
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        stats[class_name] = len(image_files)
    
    print("\n" + "="*60)
    print("STATISTIQUES AVANT EQUILIBRAGE")
    print("="*60)
    for class_name, count in stats.items():
        print(f"  {class_name:20s}: {count:4d} echantillons")
    
    # Déterminer le nombre cible
    if target_samples is None:
        target_samples = min(stats.values())
        print(f"\nNombre cible (minimum): {target_samples} echantillons par classe")
    else:
        print(f"\nNombre cible (specifie): {target_samples} echantillons par classe")
    
    # Créer un répertoire de sauvegarde pour les données excédentaires
    backup_dir = raw_dir.parent / "raw_backup"
    backup_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("EQUILIBRAGE DU DATASET")
    print("="*60)
    
    for class_name in classes:
        class_dir = raw_dir / class_name
        current_count = stats[class_name]
        
        if current_count == target_samples:
            print(f"✓ {class_name:20s}: Deja equilibre ({current_count})")
            continue
        
        image_files = sorted([f for f in class_dir.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if current_count > target_samples:
            # Réduire: sauvegarder les fichiers excédentaires
            print(f"↓ {class_name:20s}: {current_count} -> {target_samples} (reduction)")
            
            # Sauvegarder dans backup
            backup_class_dir = backup_dir / class_name
            backup_class_dir.mkdir(exist_ok=True)
            
            # Sélectionner aléatoirement les fichiers à garder
            random.seed(42)  # Pour la reproductibilité
            files_to_keep = random.sample(image_files, target_samples)
            files_to_backup = [f for f in image_files if f not in files_to_keep]
            
            # Déplacer les fichiers excédentaires vers backup
            for file_to_backup in files_to_backup:
                shutil.move(str(file_to_backup), str(backup_class_dir / file_to_backup.name))
            
            print(f"  {len(files_to_backup)} fichiers deplaces vers {backup_class_dir}")
            
        elif current_count < target_samples and strategy == 'upsample':
            # Augmenter: dupliquer des fichiers aléatoirement
            print(f"↑ {class_name:20s}: {current_count} -> {target_samples} (augmentation)")
            
            needed = target_samples - current_count
            random.seed(42)
            files_to_duplicate = random.choices(image_files, k=needed)
            
            for i, file_to_dup in enumerate(files_to_duplicate):
                new_name = f"{file_to_dup.stem}_dup{i:04d}{file_to_dup.suffix}"
                shutil.copy2(str(file_to_dup), str(class_dir / new_name))
            
            print(f"  {needed} fichiers dupliques")
    
    # Vérification finale
    print("\n" + "="*60)
    print("STATISTIQUES APRES EQUILIBRAGE")
    print("="*60)
    final_stats = {}
    for class_name in classes:
        class_dir = raw_dir / class_name
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        final_stats[class_name] = len(image_files)
        print(f"  {class_name:20s}: {final_stats[class_name]:4d} echantillons")
    
    print("\n✅ Equilibrage termine!")
    print(f"📁 Fichiers excédentaires sauvegardes dans: {backup_dir}")
    print("\n💡 Prochaine etape: Re-executez le preprocessing")
    print("   python src/preprocessing.py")


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Equilibrer le dataset en reduisant les classes sur-representees',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Equilibrer au minimum (recommandé)
  python utils/balance_dataset.py
  
  # Equilibrer à un nombre spécifique
  python utils/balance_dataset.py --target 200
  
  # Augmenter les classes minoritaires (duplication)
  python utils/balance_dataset.py --strategy upsample
        """
    )
    parser.add_argument('--target', type=int, default=None,
                       help='Nombre cible d\'echantillons par classe (defaut: minimum)')
    parser.add_argument('--strategy', type=str, default='downsample',
                       choices=['downsample', 'upsample'],
                       help='Strategie: reduire (downsample) ou augmenter (upsample)')
    
    args = parser.parse_args()
    
    balance_dataset(target_samples=args.target, strategy=args.strategy)


if __name__ == "__main__":
    main()

