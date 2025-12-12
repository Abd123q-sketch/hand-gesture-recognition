"""
Script de démarrage rapide pour le projet de reconnaissance de gestes
Guide interactif pour les nouveaux utilisateurs
"""
import os
import sys
from pathlib import Path


def print_header(text):
    """Affiche un en-tête stylisé"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def check_installation():
    """Vérifie que les dépendances sont installées"""
    print_header("VERIFICATION DE L'INSTALLATION")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'mediapipe': 'mediapipe',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package:20s} - Installe")
        except ImportError:
            print(f"✗ {package:20s} - MANQUANT")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Packages manquants: {', '.join(missing)}")
        print("Installez-les avec: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ Toutes les dependances sont installees!")
        return True


def check_data():
    """Vérifie si des données existent"""
    print_header("VERIFICATION DES DONNEES")
    
    raw_dir = Path("data/raw")
    processed_dir = Path("data/landmarks")
    
    if raw_dir.exists():
        classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
        if classes:
            print(f"✓ Donnees brutes trouvees: {len(classes)} classes")
            for cls in classes:
                count = len(list((raw_dir / cls).glob("*.jpg")))
                print(f"  - {cls}: {count} images")
        else:
            print("⚠️  Repertoire data/raw existe mais aucune classe trouvee")
    else:
        print("⚠️  Aucune donnee brute trouvee")
        print("   Executez: python src/data_collection.py")
    
    if processed_dir.exists() and (processed_dir / "X_train.npy").exists():
        print("\n✓ Donnees pretraitees trouvees")
        return True
    else:
        print("\n⚠️  Aucune donnee pretraitee trouvee")
        print("   Executez: python src/preprocessing.py")
        return False


def check_models():
    """Vérifie si des modèles existent"""
    print_header("VERIFICATION DES MODELES")
    
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.h5"))
        if model_files:
            print(f"✓ {len(model_files)} modele(s) trouve(s):")
            for model_file in model_files:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"  - {model_file.name} ({size_mb:.1f} MB)")
            return True
        else:
            print("⚠️  Aucun modele trouve")
            print("   Executez: python src/train.py")
            return False
    else:
        print("⚠️  Repertoire models/ n'existe pas")
        print("   Executez: python src/train.py")
        return False


def show_next_steps():
    """Affiche les prochaines étapes recommandées"""
    print_header("PROCHAINES ETAPES RECOMMANDEES")
    
    steps = []
    
    # Vérifier les données
    raw_dir = Path("data/raw")
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        steps.append(("1. Collecter des donnees", "python src/data_collection.py"))
    
    # Vérifier le prétraitement
    processed_dir = Path("data/landmarks")
    if not (processed_dir.exists() and (processed_dir / "X_train.npy").exists()):
        steps.append(("2. Pretraiter les donnees", "python src/preprocessing.py"))
    
    # Vérifier les modèles
    models_dir = Path("models")
    if not models_dir.exists() or not list(models_dir.glob("*.h5")):
        steps.append(("3. Entrainer un modele", "python src/train.py --model auto"))
    
    # Toujours disponible
    steps.append(("4. Tester en temps reel", "python src/inference.py --model models/cnn_best.h5"))
    
    if steps:
        for i, (description, command) in enumerate(steps, 1):
            print(f"{i}. {description}")
            print(f"   Commande: {command}\n")
    else:
        print("✓ Tout est pret! Vous pouvez tester votre modele:")
        print("  python src/inference.py --model models/cnn_best.h5")


def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("  GUIDE DE DEMARRAGE RAPIDE")
    print("  Hand Gesture Recognition Project")
    print("="*60)
    
    # Vérifications
    installed = check_installation()
    if not installed:
        print("\n⚠️  Veuillez installer les dependances avant de continuer")
        return
    
    has_data = check_data()
    has_models = check_models()
    
    # Afficher les prochaines étapes
    show_next_steps()
    
    # Menu interactif
    print_header("MENU INTERACTIF")
    print("1. Collecter des donnees")
    print("2. Pretraiter les donnees")
    print("3. Entrainer un modele")
    print("4. Evaluer un modele")
    print("5. Tester en temps reel")
    print("6. Visualiser les donnees")
    print("0. Quitter")
    
    choice = input("\nVotre choix: ").strip()
    
    commands = {
        '1': 'python src/data_collection.py',
        '2': 'python src/preprocessing.py',
        '3': 'python src/train.py --model auto',
        '4': 'python src/evaluate.py --model models/cnn_best.h5',
        '5': 'python src/inference.py --model models/cnn_best.h5',
        '6': 'python utils/visualize_data.py --stats',
    }
    
    if choice in commands:
        print(f"\nExecution: {commands[choice]}")
        os.system(commands[choice])
    elif choice == '0':
        print("\nAu revoir!")
    else:
        print("\nChoix invalide")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterruption par l'utilisateur")
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()

