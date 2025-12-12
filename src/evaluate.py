"""
Script d'évaluation des modèles de reconnaissance de gestes
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Ajouter le répertoire parent au path pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.models import get_model


def load_model_and_data(model_path, data_dir=None):
    """
    Charge un modèle et les données de test
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
        data_dir: Répertoire contenant les données
        
    Returns:
        model, (X_test, y_test), class_mapping
    """
    model_path = Path(model_path)
    
    # Charger le modèle
    print(f"Chargement du modele: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    
    # Déterminer le répertoire de données
    if data_dir is None:
        landmarks_dir = Path(config.LANDMARKS_DATA_DIR)
        processed_dir = Path(config.PROCESSED_DATA_DIR)
        
        if landmarks_dir.exists() and (landmarks_dir / "X_test.npy").exists():
            data_dir = landmarks_dir
        elif processed_dir.exists() and (processed_dir / "X_test.npy").exists():
            data_dir = processed_dir
        else:
            raise FileNotFoundError("Aucune donnee de test trouvee")
    else:
        data_dir = Path(data_dir)
    
    # Charger les données de test
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    # Charger le mapping des classes
    with open(data_dir / "class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    
    print(f"Donnees de test chargees: {X_test.shape}")
    print(f"Classes: {class_mapping}")
    
    return model, (X_test, y_test), class_mapping


def evaluate_model(model, X_test, y_test, class_mapping, save_dir=None):
    """
    Évalue un modèle et génère des rapports détaillés
    
    Args:
        model: Modèle Keras
        X_test: Données de test
        y_test: Labels de test
        class_mapping: Mapping des classes
        save_dir: Répertoire pour sauvegarder les résultats
    """
    print("\n" + "="*60)
    print("EVALUATION DU MODELE")
    print("="*60)
    
    # Prédictions
    print("Generation des predictions...")
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Métriques globales
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Rapport de classification
    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    print("\n" + "="*60)
    print("RAPPORT DE CLASSIFICATION")
    print("="*60)
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualisations
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Matrice de confusion
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Matrice de Confusion')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe predite')
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300)
        print(f"\nMatrice de confusion sauvegardee: {save_dir / 'confusion_matrix.png'}")
        plt.close()
        
        # Métriques par classe
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        metrics_df = {
            'class': class_names,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': np.sum(cm, axis=1).tolist()
        }
        
        # Graphique des métriques par classe
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(class_names))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classe')
        ax.set_ylabel('Score')
        ax.set_title('Metriques par Classe')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_per_class.png', dpi=300)
        print(f"Metriques par classe sauvegardees: {save_dir / 'metrics_per_class.png'}")
        plt.close()
        
        # Sauvegarder le rapport textuel
        with open(save_dir / 'classification_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("RAPPORT D'EVALUATION\n")
            f.write("="*60 + "\n\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n\n")
            f.write(report)
            f.write("\n\n" + "="*60 + "\n")
            f.write("METRIQUES PAR CLASSE\n")
            f.write("="*60 + "\n")
            for i, name in enumerate(class_names):
                f.write(f"\n{name}:\n")
                f.write(f"  Precision: {precision[i]:.4f}\n")
                f.write(f"  Recall: {recall[i]:.4f}\n")
                f.write(f"  F1-Score: {f1[i]:.4f}\n")
                f.write(f"  Support: {metrics_df['support'][i]}\n")
        
        print(f"Rapport textuel sauvegarde: {save_dir / 'classification_report.txt'}")
    
    return {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'true_labels': y_test.tolist()
    }


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluation du modele de reconnaissance de gestes')
    parser.add_argument('--model', type=str, required=True,
                       help='Chemin vers le modele sauvegarde (.h5)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Repertoire contenant les donnees de test')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Repertoire pour sauvegarder les resultats')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(config.LOGS_DIR) / "evaluation"
    
    # Charger le modèle et les données
    model, (X_test, y_test), class_mapping = load_model_and_data(
        args.model, args.data_dir
    )
    
    # Évaluer
    results = evaluate_model(model, X_test, y_test, class_mapping, args.output_dir)
    
    print("\n" + "="*60)
    print("EVALUATION TERMINEE")
    print("="*60)


if __name__ == "__main__":
    main()

