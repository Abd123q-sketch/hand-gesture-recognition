"""
Module de collecte de données pour la reconnaissance de gestes de la main
Utilise OpenCV pour capturer des images via webcam
"""
import cv2
import os
import sys
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class DataCollector:
    """Classe pour collecter des données de gestes via webcam"""
    
    def __init__(self):
        self.raw_data_dir = Path(config.RAW_DATA_DIR)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.current_class = None
        self.sample_count = 0
        self.cap = None
        
    def initialize_camera(self):
        """Initialise la caméra"""
        self.cap = cv2.VideoCapture(config.COLLECTION_SETTINGS["webcam_index"])
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la webcam")
        return True
    
    def create_class_directory(self, class_name):
        """Crée un répertoire pour une classe de geste"""
        class_dir = self.raw_data_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        return class_dir
    
    def collect_samples(self, class_name, num_samples=None):
        """
        Collecte des échantillons pour une classe donnée
        
        Args:
            class_name: Nom de la classe de geste
            num_samples: Nombre d'échantillons à collecter (par défaut depuis config)
        """
        if num_samples is None:
            num_samples = config.COLLECTION_SETTINGS["samples_per_class"]
        
        if not self.cap:
            self.initialize_camera()
        
        class_dir = self.create_class_directory(class_name)
        self.current_class = class_name
        self.sample_count = 0
        
        print(f"\n{'='*60}")
        print(f"Collecte de données pour la classe: {class_name}")
        print(f"Nombre d'échantillons à collecter: {num_samples}")
        print(f"{'='*60}")
        print("Instructions:")
        print("- Appuyez sur ESPACE pour capturer une image")
        print("- Appuyez sur 'q' pour quitter")
        print("- Assurez-vous que votre main est bien visible")
        print(f"{'='*60}\n")
        
        while self.sample_count < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                print("Erreur: Impossible de lire depuis la webcam")
                break
            
            # Miroir horizontal pour une meilleure UX
            frame = cv2.flip(frame, 1)
            
            # Afficher les informations
            info_text = f"Classe: {class_name} | Echantillon: {self.sample_count}/{num_samples}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "ESPACE: Capturer | Q: Quitter", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Dessiner un rectangle pour guider l'utilisateur
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 2
            cv2.rectangle(frame, 
                         (center_x - size, center_y - size),
                         (center_x + size, center_y + size),
                         (0, 255, 0), 2)
            
            cv2.imshow("Collecte de Donnees - Hand Gesture Recognition", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Espace pour capturer
                # Sauvegarder l'image
                img_path = class_dir / f"{self.sample_count:04d}.jpg"
                cv2.imwrite(str(img_path), frame)
                self.sample_count += 1
                print(f"Image {self.sample_count}/{num_samples} sauvegardee: {img_path}")
                
                # Feedback visuel
                cv2.putText(frame, "IMAGE CAPTUREE!", (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("Collecte de Donnees - Hand Gesture Recognition", frame)
                cv2.waitKey(300)  # Pause de 300ms
                
            elif key == ord('q'):
                print("Collecte interrompue par l'utilisateur")
                break
        
        print(f"\nCollecte terminee pour {class_name}: {self.sample_count} echantillons")
        return self.sample_count
    
    def collect_all_classes(self):
        """Collecte des données pour toutes les classes"""
        print("\n" + "="*60)
        print("COLLECTE DE DONNEES - TOUTES LES CLASSES")
        print("="*60)
        print(f"Classes disponibles: {config.GESTURE_CLASSES}")
        print("="*60 + "\n")
        
        if not self.cap:
            self.initialize_camera()
        
        results = {}
        for class_name in config.GESTURE_CLASSES:
            response = input(f"Collecter des donnees pour '{class_name}'? (o/n): ")
            if response.lower() == 'o':
                num_samples = input(f"Nombre d'echantillons (defaut: {config.COLLECTION_SETTINGS['samples_per_class']}): ")
                num_samples = int(num_samples) if num_samples.isdigit() else None
                count = self.collect_samples(class_name, num_samples)
                results[class_name] = count
            else:
                print(f"Classe {class_name} ignoree")
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        return results
    
    def release(self):
        """Libère les ressources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Fonction principale pour la collecte de données"""
    collector = DataCollector()
    
    try:
        print("Initialisation de la webcam...")
        collector.initialize_camera()
        
        print("\nOptions:")
        print("1. Collecter pour toutes les classes")
        print("2. Collecter pour une classe specifique")
        choice = input("Votre choix (1 ou 2): ")
        
        if choice == "1":
            collector.collect_all_classes()
        elif choice == "2":
            print(f"\nClasses disponibles: {config.GESTURE_CLASSES}")
            class_name = input("Nom de la classe: ")
            if class_name in config.GESTURE_CLASSES:
                num_samples = input(f"Nombre d'echantillons (defaut: {config.COLLECTION_SETTINGS['samples_per_class']}): ")
                num_samples = int(num_samples) if num_samples.isdigit() else None
                collector.collect_samples(class_name, num_samples)
            else:
                print(f"Classe '{class_name}' non reconnue")
        else:
            print("Choix invalide")
            
    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        collector.release()


if __name__ == "__main__":
    main()

