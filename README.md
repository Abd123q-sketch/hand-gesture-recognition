# Projet de Reconnaissance des Gestes de la Main (Hand Gesture Recognition)

## 📋 Table des Matières

1. [Description du Projet](#description-du-projet)
2. [Gestes Reconnaissables](#gestes-reconnaissables)
3. [Objectifs Pédagogiques](#objectifs-pédagogiques)
4. [Architecture du Projet](#architecture-du-projet)
5. [Installation](#installation)
6. [Utilisation](#utilisation)
7. [Structure des Données](#structure-des-données)
8. [Modèles Disponibles](#modèles-disponibles)
9. [Workflow Complet](#workflow-complet)
10. [Résultats et Évaluation](#résultats-et-évaluation)
11. [Dépannage](#dépannage)
12. [Contributions](#contributions)
13. [Licence](#licence)

---

## 🎯 Description du Projet

Ce projet implémente un système complet de reconnaissance de gestes de la main en temps réel, capable d'identifier différents gestes tels que :
- **Paume ouverte** (`open_palm`)
- **Poing fermé** (`closed_fist`)
- **Signe de victoire** (`victory`)
- **Pouce levé** (`thumbs_up`)
- **Signe OK** (`okay`)
- **Pointage avec index** (`pointing`)

Le système utilise des techniques de deep learning (CNN et CNN+LSTM) combinées avec MediaPipe pour le suivi des mains et OpenCV pour le traitement d'images.

---

## ✋ Gestes Reconnaissables

Voici les gestes que le système peut reconnaître :

| Geste | Emoji | Nom Technique | Description |
|-------|-------|---------------|-------------|
| **Paume ouverte** | ✋ | `open_palm` | Main ouverte avec tous les doigts étendus |
| **Poing fermé** | ✊ | `closed_fist` | Main fermée en poing, tous les doigts repliés |
| **Signe de victoire** | ✌️ | `victory` | Index et majeur levés en forme de V (peace sign) |
| **Pouce levé** | 👍 | `thumbs_up` | Pouce levé, autres doigts fermés |
| **Signe OK** | 👌 | `okay` | Pouce et index formant un cercle, autres doigts levés |
| **Pointage** | 👉 | `pointing` | Index pointé, autres doigts repliés |

### 📸 Exemples Visuels

```
✋ Paume ouverte          ✊ Poing fermé           ✌️ Signe de victoire
   ┌─────────┐              ┌─────────┐              ┌─────────┐
   │  ╱│╲    │              │  ╱│╲    │              │  ╱│╲    │
   │ ╱ │ ╲   │              │ ╱ │ ╲   │              │ ╱ │ ╲   │
   ││  │  │  │              ││  │  │  │              ││  │  │  │
   ││  │  │  │              ││  │  │  │              ││  │  │  │
   └─────────┘              └─────────┘              └─────────┘

👍 Pouce levé            👌 Signe OK             👉 Pointage
   ┌─────────┐              ┌─────────┐              ┌─────────┐
   │  ╱│╲    │              │  ╱│╲    │              │  ╱│╲    │
   │ ╱ │ ╲   │              │ ╱ │ ╲   │              │ ╱ │ ╲   │
   ││  │  │  │              ││  │  │  │              ││  │  │  │
   ││  │  │  │              ││  │  │  │              ││  │  │  │
   └─────────┘              └─────────┘              └─────────┘
```

### 💡 Conseils pour la Collecte

- **✋ Paume ouverte** : Main bien ouverte, doigts espacés, paume face à la caméra
- **✊ Poing fermé** : Poing serré, pouce par-dessus les doigts
- **✌️ Signe de victoire** : Index et majeur bien séparés, autres doigts repliés
- **👍 Pouce levé** : Pouce bien visible et levé, main fermée
- **👌 Signe OK** : Cercle formé par pouce et index, autres doigts relevés
- **👉 Pointage** : Index bien tendu, autres doigts repliés, pouce sur le côté

**Astuce** : Variez les angles, distances et éclairages lors de la collecte pour améliorer la robustesse du modèle !

---

## 🎓 Objectifs Pédagogiques

Ce projet permet d'apprendre :

1. **Collecte et annotation de données** : Capture de données personnalisées via webcam avec OpenCV
2. **Prétraitement d'images** : Utilisation d'OpenCV pour le traitement et MediaPipe pour l'extraction de landmarks
3. **Classification d'images** : Implémentation de modèles CNN pour la classification statique
4. **Classification de séquences** : Utilisation de LSTM pour inclure la dimension temporelle
5. **Inférence en temps réel** : Mise en œuvre d'un système de reconnaissance en temps réel

---

## 🏗️ Architecture du Projet

```
mon_projet_deeplearning/
│
├── README.md                 # Documentation complète (ce fichier)
├── requirements.txt          # Dépendances Python
├── config.py                 # Configuration centralisée
├── .gitignore               # Fichiers à ignorer par Git
│
├── data/                    # Données du projet
│   ├── raw/                 # Images brutes capturées
│   │   ├── open_palm/
│   │   ├── closed_fist/
│   │   └── ...
│   ├── processed/           # Images prétraitées (si mode image)
│   └── landmarks/           # Landmarks extraits (si mode landmarks)
│
├── models/                  # Modèles sauvegardés
│   ├── cnn_best.h5
│   ├── cnn_lstm_best.h5
│   └── ...
│
├── logs/                    # Logs d'entraînement et évaluations
│   └── evaluation/
│
└── src/                     # Code source
    ├── __init__.py
    ├── data_collection.py   # Collecte de données via webcam
    ├── preprocessing.py     # Prétraitement et extraction de features
    ├── models.py            # Définitions des modèles (CNN, CNN+LSTM)
    ├── train.py             # Script d'entraînement
    ├── evaluate.py          # Script d'évaluation
    └── inference.py         # Inférence en temps réel
```

---

## 🔧 Installation

### Prérequis

- Python 3.8 ou supérieur
- Webcam fonctionnelle
- Système d'exploitation : Windows, Linux ou macOS

### Installation des Dépendances

1. **Cloner ou télécharger le projet**

2. **Créer un environnement virtuel** (recommandé) :

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **Installer les dépendances** :

```bash
pip install -r requirements.txt
```

### Vérification de l'Installation

Pour vérifier que tout est correctement installé :

```bash
python -c "import cv2, tensorflow, mediapipe; print('Installation reussie!')"
```

---

## 🚀 Utilisation

### 1. Collecte de Données

La première étape consiste à collecter vos propres données de gestes via la webcam.

**Important** : Exécutez les scripts depuis la racine du projet pour que les imports fonctionnent correctement.

#### Collecte pour une classe spécifique :

```bash
# Depuis la racine du projet
python src/data_collection.py
```

Suivez les instructions à l'écran :
- Choisissez l'option 2 pour une classe spécifique
- Entrez le nom de la classe (ex: `open_palm`)
- Positionnez votre main dans le rectangle vert
- Appuyez sur **ESPACE** pour capturer une image
- Appuyez sur **Q** pour quitter

#### Collecte pour toutes les classes :

```bash
python src/data_collection.py
```

Choisissez l'option 1 et suivez les instructions pour chaque classe.

**Conseils pour une bonne collecte** :
- Assurez-vous d'avoir un bon éclairage
- Variez les angles et positions de votre main
- Capturez environ 200 échantillons par classe
- Maintenez une distance constante avec la caméra

### 2. Prétraitement des Données

Une fois les données collectées, il faut les prétraiter :

```bash
# Depuis la racine du projet
python src/preprocessing.py
```

Ce script :
- Extrait les landmarks MediaPipe de chaque image (ou prétraite les images)
- Normalise les données
- Divise le dataset en train/validation/test (70%/15%/15%)
- Sauvegarde les données prétraitées dans `data/landmarks/` ou `data/processed/`

**Options de prétraitement** (modifiables dans `config.py`) :
- **Mode Landmarks** : Extrait uniquement les 21 points MediaPipe (plus rapide, moins de données)
- **Mode Images** : Utilise les images complètes (plus de données, meilleure précision potentielle)

### 3. Entraînement du Modèle

#### Entraînement avec CNN simple :

```bash
# Depuis la racine du projet
python src/train.py --model cnn
```

#### Entraînement avec CNN+LSTM (dimension temporelle) :

```bash
python src/train.py --model cnn --lstm
```

#### Entraînement avec Landmarks :

```bash
python src/train.py --model landmark
```

#### Entraînement avec Landmarks+LSTM :

```bash
python src/train.py --model landmark --lstm
```

#### Mode automatique (détection du type de données) :

```bash
python src/train.py --model auto
```

**Paramètres d'entraînement** (modifiables dans `config.py`) :
- `batch_size` : Taille des batches (défaut: 32)
- `epochs` : Nombre d'époques (défaut: 50)
- `learning_rate` : Taux d'apprentissage (défaut: 0.001)
- `sequence_length` : Longueur des séquences pour LSTM (défaut: 21)

**Callbacks automatiques** :
- **Early Stopping** : Arrêt si pas d'amélioration pendant 10 époques
- **Reduce LR on Plateau** : Réduction du learning rate si stagnation
- **Model Checkpoint** : Sauvegarde du meilleur modèle
- **TensorBoard** : Logs pour visualisation (dans `logs/`)

### 4. Évaluation du Modèle

Pour évaluer un modèle entraîné :

```bash
python src/evaluate.py --model models/cnn_best.h5
```

Ce script génère :
- Rapport de classification détaillé
- Matrice de confusion
- Graphiques des métriques par classe
- Fichiers sauvegardés dans `logs/evaluation/`

**Options** :
```bash
python src/evaluate.py --model models/cnn_best.h5 --data_dir data/landmarks --output_dir logs/my_evaluation
```

### 5. Inférence en Temps Réel

Pour tester le modèle en temps réel avec votre webcam :

```bash
python src/inference.py --model models/cnn_best.h5
```

**Pour un modèle LSTM** :
```bash
python src/inference.py --model models/cnn_lstm_best.h5 --lstm
```

**Pour un modèle Landmarks** :
```bash
python src/inference.py --model models/landmark_best.h5 --landmarks True
```

**Contrôles** :
- **Q** : Quitter l'application
- La prédiction s'affiche en haut à gauche
- Les probabilités de toutes les classes s'affichent en barres

---

## 📊 Structure des Données

### Données Brutes (`data/raw/`)

```
data/raw/
├── open_palm/
│   ├── 0000.jpg
│   ├── 0001.jpg
│   └── ...
├── closed_fist/
│   └── ...
└── ...
```

### Données Prétraitées (`data/landmarks/` ou `data/processed/`)

- `X_train.npy` : Données d'entraînement
- `y_train.npy` : Labels d'entraînement
- `X_val.npy` : Données de validation
- `y_val.npy` : Labels de validation
- `X_test.npy` : Données de test
- `y_test.npy` : Labels de test
- `class_mapping.json` : Mapping des classes

### Format des Landmarks

Si vous utilisez le mode landmarks, chaque échantillon est un vecteur de 63 valeurs (21 points × 3 coordonnées x, y, z).

### Format des Images

Si vous utilisez le mode images, chaque échantillon est une image de taille 64×64×3 (RGB).

---

## 🤖 Modèles Disponibles

### 1. CNN Simple (`build_cnn_model`)

Architecture :
- 4 couches de convolution (32, 64, 128, 128 filtres)
- MaxPooling et BatchNormalization après chaque couche
- Couches fully connected (512, 256 neurones)
- Dropout pour la régularisation
- Softmax pour la classification

**Utilisation** : Classification d'images statiques

### 2. CNN+LSTM (`build_cnn_lstm_model`)

Architecture :
- CNN pour extraire les features de chaque frame
- 2 couches LSTM (128, 64 neurones) pour la dimension temporelle
- Couches fully connected
- Softmax pour la classification

**Utilisation** : Classification de séquences temporelles (gestes dynamiques)

### 3. Landmark Model (`build_landmark_model`)

Architecture :
- Réseau de neurones dense (128, 256, 128, 64 neurones)
- BatchNormalization et Dropout
- Softmax pour la classification

**Utilisation** : Classification basée sur les landmarks MediaPipe (plus rapide)

### 4. Landmark+LSTM (`build_landmark_lstm_model`)

Architecture :
- 3 couches LSTM (128, 64, 32 neurones)
- Couches fully connected
- Softmax pour la classification

**Utilisation** : Classification de séquences de landmarks (rapide avec dimension temporelle)

---

## 🔄 Workflow Complet

Voici le workflow recommandé pour utiliser ce projet :

```
1. Installation
   └── pip install -r requirements.txt

2. Collecte de Données
   └── python src/data_collection.py
       └── Capturer ~200 images par classe

3. Prétraitement
   └── python src/preprocessing.py
       └── Génère les données prétraitées

4. Entraînement
   └── python src/train.py --model cnn
       └── Génère le modèle dans models/

5. Évaluation
   └── python src/evaluate.py --model models/cnn_best.h5
       └── Génère les rapports dans logs/evaluation/

6. Inférence Temps Réel
   └── python src/inference.py --model models/cnn_best.h5
       └── Test en temps réel avec webcam
```

---

## 📈 Résultats et Évaluation

### Métriques Disponibles

Le script `evaluate.py` génère plusieurs métriques :

1. **Accuracy globale** : Précision sur l'ensemble du test set
2. **Precision par classe** : Précision pour chaque geste
3. **Recall par classe** : Rappel pour chaque geste
4. **F1-Score par classe** : Score F1 pour chaque geste
5. **Matrice de confusion** : Visualisation des erreurs de classification

### Visualisations Générées

- `confusion_matrix.png` : Matrice de confusion colorée
- `metrics_per_class.png` : Graphique des métriques par classe
- `classification_report.txt` : Rapport textuel détaillé

### Amélioration des Performances

Pour améliorer les performances :

1. **Plus de données** : Collecter plus d'échantillons par classe
2. **Augmentation de données** : Rotation, translation, changement de luminosité
3. **Hyperparamètres** : Ajuster learning_rate, batch_size, architecture
4. **Transfer Learning** : Utiliser des modèles pré-entraînés (MobileNet, ResNet)
5. **Ensemble** : Combiner plusieurs modèles

---

## 🐛 Dépannage

### Problème : Webcam ne s'ouvre pas

**Solution** :
- Vérifiez que la webcam n'est pas utilisée par une autre application
- Modifiez `webcam_index` dans `config.py` (essayez 0, 1, 2...)
- Sur Linux, vérifiez les permissions : `sudo chmod 666 /dev/video0`

### Problème : Aucune main détectée

**Solution** :
- Assurez-vous d'avoir un bon éclairage
- Vérifiez que votre main est bien visible dans le cadre
- Ajustez `min_detection_confidence` dans `config.py`

### Problème : Erreur "Out of Memory"

**Solution** :
- Réduisez `batch_size` dans `config.py`
- Réduisez `image_size` dans `config.py`
- Utilisez le mode landmarks au lieu des images complètes

### Problème : Prédictions incorrectes

**Solution** :
- Vérifiez que les classes dans `config.py` correspondent à vos données
- Ré-entraînez avec plus de données
- Vérifiez que le prétraitement est identique entre entraînement et inférence
- Augmentez `confidence_threshold` dans `config.py`

### Problème : Modèle ne converge pas

**Solution** :
- Vérifiez que vous avez assez de données (minimum 50-100 par classe)
- Réduisez le `learning_rate`
- Augmentez le nombre d'époques
- Vérifiez que les données sont bien équilibrées entre les classes

### Problème : Import errors

**Solution** :
```bash
# Réinstaller les dépendances
pip install --upgrade -r requirements.txt

# Vérifier l'environnement virtuel
python --version
pip list
```

---

## ⚙️ Configuration Avancée

Tous les paramètres sont centralisés dans `config.py`. Voici les principaux :

### Collecte de Données

```python
COLLECTION_SETTINGS = {
    "samples_per_class": 200,      # Nombre d'échantillons par classe
    "image_size": (224, 224),       # Taille des images capturées
    "webcam_index": 0,              # Index de la webcam
}
```

### Prétraitement

```python
PREPROCESSING_SETTINGS = {
    "image_size": 64,               # Taille après redimensionnement
    "use_landmarks": True,          # Utiliser MediaPipe landmarks
    "normalize": True,              # Normaliser les valeurs
}
```

### Modèle

```python
MODEL_SETTINGS = {
    "sequence_length": 21,          # Longueur de séquence pour LSTM
    "batch_size": 32,               # Taille des batches
    "epochs": 50,                   # Nombre d'époques
    "learning_rate": 0.001,         # Taux d'apprentissage
}
```

### Inférence

```python
INFERENCE_SETTINGS = {
    "confidence_threshold": 0.5,   # Seuil de confiance minimum
    "smoothing_window": 5,         # Frames pour lissage
}
```

---

## 📚 Ressources et Références

### Documentation Officielle

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)

### Datasets Alternatifs

- **Sign Language MNIST** : Dataset de signes de la main
- **HaGRID** : Dataset de gestes de la main
- **EgoHands** : Dataset de gestes en première personne

### Articles et Tutoriels

- [Hand Gesture Recognition using Deep Learning](https://towardsdatascience.com/)
- [MediaPipe Hands Tutorial](https://google.github.io/mediapipe/solutions/hands.html)
- [CNN+LSTM for Video Classification](https://keras.io/examples/)

---

## 🎯 Améliorations Futures

Idées pour améliorer le projet :

1. **Support multi-mains** : Détecter et classifier plusieurs mains simultanément
2. **Gestes dynamiques** : Reconnaissance de gestes complexes avec mouvement
3. **Interface graphique** : Créer une GUI avec Tkinter ou PyQt
4. **API REST** : Exposer le modèle via une API Flask/FastAPI
5. **Application mobile** : Portage vers Android/iOS avec TensorFlow Lite
6. **Transfer Learning** : Utiliser MobileNet ou EfficientNet pré-entraînés
7. **Data Augmentation** : Rotation, translation, changement de couleur automatique
8. **Export ONNX** : Exporter le modèle pour utilisation dans d'autres frameworks

---

## 🤝 Contributions

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## 📝 Licence

Ce projet est fourni à des fins éducatives. Libre d'utilisation et de modification.

---

## 👤 Auteur

Projet développé dans le cadre d'un cours de Deep Learning.

**Contact** : Pour toute question ou suggestion, ouvrez une issue sur le repository.

---

## 🙏 Remerciements

- **MediaPipe** : Pour l'excellent outil de suivi des mains
- **TensorFlow/Keras** : Pour le framework de deep learning
- **OpenCV** : Pour le traitement d'images
- La communauté open-source pour les outils et ressources

---

## 📞 Support

Si vous rencontrez des problèmes :

1. Consultez la section [Dépannage](#dépannage)
2. Vérifiez les issues existantes sur le repository
3. Ouvrez une nouvelle issue avec :
   - Description du problème
   - Messages d'erreur complets
   - Configuration utilisée
   - Étapes pour reproduire

---

**Bon apprentissage et bon développement ! 🚀**

