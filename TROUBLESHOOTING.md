# Guide de Dépannage - Hand Gesture Recognition

## Problème : FileNotFoundError lors de l'entraînement

### Erreur typique
```
FileNotFoundError: [Errno 2] No such file or directory: 
'.../data/processed/X_train.npy'
```

### Solutions

#### Solution 1 : Utiliser les Landmarks (Recommandé si disponibles)

Si vous avez des données de landmarks dans `data/landmarks/`, utilisez-les :

```bash
# Pour un modèle simple avec landmarks
python src/train.py --model landmark

# Pour un modèle LSTM avec landmarks
python src/train.py --model landmark --lstm
```

#### Solution 2 : Créer des Données d'Images

Si vous voulez utiliser un modèle CNN avec des images :

1. **Collecter des données** (si pas déjà fait) :
```bash
python src/data_collection.py
```

2. **Prétraiter les données** :
```bash
python src/preprocessing.py
```

3. **Entraîner le modèle** :
```bash
# CNN simple
python src/train.py --model cnn

# CNN + LSTM
python src/train.py --model cnn --lstm
```

#### Solution 3 : Utiliser les Anciennes Données (Fallback)

Le système détecte automatiquement les données dans `dataset/` si elles n'existent pas dans `data/`. 

Si vous avez des données dans `dataset/`, elles seront utilisées automatiquement avec un message d'avertissement.

### Vérification des Données Disponibles

Pour vérifier quelles données sont disponibles :

```bash
# Vérifier les landmarks
ls data/landmarks/
# ou
ls dataset/landmarks/

# Vérifier les images prétraitées
ls data/processed/
# ou
ls dataset/
```

### Types de Modèles et Données Requises

| Modèle | Type de Données | Commande |
|--------|----------------|----------|
| CNN | Images prétraitées | `python src/train.py --model cnn` |
| CNN+LSTM | Images prétraitées | `python src/train.py --model cnn --lstm` |
| Landmark | Landmarks MediaPipe | `python src/train.py --model landmark` |
| Landmark+LSTM | Landmarks MediaPipe | `python src/train.py --model landmark --lstm` |

### Messages d'Erreur Améliorés

Le système affiche maintenant des messages d'erreur plus clairs :

- ✅ **Si des landmarks existent** : Suggère d'utiliser `--model landmark`
- ✅ **Si aucune donnée** : Donne les étapes pour créer des données
- ✅ **Fallback automatique** : Utilise `dataset/` si `data/` n'existe pas

### Exemple de Workflow Complet

```bash
# 1. Vérifier les données disponibles
python quick_start.py

# 2a. Si landmarks disponibles, entraîner directement
python src/train.py --model landmark --lstm

# 2b. Sinon, créer des données d'abord
python src/data_collection.py
python src/preprocessing.py
python src/train.py --model cnn --lstm
```

---

## Autres Problèmes Courants

### Problème : "No module named 'config'"

**Solution** : Exécutez depuis la racine du projet :
```bash
# Depuis la racine
python src/train.py --model landmark
```

Ou utilisez le chemin complet depuis n'importe où.

### Problème : "class_mapping.json not found"

**Solution** : Le système crée automatiquement un mapping par défaut basé sur `config.py`.

### Problème : Erreurs de shape avec LSTM

**Solution** : Assurez-vous d'avoir assez de données. Les modèles LSTM nécessitent des séquences complètes. Minimum recommandé : 50-100 échantillons par classe.

---

Pour plus d'aide, consultez le [README.md](README.md) principal.

