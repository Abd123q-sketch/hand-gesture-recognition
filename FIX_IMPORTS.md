# Correction des Imports

## Problème Résolu

Les scripts dans `src/` ne pouvaient pas être exécutés directement depuis le dossier `src/` car ils ne trouvaient pas le module `config` à la racine du projet.

## Solution Appliquée

Tous les fichiers dans `src/` et `utils/` ont été modifiés pour ajouter automatiquement le répertoire parent au `sys.path` avant d'importer `config`.

### Fichiers Modifiés

- ✅ `src/data_collection.py`
- ✅ `src/preprocessing.py`
- ✅ `src/train.py`
- ✅ `src/evaluate.py`
- ✅ `src/inference.py`
- ✅ `src/models.py`
- ✅ `utils/visualize_data.py`

### Code Ajouté

Chaque fichier contient maintenant ce code avant l'import de `config` :

```python
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
```

## Utilisation

Maintenant, vous pouvez exécuter les scripts de deux façons :

### Option 1 : Depuis la racine (recommandé)
```bash
python src/data_collection.py
python src/preprocessing.py
python src/train.py --model cnn
```

### Option 2 : Depuis le dossier src/
```bash
cd src
python data_collection.py
python preprocessing.py
python train.py --model cnn
```

Les deux méthodes fonctionnent maintenant correctement ! 🎉

