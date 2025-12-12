# Solution au Problème de Déséquilibre des Classes

## 🔍 Problème Identifié

Vous avez détecté que la classe `open_palm` a beaucoup plus d'échantillons que les autres classes. C'est effectivement **le problème principal** qui cause les erreurs de prédiction !

### Pourquoi c'est un problème ?

1. **Biais vers la classe majoritaire** : Le modèle apprend à toujours prédire `open_palm` car il voit cette classe beaucoup plus souvent
2. **Sous-apprentissage des autres classes** : Les classes minoritaires ne sont pas assez représentées
3. **Métriques trompeuses** : Une haute précision globale peut masquer une mauvaise performance sur les classes minoritaires

## ✅ Solutions Implémentées

### Solution 1 : Équilibrage du Dataset (RECOMMANDÉ)

Un script a été créé pour équilibrer automatiquement vos données :

```bash
# Équilibrer au minimum (recommandé - réduit open_palm)
python utils/balance_dataset.py

# Équilibrer à un nombre spécifique
python utils/balance_dataset.py --target 200
```

**Ce que fait le script :**
- Analyse la distribution des classes
- Réduit les classes sur-représentées (comme `open_palm`)
- Sauvegarde les fichiers excédentaires dans `raw_backup/`
- Équilibre toutes les classes au même nombre

**Après équilibrage :**
```bash
# Re-prétraiter les données
python src/preprocessing.py

# Ré-entraîner le modèle
python src/train.py
```

### Solution 2 : Poids de Classe (Automatique)

Le script `train.py` calcule maintenant automatiquement des **poids de classe** pour compenser le déséquilibre :

- Les classes minoritaires reçoivent un poids plus élevé
- Les classes majoritaires reçoivent un poids plus faible
- Le modèle accorde plus d'importance aux classes sous-représentées

**Cela fonctionne automatiquement** - pas besoin de configuration supplémentaire !

### Solution 3 : Détection Automatique

Le script `preprocessing.py` détecte maintenant automatiquement les déséquilibres et vous avertit :

```
⚠️  DESEQUILIBRE DETECTE!
   Ratio max/min: 3.5x
   Minimum: 150, Maximum: 525
   Recommandation: Equilibrez le dataset avec:
   python utils/balance_dataset.py
```

## 📊 Vérifier la Distribution

Avant d'équilibrer, visualisez la distribution actuelle :

```bash
python utils/visualize_data.py --stats
```

## 🎯 Plan d'Action Recommandé

1. **Visualiser les données actuelles** :
   ```bash
   python utils/visualize_data.py --stats
   ```

2. **Équilibrer le dataset** :
   ```bash
   python utils/balance_dataset.py
   ```

3. **Re-prétraiter les données équilibrées** :
   ```bash
   python src/preprocessing.py
   ```

4. **Ré-entraîner le modèle** (avec poids de classe automatiques) :
   ```bash
   python src/train.py
   ```

5. **Tester en temps réel** :
   ```bash
   python src/inference.py --model models/cnn_lstm_best.h5
   ```

## 💡 Conseils Supplémentaires

- **Collecter plus de données** pour les classes minoritaires est la meilleure solution à long terme
- **Augmenter les classes minoritaires** avec duplication si nécessaire :
  ```bash
  python utils/balance_dataset.py --strategy upsample
  ```
- **Vérifier les métriques par classe** après entraînement avec `src/evaluate.py`

## 📈 Résultats Attendus

Après équilibrage, vous devriez voir :
- ✅ Meilleure précision sur toutes les classes
- ✅ Moins de biais vers `open_palm`
- ✅ Prédictions plus équilibrées en temps réel
- ✅ Métriques plus fiables

