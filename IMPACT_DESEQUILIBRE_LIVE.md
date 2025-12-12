# Impact du Déséquilibre sur les Prédictions en Temps Réel

## 🎯 Réponse Courte : **OUI, ABSOLUMENT !**

Le déséquilibre des classes affecte **directement** les prédictions en temps réel. Voici pourquoi :

## 🔍 Comment ça fonctionne en temps réel

### 1. Le Modèle Utilise les Probabilités Apprises

Quand vous faites une prédiction en live (ligne 185 de `inference.py`) :

```python
pred_proba = self.model.predict(seq, verbose=0)
probabilities_array = pred_proba[0]  # Ex: [0.65, 0.15, 0.10, 0.10]
```

Le modèle retourne des **probabilités** pour chaque classe. Ces probabilités sont **biaisées** par l'entraînement.

### 2. Exemple Concret du Problème

**Situation actuelle (déséquilibré) :**
- `open_palm`: 500 échantillons (70%)
- `fist`: 100 échantillons (14%)
- `victory`: 100 échantillons (14%)
- `thumbs_up`: 20 échantillons (2%)

**Ce que le modèle apprend :**
- "Si je ne suis pas sûr, je prédits `open_palm` car c'est statistiquement plus probable"
- Les probabilités sont biaisées : `[0.65, 0.15, 0.10, 0.10]` même pour un geste `fist`

**En temps réel, quand vous faites un `fist` :**
```
Probabilités prédites :
- open_palm: 0.65 (65%) ❌ FAUX mais le modèle choisit ça !
- fist: 0.15 (15%) ✅ CORRECT mais ignoré
- victory: 0.10 (10%)
- thumbs_up: 0.10 (10%)
```

Le modèle prédit **`open_palm`** alors que vous faites un **`fist`** !

### 3. Le Code de Prédiction (ligne 199)

```python
smoothed_class_id = np.argmax(smoothed_probs)  # Prend la classe avec la probabilité max
```

`argmax` choisit toujours la classe avec la probabilité la plus élevée. Si `open_palm` a toujours la probabilité la plus élevée (même faiblement), elle sera toujours choisie.

## 📊 Impact Visuel en Temps Réel

Quand vous testez en live, vous voyez probablement :

```
┌─────────────────────────────────┐
│ open_palm (65%) [HIGH]          │ ← Toujours prédit
│                                 │
│ Probabilités:                   │
│ ████████████ open_palm: 65%     │
│ ███ fist: 15%                   │ ← Votre vrai geste
│ ██ victory: 10%                 │
│ ██ thumbs_up: 10%               │
└─────────────────────────────────┘
```

Même si vous faites un `fist`, le modèle prédit `open_palm` !

## ⚠️ Pourquoi le Lissage N'aide Pas

Le lissage des prédictions (lignes 188-196) **moyenne** les probabilités, mais si elles sont toutes biaisées vers `open_palm`, la moyenne sera aussi biaisée :

```python
# Si les 7 dernières prédictions sont toutes biaisées :
prob_history = [
    [0.65, 0.15, 0.10, 0.10],  # open_palm
    [0.60, 0.20, 0.10, 0.10],  # open_palm
    [0.70, 0.12, 0.10, 0.08],  # open_palm
    ...
]
# La moyenne sera toujours biaisée vers open_palm !
```

## ✅ Solution : Équilibrer les Données

Après équilibrage (ex: 150 échantillons par classe), le modèle apprend :

**Probabilités équilibrées pour un `fist` :**
```
- open_palm: 0.10 (10%)
- fist: 0.75 (75%) ✅ CORRECT et choisi !
- victory: 0.10 (10%)
- thumbs_up: 0.05 (5%)
```

## 🎯 Résumé

| Aspect | Avant Équilibrage | Après Équilibrage |
|--------|-------------------|-------------------|
| **Prédiction pour `fist`** | `open_palm` (65%) ❌ | `fist` (75%) ✅ |
| **Confiance** | Haute mais fausse | Haute et correcte |
| **Biais** | Fort vers `open_palm` | Équilibré |
| **Précision live** | ~30-40% | ~80-90% |

## 💡 Conclusion

**OUI, le déséquilibre affecte directement les prédictions en temps réel** car :

1. ✅ Le modèle a appris un biais vers la classe majoritaire
2. ✅ Les probabilités sont systématiquement biaisées
3. ✅ `argmax` choisit toujours la classe avec la probabilité max (même si fausse)
4. ✅ Le lissage ne corrige pas le biais, il le moyenne seulement

**La solution :** Équilibrer le dataset et ré-entraîner le modèle !

