# Prédiction du State of Health (SoH) des Batteries Li-ion
## Projet Deep Learning — LSTM

---

## Structure du projet

```
projet_soh_batterie/
├── data/
│   ├── battery_health_dataset.csv     ← Dataset (29 180 lignes × 7 colonnes)
│   └── prepared_data.pkl              ← Pipeline sérialisé (généré par 02)
├── models/
│   └── best_model.keras               ← Meilleurs poids LSTM (généré par 03)
├── figures/
│   ├── fig_cycles_distribution.png
│   ├── fig_soh_evolution.png
│   ├── fig_distributions.png
│   ├── fig_correlation.png
│   ├── fig_learning_curves.png
│   ├── fig_evaluation_global.png
│   ├── fig_soh_per_battery.png
│   ├── fig_b0029_analysis.png
│   └── fig_b0039_analysis.png
├── 01_exploration.ipynb               ← EDA + 10 questions de réflexion de l'énoncé
├── 02_preparation.ipynb               ← Nettoyage + Fenêtres + Split + Normalisation
├── 03_modele_lstm.ipynb               ← Baseline Ridge + Baseline MLP + LSTM + Entraînement
├── 04_evaluation.ipynb                ← Métriques fenêtre + cycle + Analyse par batterie
└── README.md
```

---

## Prérequis et installation

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow jupyter
```

**Versions testées :** Python 3.10.11 · TensorFlow 2.21.0 · scikit-learn ≥ 1.3

---

## Exécution dans l'ordre

```
01_exploration → 02_preparation → 03_modele_lstm → 04_evaluation
```

**Important :** chaque notebook lit `data/prepared_data.pkl` produit par le notebook précédent.  
Le modèle entraîné est sauvegardé dans `models/best_model.keras` et rechargé par `04_evaluation`.

---

## Données

| Variable | Description |
|----------|-------------|
| `battery_id` | Identifiant de la batterie |
| `cycle_number` | Numéro du cycle charge/décharge (1 → 197) |
| `Voltage_measured` | Tension mesurée (V) |
| `Current_measured` | Courant mesuré (A) |
| `Temperature_measured` | Température (°C) |
| `SoC` | État de charge (%) |
| `SoH` | **État de santé (%) — variable cible** |

**24 batteries · 29 180 lignes · 1 459 combinaisons (batterie × cycle) · 0 valeur manquante**

> Note : `cycle_number` va de 1 à 197 (max global). Les 1 459 cycles distincts correspondent
> aux combinaisons uniques `(battery_id, cycle_number)` — chaque batterie ayant son propre compteur.

---

## Méthodologie

| Étape | Implémentation |
|-------|----------------|
| Nettoyage | SoH > 100% supprimés (140 lignes) + Temp > 60°C supprimés (160 lignes) — 1% du dataset |
| Features | Tension, Courant, Température, SoC, Numéro de cycle |
| Fenêtre glissante | WINDOW = 10 bins (50% du cycle de 20 bins — validé par comparaison 5/10/15) |
| Split | Par **batterie entière** : 18 train / 3 val / 3 test — aucun data leakage |
| Normalisation X | `MinMaxScaler` (fit sur train uniquement) |
| Normalisation y | `StandardScaler` (fit sur train uniquement) |
| Baseline 1 | Ridge Regression sur features agrégées (mean, std, min, max) |
| Baseline 2 | MLP (Dense 64→32→16→1) sur séquences aplaties — valide l'apport du LSTM |
| Modèle | LSTM(64) → Dropout(0.30) → BatchNorm → LSTM(32) → Dropout(0.25) → Dense(16) → Dense(1) |
| Entraînement | Adam lr=0.001 · batch=64 · EarlyStopping patience=8 · ReduceLROnPlateau |

---

## Architecture LSTM

```
Input (batch, 10, 5)
  → LSTM(64, return_sequences=True) + Dropout(0.30)
  → BatchNormalization
  → LSTM(32, return_sequences=False) + Dropout(0.25)
  → Dense(16, relu)
  → Dense(1)                    ← SoH prédit (%)
Total : 31 137 paramètres
```

---

## Résultats

### Comparaison des modèles

| Métrique | LSTM ★ | MLP Baseline | Ridge Baseline |
|----------|--------|--------------|----------------|
| MAE (%)  | **2.45** | 2.86 | 4.34 |
| RMSE (%) | **3.12** | 3.61 | 5.03 |
| R² global | **0.755** | 0.672 | 0.363 |

> Le LSTM surpasse le MLP de **+0.41% MAE**, confirmant que la mémoire temporelle
> apporte un gain réel au-delà de la simple lecture des valeurs instantanées.
> Le LSTM améliore Ridge de **+1.89% MAE** — la complexité du modèle est justifiée.

### Évaluation à deux niveaux

| Niveau | N | MAE (%) | RMSE (%) | R² |
|--------|---|---------|----------|----|
| Fenêtre (brut) | 2 156 | 2.4503 | 3.1169 | 0.7553 |
| **Cycle (honnête) ★** | **196** | **2.4474** | **3.0875** | **0.7599** |

> Avec stride=1 et WINDOW=10, chaque cycle génère 11 fenêtres quasi-identiques.
> L'évaluation **au niveau cycle** (prédiction moyennée par cycle) est l'indicateur de référence,
> car elle évite de compter 11 fois le même cycle.

### Résultats par batterie de test

| Batterie | Cycles | MAE (%) | RMSE (%) | R² | Statut |
|----------|--------|---------|----------|----|--------|
| B0005 | 123 | 2.61 | 3.30 | 0.796 | ✅ Bon |
| B0029 | 40 | 2.58 | 3.02 | 0.093 | 🔴 Faible |
| B0039 | 33 | 1.70 | 2.46 | 0.071 | 🔴 Faible |

> Le R² global (0.755) est dominé par B0005 qui représente 63% des fenêtres de test.
> Le R² moyen non pondéré (0.320) reflète mieux la généralisation réelle sur batteries inconnues.

B0039 présente un profil "plateau + chute brutale" (SoH : 87.24% → 80.43%, variation -6.81%)
absent du jeu d'entraînement. Le LSTM reproduit une décroissance progressive qu'il a apprise,
mais ne peut pas anticiper un comportement qu'il n'a jamais vu.
**Ce n'est pas un bug — c'est une limite fondamentale de l'apprentissage supervisé.**

### Surapprentissage

| Indicateur | Valeur |
|-----------|--------|
| Époques réalisées | 11 |
| Meilleure val_loss | 0.2642 @ époque 3 |
| Train loss @ époque 3 | 0.3529 |
| **Ratio val/train (best)** | **0.75** |

→ **Pas de surapprentissage significatif** (ratio < 1.5).  
`restore_best_weights=True` garantit l'utilisation des poids de l'époque 3.

---

## Limites identifiées

1. **Généralisation** : profils atypiques non représentés en entraînement (B0039, R²=0.071) — solution : diversifier le jeu d'entraînement avec des profils "plateau+chute"
2. **R² global pondéré** : dominé par B0005 (63% des données de test) — préférer le R² moyen non pondéré (0.320) pour évaluer la robustesse réelle
3. **`cycle_number` comme feature** : introduit un proxy temporel — le modèle peut apprendre "cycle élevé = SoH bas" au lieu des signatures électrochimiques pures
4. **Fenêtres chevauchantes** : stride=1 génère 11 fenêtres autocorrélées par cycle — les métriques au niveau fenêtre surestiment légèrement les performances réelles

---

## Métriques cibles

- LSTM > Ridge → ✅ amélioration +1.89% MAE (+52% de réduction d'erreur)
- LSTM > MLP → ✅ la séquentialité apporte un gain mesurable (+0.41% MAE)
- Pas de surapprentissage → ✅ ratio val/train = 0.75 à la meilleure époque
- 86.2% des prédictions avec erreur < 5% → ✅ utilisable en contexte industriel