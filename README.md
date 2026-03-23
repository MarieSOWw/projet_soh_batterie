Prédiction du State of Health (SoH) des Batteries Li-ion
Deep Learning avec LSTM
 Objectif du projet:

Ce projet vise à prédire le State of Health (SoH) d’une batterie lithium-ion à partir de mesures temporelles issues de cycles de charge/décharge.

Le SoH représente le niveau de dégradation de la batterie (100% = état neuf).
Sa prédiction est un enjeu clé pour :

la maintenance prédictive,

la gestion énergétique,

les véhicules électriques.

L’approche utilisée repose sur un modèle LSTM (Long Short-Term Memory), adapté aux données séquentielles.

 Approche générale:

L’idée principale du projet est la suivante :

Transformer les données brutes en séquences temporelles (fenêtres glissantes), puis apprendre un modèle capable de prédire le SoH à partir de ces séquences.

Le pipeline complet est :

Analyse exploratoire (EDA)

Nettoyage des données

Construction de séquences temporelles

Split par batterie (anti data leakage)

Normalisation

Baselines (Ridge, MLP)

Modèle LSTM

Évaluation multi-niveaux (fenêtre + cycle + batterie)

📁 Structure du projet
projet_soh_batterie/
├── data/
│   ├── battery_health_dataset.csv
│   └── prepared_data.pkl
├── models/
│   └── best_model.keras
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
├── 01_exploration.ipynb
├── 02_preparation.ipynb
├── 03_modele_lstm.ipynb
├── 04_evaluation.ipynb
└── README.md
⚙️ Installation
pip install pandas numpy matplotlib scikit-learn tensorflow jupyter

Versions testées :

Python 3.10.11

TensorFlow 2.21

scikit-learn ≥ 1.3

▶️ Exécution

Exécuter les notebooks dans l’ordre :

01_exploration → 02_preparation → 03_modele_lstm → 04_evaluation

Chaque étape produit des artefacts utilisés par la suivante (prepared_data.pkl, modèle entraîné, etc.).

 Données
Variable	Description
battery_id	Identifiant batterie
cycle_number	Numéro du cycle
Voltage_measured	Tension (V)
Current_measured	Courant (A)
Temperature_measured	Température (°C)
SoC	State of Charge (%)
SoH	State of Health (%) — cible

Dataset :

24 batteries

29 180 lignes

1 459 cycles (battery × cycle)

0 valeur manquante

⚙️ Méthodologie
Nettoyage

Suppression des anomalies :

SoH > 100% (140 lignes)

Température > 60°C (160 lignes)

Impact total : ~1% du dataset

Features utilisées

Voltage

Current

Temperature

SoC

cycle_number

   cycle_number agit comme un proxy temporel du vieillissement.

Fenêtres glissantes

Chaque cycle contient 20 points temporels.
On construit des séquences de longueur :

WINDOW = 10

→ soit 50% d’un cycle

Chaque séquence est associée au SoH du cycle.

Split (point critique)

Le split est réalisé par batterie entière :

Train : 18 batteries

Validation : 3 batteries

Test : 3 batteries

→ Aucun data leakage

C’est un point fondamental pour garantir une évaluation réaliste.

Normalisation

X : MinMaxScaler

y : StandardScaler

Fit uniquement sur le train

🧪 Baselines
1. Ridge Regression

Features agrégées (mean, std, min, max)

2. MLP (sans mémoire)

Dense 64 → 32 → 16 → 1

Permet de vérifier si la temporalité apporte un gain

🤖 Modèle LSTM
Input (batch, 10, 5)
 → LSTM(64) + Dropout(0.30)
 → BatchNormalization
 → LSTM(32) + Dropout(0.25)
 → Dense(16)
 → Dense(1)

~31k paramètres

Optimizer : Adam

Loss : MSE

Metric : MAE

EarlyStopping + ReduceLROnPlateau

📈 Résultats
Comparaison des modèles
Modèle	MAE (%)	RMSE (%)	R²
Ridge	4.3354	5.0274	0.3634
MLP	2.9219	3.6215	0.6696
LSTM ★	2.4679	2.8835	0.7906
➜ Interprétation

Le LSTM améliore fortement Ridge (+1.87% MAE)

Le LSTM bat le MLP (+0.45% MAE)
➡️ La mémoire temporelle apporte un gain réel

📊 Évaluation à deux niveaux
Pourquoi ?

Avec stride=1, un cycle génère 11 fenêtres très similaires → biais potentiel.

Résultats
Niveau	N	MAE	RMSE	R²
Fenêtre	2156	2.4679	2.8835	0.7906
Cycle (référence) ★	196	2.4541	2.8535	0.7949

➡️ L’évaluation cycle est plus représentative.

🔍 Analyse par batterie
Batterie	MAE	RMSE	R²
B0005	2.72	3.02	0.828
B0029	2.55	2.98	0.120
B0039	1.43	2.13	0.305
Analyse

Le R² global est dominé par B0005 (beaucoup de données)

Le R² moyen = 0.418 → meilleure mesure de généralisation

Cas intéressant

B0039 présente une chute brutale du SoH non vue en train.

➡️ Le modèle prédit une décroissance lisse
➡️ Limite normale du supervised learning

📉 Surapprentissage

Meilleure époque : 4

Ratio val/train : 0.68

➡️ Pas de surapprentissage significatif

 Limites

Généralisation

Peu de batteries test

profils atypiques mal capturés

Variable cycle_number

facilite la prédiction

mais réduit l’interprétabilité physique

Un seul split

résultats dépendants des batteries choisies

Fenêtres corrélées

biais potentiel au niveau fenêtre

 Conclusion

Le projet montre que :

Le LSTM permet une prédiction précise du SoH (~2.45% MAE)

La temporalité apporte une réelle valeur

Le pipeline est robuste et sans data leakage

Cependant :

Les performances doivent être confirmées avec plus de batteries et une validation groupée.

Perspectives

À partir de l’analyse du modèle et de recherches complémentaires que j'ai effectuee , plusieurs pistes d’amélioration peuvent être envisagées :

Modèles plus avancés
Explorer des architectures récentes adaptées aux séries temporelles, comme :

les Transformers (meilleure capture des dépendances longues),

les Temporal CNN (TCN), souvent plus stables et rapides à entraîner que les LSTM.

Data augmentation ciblée
Augmenter le jeu d’entraînement en générant des profils rares ou atypiques (ex : dégradations brutales), afin d’améliorer la capacité du modèle à généraliser sur des batteries non vues.

 Remarque finale

Ce projet constitue une base solide pour la prédiction du SoH, avec une approche rigoureuse et des résultats cohérents.
