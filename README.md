# Examen DVC & DagsHub

Pipeline de Machine Learning versionné avec **DVC**, données et modèles stockés
sur le **remote DagsHub (S3)**, suivi des expériences avec **MLflow** sur DagsHub.

Jeu de données : prédiction de la concentration en silice (`silica_concentrate`)
à partir de mesures d'un procédé de flottation.
Modèle : `RandomForestRegressor` (scikit-learn).

> Dans ce README, remplacer `<USER>` par votre utilisateur DagsHub et `<REPO>`
> par le nom de votre dépôt.

## Structure du dépôt

```text
exam_dvc
├── data
│   ├── raw_data
│   │   ├── raw.csv            # données brutes (suivi DVC : raw.csv.dvc)
│   │   └── raw.csv.dvc
│   ├── processed              # sorties DVC (train/test, versions scalées, scaler)
│   └── predictions.csv        # sortie DVC de l'étape d'évaluation
├── metrics
│   └── scores.json            # métriques (versionnées par Git, cache: false)
├── models
│   ├── best_params.pkl        # sortie DVC (GridSearchCV)
│   └── trained_model.pkl      # sortie DVC (modèle entraîné)
├── src
│   ├── data
│   │   ├── make_dataset.py    # split train/test
│   │   └── scaling.py         # normalisation (StandardScaler)
│   └── models
│       ├── grid_search.py     # recherche d'hyperparamètres
│       ├── train_model.py     # entraînement (+ logging MLflow optionnel)
│       └── evaluate_model.py  # prédictions + métriques
├── dvc.yaml / dvc.lock        # pipeline DVC
├── .env.example               # modèle de configuration des accès (à copier en .env)
└── requirements.txt
```

## Pipeline DVC

| Étape        | Commande                              | Sorties                                              |
|--------------|---------------------------------------|-----------------------------------------------------|
| `split`      | `python src/data/make_dataset.py`     | `data/processed/{X,y}_{train,test}.csv`             |
| `normalize`  | `python src/data/scaling.py`          | `X_{train,test}_scaled.csv`, `scaler.pkl`          |
| `gridsearch` | `python src/models/grid_search.py`    | `models/best_params.pkl`                            |
| `training`   | `python src/models/train_model.py`    | `models/trained_model.pkl` (+ run MLflow si activé) |
| `evaluate`   | `python src/models/evaluate_model.py` | `data/predictions.csv`, `metrics/scores.json`       |

## 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configuration des accès DagsHub

Le token se génère sur DagsHub : **Settings → Tokens** (utiliser l'*Access Token*).
**Aucun identifiant n'est versionné.**

### MLflow (fichier `.env`)

```bash
cp .env.example .env
# éditer .env : renseigner MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME,
#               MLFLOW_TRACKING_PASSWORD (= token)
```

`.env` est ignoré par Git et chargé automatiquement par `train_model.py`
(via `python-dotenv`). Sans lui, le pipeline s'exécute quand même : le logging
MLflow est simplement désactivé.

### Remote DVC S3 (identifiants en local uniquement)

```bash
dvc remote modify origin endpointurl https://dagshub.com/<USER>/<REPO>.s3
dvc remote modify --local origin access_key_id     <token_dagshub>
dvc remote modify --local origin secret_access_key <token_dagshub>
```

`--local` écrit dans `.dvc/config.local`, ignoré par Git.

## 3. Utilisation

```bash
# Récupérer données et modèles versionnés depuis DagsHub
dvc pull

# (Ré)exécuter le pipeline
dvc repro

# Publier les nouvelles versions
dvc push
git add dvc.lock metrics/scores.json data/raw_data/raw.csv.dvc
git commit -m "Update pipeline"
git push

# Consulter les métriques
dvc metrics show
```

### Récupérer les données brutes (première fois)

```bash
mkdir -p data/raw_data
curl -o data/raw_data/raw.csv https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv
dvc add data/raw_data/raw.csv
git add data/raw_data/raw.csv.dvc data/raw_data/.gitignore
```

## Rendu

Le rendu est le lien vers le dépôt sur **DagsHub**. Ajouter
`https://dagshub.com/licence.pedago` comme collaborateur en **lecture seule**
(*Settings → Collaborators*).
