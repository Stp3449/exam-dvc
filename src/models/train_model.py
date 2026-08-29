import pandas as pd
from pathlib import Path
import click
import logging
import joblib
import numpy as np
import os
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor


DATA = Path("data/processed")
MODELS = Path("models")


# ============================================================
# CONFIGURATION MLFLOW / DAGSHUB  (optionnelle)
# ============================================================
# Rien n'est ecrit en dur : l'URI et les identifiants sont lus depuis un
# fichier .env local (non versionne) ou depuis les variables d'environnement
# du shell. Voir .env.example.
#   MLFLOW_TRACKING_URI       -> https://dagshub.com/<USER>/<REPO>.mlflow
#   MLFLOW_TRACKING_USERNAME  -> nom d'utilisateur DagsHub
#   MLFLOW_TRACKING_PASSWORD  -> token DagsHub (Settings > Tokens)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logging.info("python-dotenv absent : lecture des variables d'environnement du shell.")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_ENABLED = bool(
    MLFLOW_TRACKING_URI
    and os.environ.get("MLFLOW_TRACKING_USERNAME")
    and os.environ.get("MLFLOW_TRACKING_PASSWORD")
)

if MLFLOW_ENABLED:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("exam-dvc")
else:
    logging.warning(
        "MLflow desactive : definir MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / "
        "MLFLOW_TRACKING_PASSWORD (dans .env) pour logger vers DagsHub. "
        "Le pipeline s'execute normalement sans."
    )


@click.command()
def main():
    """Train the regression model using the best parameters."""

    # --------------------------------------------------------
    # Chargement des données
    # --------------------------------------------------------
    X_train = pd.read_csv(DATA / "X_train_scaled.csv")
    y_train = np.ravel(pd.read_csv(DATA / "y_train.csv"))

    # --------------------------------------------------------
    # Chargement des meilleurs paramètres
    # --------------------------------------------------------
    best_params = joblib.load(MODELS / "best_params.pkl")
    print(f"Best parameters: {best_params}")

    # --------------------------------------------------------
    # Entraînement + sauvegarde locale (sortie DVC)
    # Cette partie ne dépend PAS de MLflow : `dvc repro` aboutit
    # même sans accès à DagsHub.
    # --------------------------------------------------------
    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    MODELS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / "trained_model.pkl"
    joblib.dump(model, model_path)
    print("Model trained and saved successfully.")
    print(f"Model file: {model_path}")

    # --------------------------------------------------------
    # Logging MLflow / DagsHub — optionnel et best effort :
    # ni l'absence d'identifiants ni un échec réseau n'interrompt le pipeline.
    # --------------------------------------------------------
    if not MLFLOW_ENABLED:
        return

    try:
        with mlflow.start_run(run_name="random_forest_training"):
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("random_state", 42)
            mlflow.log_param("n_jobs", -1)
            mlflow.sklearn.log_model(sk_model=model, name="trained_model")

            run_id = mlflow.active_run().info.run_id
            print(f"MLflow Run ID: {run_id}")
    except Exception as exc:  # noqa: BLE001
        logging.warning("Logging MLflow ignore (%s) : %s", type(exc).__name__, exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()