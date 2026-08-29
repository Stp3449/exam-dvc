import pandas as pd
import numpy as np
from joblib import load
import json
import os
from pathlib import Path
import click
import logging

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DATA = Path("data")
PROCESSED = DATA / "processed"
MODEL = Path("models/trained_model.pkl")
METRICS = Path("metrics")


# MLflow optionnel : identique a train_model.py, lu depuis .env / le shell.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MLFLOW_ENABLED = bool(
    os.environ.get("MLFLOW_TRACKING_URI")
    and os.environ.get("MLFLOW_TRACKING_USERNAME")
    and os.environ.get("MLFLOW_TRACKING_PASSWORD")
)


@click.command()
def main():
    """Evaluate the trained regression model and save predictions and metrics."""

    X_test = pd.read_csv(PROCESSED / "X_test_scaled.csv")
    y_test = np.ravel(pd.read_csv(PROCESSED / "y_test.csv"))

    model = load(MODEL)

    predictions = model.predict(X_test)

    metrics = calculate_metrics(y_test, predictions)

    save_predictions(predictions)
    save_metrics(metrics)
    log_metrics_mlflow(metrics)


def calculate_metrics(y_test, predictions):

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def save_predictions(predictions):

    DATA.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "prediction": predictions
    }).to_csv(
        DATA / "predictions.csv",
        index=False
    )


def save_metrics(metrics):

    METRICS.mkdir(parents=True, exist_ok=True)

    (METRICS / "scores.json").write_text(
        json.dumps(metrics, indent=4)
    )


def log_metrics_mlflow(metrics):
    """Logue les metriques dans MLflow / DagsHub (best effort)."""
    if not MLFLOW_ENABLED:
        raise Exception("MyFlow not enabled : define MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (in .env) to log to DagsHub.")
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("exam-dvc")
        with mlflow.start_run(run_name="random_forest_evaluation"):
            mlflow.log_metrics(metrics)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Logging MLflow ignore (%s) : %s", type(exc).__name__, exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()