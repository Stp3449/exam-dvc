import pandas as pd
import numpy as np
from joblib import load
import json
from pathlib import Path
import click
import logging

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DATA = Path("data")
PROCESSED = DATA / "processed"
MODEL = Path("models/trained_model.pkl")
METRICS = Path("metrics")


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()