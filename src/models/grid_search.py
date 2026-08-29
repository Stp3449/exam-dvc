import pandas as pd
from pathlib import Path
import click
import logging
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

DATA = Path("data/processed")
MODELS = Path("models")


@click.command()
def main():
    """Find the best hyperparameters using GridSearchCV."""

    X_train = pd.read_csv(DATA / "X_train_scaled.csv")
    y_train = pd.read_csv(DATA / "y_train.csv").squeeze()

    model = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid_search.best_params_, MODELS / "best_params.pkl")

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()