import pandas as pd
from pathlib import Path
import click
import logging
from sklearn.preprocessing import StandardScaler
import joblib

DATA = Path("data/processed")

@click.command()
def main():
    X_train = pd.read_csv(DATA / "X_train.csv")
    X_test = pd.read_csv(DATA / "X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    X_train_scaled.to_csv(DATA / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(DATA / "X_test_scaled.csv", index=False)

    joblib.dump(scaler, DATA / "scaler.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()