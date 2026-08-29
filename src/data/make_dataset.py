import pandas as pd
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split

INPUT = Path("data/raw_data/raw.csv")
OUTPUT = Path("data/processed")

@click.command()
def main():
    df = pd.read_csv(INPUT, sep=",")

    cols = [
        "ave_flot_air_flow", "ave_flot_level", "iron_feed",
        "starch_flow", "amina_flow", "ore_pulp_flow",
        "ore_pulp_pH", "ore_pulp_density", "silica_concentrate"
    ]

    for col in cols:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    X = df.drop(columns=["silica_concentrate", "date"])
    y = df["silica_concentrate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(OUTPUT / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT / "y_test.csv", index=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()