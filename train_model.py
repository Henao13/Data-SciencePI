# train_model.py

import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor

from app.pipeline import (
    agregar_features,
    COLS_TO_DROP,
    create_train_test_split,
    aplicar_codificacion_train,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COL = "tiempo_total_min"


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "data" / "produccion.csv")
    return df


def build_dataset() -> tuple:
    df_model = load_raw_data()

    df_model = agregar_features(df_model)
    df_model = df_model.drop(columns=COLS_TO_DROP, errors="ignore")

    strat_train_set, strat_test_set = create_train_test_split(
        df_model, test_size=0.2, stratify_column=TARGET_COL
    )

    X_train, y_train, X_test, y_test, metadata = aplicar_codificacion_train(
        strat_train_set,
        strat_test_set,
        target_col=TARGET_COL,
        alpha=5.0,
    )

    return X_train, y_train, X_test, y_test, metadata


def train_and_save():
    X_train, y_train, X_test, y_test, metadata = build_dataset()
    print(X_train)

    model = ExtraTreesRegressor(
        n_estimators=200,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        max_depth=30,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    bundle = {
        "model": model,
        "metadata": metadata,
    }

    out_path = MODELS_DIR / "model_bundle.pkl"
    joblib.dump(bundle, out_path)
    print(f"Modelo y metadata guardados en {out_path}")


if __name__ == "__main__":
    train_and_save()

