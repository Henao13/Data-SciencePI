# app/model_service.py

import os
import joblib
import pandas as pd

from .pipeline import transformar_nueva_muestra

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_bundle.pkl")

_bundle = joblib.load(BUNDLE_PATH)
_model = _bundle["model"]
_metadata = _bundle["metadata"]


def preprocess_input(raw_data: dict) -> pd.DataFrame:
    """
    raw_data: dict con las columnas crudas como las recibes por la API.
    """
    df = pd.DataFrame([raw_data])
    X = transformar_nueva_muestra(df, _metadata)
    return X


def predict_single(raw_data: dict) -> float:
    X = preprocess_input(raw_data)
    y_pred = _model.predict(X)
    return float(y_pred[0])
