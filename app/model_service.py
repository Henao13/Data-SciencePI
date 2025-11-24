# app/model_service.py

import io
import joblib
import pandas as pd
import boto3

from .pipeline import transformar_nueva_muestra

S3_BUCKET = "proyecto-ultra"
S3_KEY = "modelsRepository/model_bundle_slim.pkl"  # o model_bundle_slim.pkl si ya lo adelgazaste

s3 = boto3.client("s3")

_model = None
_metadata = None


def _load_model_and_metadata_from_s3():
    """
    Descarga model_bundle.pkl desde S3 y carga:
      - model
      - metadata

    Usando BytesIO para que joblib tenga un archivo seekable.
    """
    global _model, _metadata

    if _model is not None and _metadata is not None:
        return

    print(f"Loading model bundle from s3://{S3_BUCKET}/{S3_KEY} ...")
    response = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)

    # Leer todos los bytes del objeto
    data = response["Body"].read()

    # Envolver en BytesIO (joblib necesita un objeto con .read() y .seek())
    with io.BytesIO(data) as f:
        bundle = joblib.load(f)

    _model = bundle["model"]
    _metadata = bundle["metadata"]
    del bundle

    print("Model and metadata loaded successfully from S3.")


def preprocess_input(raw_data: dict) -> pd.DataFrame:
    _load_model_and_metadata_from_s3()

    df = pd.DataFrame([raw_data])
    X = transformar_nueva_muestra(df, _metadata)
    return X


def predict_single(raw_data: dict) -> float:
    _load_model_and_metadata_from_s3()
    X = preprocess_input(raw_data)
    y_pred = _model.predict(X)
    return float(y_pred[0])
