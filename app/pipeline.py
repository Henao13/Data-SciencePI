# app/pipeline.py

import numpy as np
import pandas as pd

COLS_TO_DROP = [
    "TiempoAlistamiento",
    "TiempoMaquina",
    "TiempoAplique",
    "TiempoPulida",
    "TiempoAlistamiento_min",
    "TiempoMaquina_min",
    "TiempoAplique_min",
    "TiempoPulida_min",
    "TiempoenMaquina",
    "FechaInicio",
    "FechaFin",
    "MaterialIdOrden",
    "HiloInferiorId",
    "CantidadMala",
    "CantidadReprocesada",
    "CantidadProgramada",
    "CantidadOrden",
    "FechaInicio_dt",
]

RAW_COLS_TO_DROP = [
    "Plastico",
    "MaquinaPlasticoId",
    "TiempoQuitarPlastico",
    "Canutillo",
    "Chenille",
    "Cordon",
    "Presion",
    "Temperatura",
    "Unnamed: 37",
    "Velocidad",
    "NumeroApliques",
    "InterlonId",
    "DisenosTPTG",
    "Ubicacion",
    "TipoMaterialId",
    "MaterialId",
    "PedidoId",
    "NumeroOrden",
]


DROP_VALUE_PLANTILLA = "dejar arriba hacia abajo 11cm"

TE_COLS = ["Operario", "NombreMaquina", "codigoMaterial"]


# =========================================================
# 0) FEATURE ENGINEERING
# =========================================================
def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -------------------------------------------------
    # A) Drop de columnas crudas (lo que hacías al inicio del notebook)
    # -------------------------------------------------
    df = df.drop(columns=RAW_COLS_TO_DROP, errors="ignore")


    # Fechas a datetime (FechaInicio, FechaFin)
    for col in ["FechaInicio", "FechaFin"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Tiempos hh:mm:ss → minutos
    cols_hhmm = ["TiempoAlistamiento", "TiempoMaquina", "TiempoAplique", "TiempoPulida"]
    for c in cols_hhmm:
        if c in df.columns:
            # Si ya es numérico, asumimos que está en minutos
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c + "_min"] = df[c].fillna(0)
            else:
                df[c + "_min"] = (
                    pd.to_timedelta(df[c], errors="coerce")
                      .dt.total_seconds() / 60
                )

    # tiempo_total_min (solo si no existe aún)
    # tiempo_total_min (solo si no existe aún)
    if "tiempo_total_min" not in df.columns:
        # Siempre devolvemos una Serie (nunca un int)
        if "TiempoenMaquina" in df.columns:
            s_tiempo = df["TiempoenMaquina"].fillna(0)
        else:
            s_tiempo = pd.Series(0, index=df.index, dtype="float64")

        if "TiempoAlistamiento_min" in df.columns:
            s_alist = df["TiempoAlistamiento_min"].fillna(0)
        else:
            s_alist = pd.Series(0, index=df.index, dtype="float64")

        if "TiempoMaquina_min" in df.columns:
            s_maq = df["TiempoMaquina_min"].fillna(0)
        else:
            s_maq = pd.Series(0, index=df.index, dtype="float64")

        if "TiempoAplique_min" in df.columns:
            s_apl = df["TiempoAplique_min"].fillna(0)
        else:
            s_apl = pd.Series(0, index=df.index, dtype="float64")

        if "TiempoPulida_min" in df.columns:
            s_pul = df["TiempoPulida_min"].fillna(0)
        else:
            s_pul = pd.Series(0, index=df.index, dtype="float64")

        df["tiempo_total_min"] = s_tiempo + s_alist + s_maq + s_apl + s_pul


    # -------------------------------------------------
    # C) Resto de features que ya tenías (Puntadas, fecha, turno)
    # -------------------------------------------------

    # 0) Imputación de Puntadas + nueva feature
    if "Puntadas" in df.columns:
        df["Puntadas"] = df["Puntadas"].fillna(0)

        if "CantidadBuena" in df.columns:
            df["CantidadBuena"] = df["CantidadBuena"].fillna(0)
        else:
            df["CantidadBuena"] = 0

        df["puntadas_totales"] = df["Puntadas"] * df["CantidadBuena"]

    # 1) Features de FECHA (antes del drop final que haces con COLS_TO_DROP)
    if "FechaInicio" in df.columns:
        df["FechaInicio_dt"] = pd.to_datetime(df["FechaInicio"], errors="coerce")
        df["dia_semana"] = df["FechaInicio_dt"].dt.dayofweek
        df["mes"] = df["FechaInicio_dt"].dt.month
        df["hora"] = df["FechaInicio_dt"].dt.hour
        df["es_fin_semana"] = df["FechaInicio_dt"].dt.dayofweek.isin([5, 6]).astype(int)

        def clasificar_turno(h):
            if pd.isna(h):
                return "desconocido"
            h = int(h)
            if 6 <= h < 14:
                return "mañana"
            elif 14 <= h < 22:
                return "tarde"
            else:
                return "noche"

        df["turno"] = df["hora"].apply(clasificar_turno)

    return df


# =========================================================
# 1) SPLIT ESTRATIFICADO (solo entrenamiento)
# =========================================================
from sklearn.model_selection import StratifiedShuffleSplit

def create_train_test_split(df, test_size=0.2, stratify_column="tiempo_total_min"):
    bins = [-np.inf] + df[stratify_column].quantile([0.2, 0.4, 0.6, 0.8]).tolist() + [np.inf]

    df_temp = df.copy()
    df_temp["stratify_cat"] = pd.cut(
        df_temp[stratify_column],
        bins=bins,
        labels=[1, 2, 3, 4, 5]
    )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    for train_idx, test_idx in splitter.split(df_temp, df_temp["stratify_cat"]):
        train_set = df_temp.iloc[train_idx].copy()
        test_set = df_temp.iloc[test_idx].copy()

    # Limpieza
    for s in (train_set, test_set):
        s.drop("stratify_cat", axis=1, inplace=True)

    return train_set, test_set


# =========================================================
# 2) CODIFICACIÓN PARA ENTRENAMIENTO
# =========================================================
def aplicar_codificacion_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "tiempo_total_min",
    alpha: float = 5.0,
):
    """
    Versión para ENTRENAMIENTO.
    Devuelve: X_train, y_train, X_test, y_test, y metadatos:
      - global_mean
      - enc_maps
      - turno_cols
      - plantilla_cols
      - feature_order
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Asegurar tipo category en codigoMaterial como en el notebook
    if "codigoMaterial" in train_df.columns:
        train_df["codigoMaterial"] = train_df["codigoMaterial"].astype("category")
    if "codigoMaterial" in test_df.columns:
        test_df["codigoMaterial"] = test_df["codigoMaterial"].astype("category")

    # 1) Columnas a codificar por TARGET ENCODING
    te_cols = TE_COLS

    for col in te_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str)

    y_train = train_df[target_col].copy()
    y_test = test_df[target_col].copy()

    global_mean = y_train.mean()
    enc_maps = {}

    # 2) TARGET ENCODING CON SMOOTHING
    for col in te_cols:
        if col not in train_df.columns:
            continue

        stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
        stats["enc"] = (stats["mean"] * stats["count"] + global_mean * alpha) / (
            stats["count"] + alpha
        )

        enc_maps[col] = stats["enc"]

        train_df[f"{col}_encoded"] = train_df[col].map(enc_maps[col]).fillna(global_mean)
        if col in test_df.columns:
            test_df[f"{col}_encoded"] = test_df[col].map(enc_maps[col]).fillna(global_mean)
        else:
            test_df[f"{col}_encoded"] = global_mean

    # Borramos originales
    train_df = train_df.drop(columns=te_cols, errors="ignore")
    test_df = test_df.drop(columns=te_cols, errors="ignore")

    # 3) ONE-HOT turno
    if "turno" in train_df.columns:
        train_df["turno"] = train_df["turno"].astype(str).str.strip()
    if "turno" in test_df.columns:
        test_df["turno"] = test_df["turno"].astype(str).str.strip()

    if "turno" in train_df.columns:
        train_df = pd.get_dummies(train_df, columns=["turno"], prefix="turno", drop_first=True)
    if "turno" in test_df.columns:
        test_df = pd.get_dummies(test_df, columns=["turno"], prefix="turno", drop_first=True)

    turno_cols = sorted(
        set(
            [c for c in train_df.columns if c.startswith("turno_")]
            + [c for c in test_df.columns if c.startswith("turno_")]
        )
    )
    for df_ in (train_df, test_df):
        for c in turno_cols:
            if c not in df_.columns:
                df_[c] = 0

    # 4) Filtro por PuntoPlantilla (como en tu código)
    if "PuntoPlantilla" in train_df.columns:
        train_df["PuntoPlantilla"] = train_df["PuntoPlantilla"].astype(str).str.strip()
    if "PuntoPlantilla" in test_df.columns:
        test_df["PuntoPlantilla"] = test_df["PuntoPlantilla"].astype(str).str.strip()

    # Construir máscaras
    if "PuntoPlantilla" in train_df.columns:
        mask_train = train_df["PuntoPlantilla"].str.lower() != DROP_VALUE_PLANTILLA
    else:
        mask_train = pd.Series(True, index=train_df.index)

    if "PuntoPlantilla" in test_df.columns:
        mask_test = test_df["PuntoPlantilla"].str.lower() != DROP_VALUE_PLANTILLA
    else:
        mask_test = pd.Series(True, index=test_df.index)

    train_df = train_df.loc[mask_train].copy()
    test_df = test_df.loc[mask_test].copy()
    y_train = y_train.loc[mask_train].copy()
    y_test = y_test.loc[mask_test].copy()

    # One-hot de PuntoPlantilla
    if "PuntoPlantilla" in train_df.columns:
        train_df = pd.get_dummies(
            train_df, columns=["PuntoPlantilla"], prefix="Plantilla", drop_first=True
        )
    if "PuntoPlantilla" in test_df.columns:
        test_df = pd.get_dummies(
            test_df, columns=["PuntoPlantilla"], prefix="Plantilla", drop_first=True
        )

    # 5) Imputación Puntadas
    if "Puntadas" in train_df.columns:
        train_df["Puntadas"] = train_df["Puntadas"].fillna(0)
    if "Puntadas" in test_df.columns:
        test_df["Puntadas"] = test_df["Puntadas"].fillna(0)

    # 6) Ordenar columnas
    train_df = train_df.reindex(sorted(train_df.columns), axis=1)
    test_df = test_df.reindex(sorted(test_df.columns), axis=1)

    # 7) Separar X e y
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    # Columnas one-hot de Plantilla (para reusarlas en inferencia)
    plantilla_cols = [c for c in X_train.columns if c.startswith("Plantilla_")]

    feature_order = X_train.columns.tolist()

    metadata = {
        "global_mean": global_mean,
        "enc_maps": enc_maps,
        "turno_cols": turno_cols,
        "plantilla_cols": plantilla_cols,
        "feature_order": feature_order,
        "target_col": target_col,
        "te_cols": te_cols,
    }

    return X_train, y_train, X_test, y_test, metadata


# =========================================================
# 3) TRANSFORMACIÓN PARA NUEVOS DATOS (inferencia)
# =========================================================
def transformar_nueva_muestra(
    raw_df: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """
    Versión de la transformación para PRODUCCIÓN.
    Usa los enc_maps, global_mean, feature_order, etc. guardados.
    """
    df = raw_df.copy()

    # 1) Asegurar mismo feature engineering
    df = agregar_features(df)

    # 2) Drop columnas como en entrenamiento
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")

    # 3) Target encoding usando enc_maps
    global_mean = metadata["global_mean"]
    enc_maps = metadata["enc_maps"]
    te_cols = metadata["te_cols"]

    for col in te_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col, mapping in enc_maps.items():
        new_col = f"{col}_encoded"
        df[new_col] = df[col].map(mapping).fillna(global_mean)

    df = df.drop(columns=te_cols, errors="ignore")

    # 4) ONE-HOT turno
    if "turno" in df.columns:
        df["turno"] = df["turno"].astype(str).str.strip()
        df = pd.get_dummies(df, columns=["turno"], prefix="turno", drop_first=True)

    turno_cols = metadata["turno_cols"]
    for c in turno_cols:
        if c not in df.columns:
            df[c] = 0

    # 5) ONE-HOT PuntoPlantilla
    if "PuntoPlantilla" in df.columns:
        df["PuntoPlantilla"] = df["PuntoPlantilla"].astype(str).str.strip()
        df = pd.get_dummies(
            df, columns=["PuntoPlantilla"], prefix="Plantilla", drop_first=True
        )

    # 6) Imputación Puntadas
    if "Puntadas" in df.columns:
        df["Puntadas"] = df["Puntadas"].fillna(0)

    # 7) Alinear con feature_order
    feature_order = metadata["feature_order"]
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_order]

    return df
