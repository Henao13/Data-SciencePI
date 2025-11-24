import os
import joblib

# Ruta al bundle original
bundle_path = os.path.join("models", "model_bundle.pkl")
print("Cargando bundle original desde:", bundle_path)
bundle = joblib.load(bundle_path)

model = bundle["model"]
metadata = bundle["metadata"]

# Construimos un bundle "slim" (solo lo necesario)
slim_bundle = {
    "model": model,
    "metadata": metadata,
}

slim_path = os.path.join("models", "model_bundle_slim.pkl")
joblib.dump(slim_bundle, slim_path, compress=3)

size_slim_mb = os.path.getsize(slim_path) / (1024 * 1024)
print(f"✅ Slim bundle guardado en: {slim_path}")
print(f"📦 Tamaño del slim bundle: {size_slim_mb:.2f} MB")
