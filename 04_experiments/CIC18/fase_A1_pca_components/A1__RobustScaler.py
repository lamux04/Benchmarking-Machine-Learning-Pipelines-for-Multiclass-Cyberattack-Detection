"""
Experimento A1 con RobustScaler.

El objetivo es obtener el primer k que alcanza una varianza acumulada del 99%
usando PCA sobre el dataset de entrenamiento escalado con RobustScaler.

El experimento guarda:

- Resumen del PCA y componentes seleccionados.
- Varianza explicada y acumulada por componente.
- Parametros del experimento.
- Grafica de varianza acumulada en PDF.
"""

# ============== CONFIGURACIONES ================
DATASET_ORIGINAL = "CIC18"
EXPERIMENT_NAME = "A1__RobustScaler"

# ============== LIBRERIAS =============
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "02_src"

sys.path.append(str(SRC_DIR))

import preprocessing
import reporting

# ============== RUTAS =============
NOMBRE_DATASET_ENTRADA = f"{DATASET_ORIGINAL}__split"

RUTA_DATASET_ENTRADA = PROJECT_ROOT / "01_datasets" / DATASET_ORIGINAL / "split"
RUTA_SALIDA_RESULTADOS = (
    PROJECT_ROOT / "05_results" / DATASET_ORIGINAL / "fase_A1_pca_components" / EXPERIMENT_NAME
)
RUTA_SALIDA_REPORTES = RUTA_SALIDA_RESULTADOS / "reports"
RUTA_SALIDA_GRAFICAS = RUTA_SALIDA_RESULTADOS / "figures"

NOMBRE_TRAIN = f"{NOMBRE_DATASET_ENTRADA}__train.csv"
NOMBRE_GRAFICA = f"{DATASET_ORIGINAL}__{EXPERIMENT_NAME}__pca_cumulative_variance.pdf"

RUTA_SALIDA_REPORTES.mkdir(parents=True, exist_ok=True)
RUTA_SALIDA_GRAFICAS.mkdir(parents=True, exist_ok=True)

# ============== PARAMETROS =============
VARIANCE_THRESHOLD = 0.99
LABEL_COL = "LABEL"
RANDOM_STATE = 42

# ============== ESTRUCTURAS PARA REPORTES =============
experiment_parameters = []
pca_summary = []
pca_explained_variance = []


# ======================================================
# EXPERIMENTO
# ======================================================

print("Cargando dataset de entrenamiento...")
train_df = preprocessing.cargar_dataset(
    nombre_dataset=NOMBRE_TRAIN,
    ruta_base=RUTA_DATASET_ENTRADA,
)

# ============== SEPARAR X E Y ==================
print("Separando variables predictoras y etiqueta...")
X_train = train_df.drop(columns=[LABEL_COL])

# ============== ESCALADO ======================
print("Escalando variables con RobustScaler...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ============== PCA SIN LIMITAR COMPONENTES ==============
print("Ajustando PCA...")
pca = PCA(random_state=RANDOM_STATE)
pca.fit(X_train_scaled)

# ============== VARIANZA EXPLICADA ==============
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# ============== CALCULO DE K ===================
selected_components = int(np.argmax(cumulative_variance >= VARIANCE_THRESHOLD) + 1)
selected_cumulative_variance = float(cumulative_variance[selected_components - 1])

# ============== GRAFICA ===================
print("Generando grafica de varianza acumulada...")
figure_path = RUTA_SALIDA_GRAFICAS / NOMBRE_GRAFICA

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o",
    linewidth=1.8,
    markersize=4,
    label="Varianza acumulada",
)
plt.axhline(
    y=VARIANCE_THRESHOLD,
    color="r",
    linestyle="--",
    label=f"{int(VARIANCE_THRESHOLD * 100)}%",
)
plt.axvline(
    x=selected_components,
    color="gray",
    linestyle=":",
    label=f"k = {selected_components}",
)

plt.xlabel("Numero de componentes")
plt.ylabel("Varianza acumulada")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(figure_path, format="pdf", bbox_inches="tight")
plt.close()

# ============== REPORTES ===================
print("Registrando reportes del experimento...")

reporting.registrar_parametros_experimento(
    experiment_name=EXPERIMENT_NAME,
    parameters={
        "scaler": "RobustScaler",
        "variance_threshold": VARIANCE_THRESHOLD,
        "random_state": RANDOM_STATE,
        "label_col": LABEL_COL,
        "input_dataset": NOMBRE_TRAIN,
        "rows_train": len(train_df),
        "features_train": X_train.shape[1],
    },
)

reporting.registrar_resumen_pca(
    experiment_name=EXPERIMENT_NAME,
    variance_threshold=VARIANCE_THRESHOLD,
    selected_components=selected_components,
    selected_cumulative_variance=selected_cumulative_variance,
    n_samples=X_train.shape[0],
    n_features=X_train.shape[1],
    n_components_total=len(explained_variance),
    figure_path=str(figure_path),
)

reporting.registrar_varianza_pca(
    experiment_name=EXPERIMENT_NAME,
    explained_variance=explained_variance,
    cumulative_variance=cumulative_variance,
)

print("Guardando reportes del experimento...")
reporting.guardar_reportes_experimento(EXPERIMENT_NAME)

print("Experimento finalizado correctamente.")
print(f"Componentes seleccionados: {selected_components}")
print(f"Varianza acumulada alcanzada: {selected_cumulative_variance:.6f}")
print(f"Grafica guardada en: {figure_path}")
print(f"Reportes guardados en: {RUTA_SALIDA_REPORTES}")
