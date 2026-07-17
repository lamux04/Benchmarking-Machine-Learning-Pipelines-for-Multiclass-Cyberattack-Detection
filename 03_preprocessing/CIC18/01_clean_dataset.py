"""
Fichero de limpieza inicial del dataset.

Además de generar el dataset limpio, este script guarda varios ficheros
CSV con información de trazabilidad sobre el proceso de limpieza:

- Resumen de filas/columnas por paso.
- Columnas eliminadas y motivo.
- Distribución de clases por paso.
- Información sobre imputación/eliminación de nulos.
- Mapping final de etiquetas.
"""

# ============== CONFIGURACIONES ================
DATASET_ORIGINAL = "CIC18"

# ============== LIBRERÍAS =============
from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "02_src"

sys.path.append(str(SRC_DIR))

import preprocessing
import reporting

# ============== RUTAS =============
NOMBRE_DATASET_RESULTADO = f"{DATASET_ORIGINAL}__clean"

RUTA_BASE_RAW = PROJECT_ROOT / "01_datasets" / DATASET_ORIGINAL / "raw"
RUTA_SALIDA_DATASET = PROJECT_ROOT / "01_datasets" / DATASET_ORIGINAL / "clean"
RUTA_SALIDA_REPORTES = PROJECT_ROOT / "05_results" / DATASET_ORIGINAL / "clean"

NOMBRE_DATASET_LIMPIO = f"{NOMBRE_DATASET_RESULTADO}.csv"

RUTA_SALIDA_DATASET.mkdir(parents=True, exist_ok=True)
RUTA_SALIDA_REPORTES.mkdir(parents=True, exist_ok=True)

# ============== PARÁMETROS =============
LABEL_COL = "LABEL"
CORR_THRESHOLD = 0.95
MIN_CLASS_IMPUTE = 50
BENIGN_UNDERSAMPLE_FRACTION = 0.20
ERROR_CLASS_LABEL = 15
RANDOM_STATE = 42

# ============== COLUMNAS A ELIMINAR MANUALMENTE SI EXISTEN =============
COLUMNAS_A_ELIMINAR = [
    "FLOW_ID",
    "SRC_IP",
    "SRC_PORT",
    "DST_IP",
    "TIMESTAMP",
]

# ============== ESTRUCTURAS PARA REPORTES =============
cleaning_summary = []
columns_removed = []
label_distribution_by_step = []
null_handling_summary = []


def undersample_benign_class(df, label_col, fraction, random_state):
    """Conserva solo una fraccion de las filas benignas y mantiene el resto."""
    df = df.copy()
    benign_mask = df[label_col].astype(str).str.strip().str.casefold() == "benign"
    df.loc[benign_mask, label_col] = "Benign"
    benign_df = df.loc[benign_mask]

    if benign_df.empty:
        raise ValueError("No se encontraron filas de la clase Benign para undersampling.")

    n_benign_keep = max(1, round(len(benign_df) * fraction))
    benign_sampled = benign_df.sample(
        n=n_benign_keep,
        random_state=random_state,
    )

    df_undersampled = pd.concat(
        [df.loc[~benign_mask], benign_sampled],
        ignore_index=True,
    )

    return df_undersampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ======================================================
# LIMPIEZA DEL DATASET
# ======================================================

print("Cargando dataset...")
df = preprocessing.cargar_dataset(
    ruta_base=RUTA_BASE_RAW,
)

reporting.registrar_estado(df, "initial")
reporting.registrar_distribucion_clases(df, "initial")

# ============== HOMOGENEIZAR NOMBRES DE COLUMNAS ===============
print("Homogeneizando nombres de columnas...")

prev_rows, prev_cols = len(df), len(df.columns)
df = preprocessing.homogeneizar_columnas(df)

reporting.registrar_estado(df, "homogenize_columns", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "homogenize_columns")

# ============== ELIMINAR COLUMNAS MANUALMENTE ===============
print("Eliminando columnas manuales...")

prev_rows, prev_cols = len(df), len(df.columns)
cols_before = df.columns.tolist()

columnas_presentes_para_eliminar = [c for c in COLUMNAS_A_ELIMINAR if c in df.columns]
df = preprocessing.eliminar_columnas(df, columnas_presentes_para_eliminar)

cols_after = df.columns.tolist()
columnas_eliminadas = reporting.detectar_columnas_eliminadas(cols_before, cols_after)

reporting.registrar_columnas_eliminadas(
    columnas=columnas_eliminadas,
    step="remove_manual_columns",
    reason="manual_removal",
    extra_info="Columns removed because they are identifiers, IP addresses, ports, or non-generalizable attributes.",
)

reporting.registrar_estado(df, "remove_manual_columns", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "remove_manual_columns")

# ============== LIMPIAR INFINITOS Y VACÍOS =================
print("Limpiando infinitos y vacíos...")

prev_rows, prev_cols = len(df), len(df.columns)
df = preprocessing.limpiar_infinitos_y_vacios(df)

reporting.registrar_estado(df, "remove_infinite_and_empty_values", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "remove_infinite_and_empty_values")

# ============== ELIMINAR FILAS DUPLICADAS =================
print("Eliminando filas duplicadas...")

prev_rows, prev_cols = len(df), len(df.columns)
df = preprocessing.eliminar_filas_duplicadas(df)

reporting.registrar_estado(df, "remove_duplicate_rows", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "remove_duplicate_rows")

# ============== IMPUTAR O ELIMINAR NULOS POR CLASE ===============
print("Tratando valores nulos por clase...")

prev_rows, prev_cols = len(df), len(df.columns)

df_sin_nulos, df_con_nulos = preprocessing.separar_filas_con_y_sin_nulos(df)

null_handling_summary.append(
    {
        "dataset": DATASET_ORIGINAL,
        "step": "before_null_handling",
        "rows_without_nulls": len(df_sin_nulos),
        "rows_with_nulls": len(df_con_nulos),
        "min_class_impute": MIN_CLASS_IMPUTE,
    }
)

df, filas_imputadas, filas_eliminadas_nulos = preprocessing.imputar_o_eliminar_nulos_por_clase(
    df_sin_nulos=df_sin_nulos,
    df_con_nulos=df_con_nulos,
    label_col=LABEL_COL,
    min_class_impute=MIN_CLASS_IMPUTE,
)

null_handling_summary.append(
    {
        "dataset": DATASET_ORIGINAL,
        "step": "after_null_handling",
        "imputed_rows": filas_imputadas,
        "removed_rows": filas_eliminadas_nulos,
        "min_class_impute": MIN_CLASS_IMPUTE,
    }
)

reporting.registrar_estado(df, "handle_null_values_by_class", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "handle_null_values_by_class")

# ============= ELIMINAR COLUMNAS CONSTANTES ================
print("Eliminando columnas constantes...")

prev_rows, prev_cols = len(df), len(df.columns)
cols_before = df.columns.tolist()

df = preprocessing.eliminar_columnas_constantes(df)

cols_after = df.columns.tolist()
columnas_eliminadas = reporting.detectar_columnas_eliminadas(cols_before, cols_after)

reporting.registrar_columnas_eliminadas(
    columnas=columnas_eliminadas,
    step="remove_constant_columns",
    reason="constant_column",
    extra_info="Columns with a single value or zero variance.",
)

reporting.registrar_estado(df, "remove_constant_columns", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "remove_constant_columns")

# ============= CODIFICAR COLUMNAS STRINGS ==================
print("Codificando columnas string...")

prev_rows, prev_cols = len(df), len(df.columns)

df = preprocessing.codificar_columnas_string(df, LABEL_COL)

reporting.registrar_estado(df, "encode_string_columns", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "encode_string_columns")

# ============= ELIMINAR COLUMNAS ALTAMENTE CORRELACIONADAS =================
print("Eliminando columnas altamente correlacionadas...")

prev_rows, prev_cols = len(df), len(df.columns)
cols_before = df.columns.tolist()

df = preprocessing.eliminar_columnas_altamente_correlacionadas(
    df,
    threshold=CORR_THRESHOLD,
    label_col=LABEL_COL,
)

cols_after = df.columns.tolist()
columnas_eliminadas = reporting.detectar_columnas_eliminadas(cols_before, cols_after)

reporting.registrar_columnas_eliminadas(
    columnas=columnas_eliminadas,
    step="remove_highly_correlated_columns",
    reason="high_correlation",
    extra_info=f"Correlation threshold: {CORR_THRESHOLD}",
)

reporting.registrar_estado(df, "remove_highly_correlated_columns", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "remove_highly_correlated_columns")

# ============= UNDERSAMPLING CLASE BENIGNA =================
print("Aplicando undersampling de la clase Benign...")

prev_rows, prev_cols = len(df), len(df.columns)

df = undersample_benign_class(
    df=df,
    label_col=LABEL_COL,
    fraction=BENIGN_UNDERSAMPLE_FRACTION,
    random_state=RANDOM_STATE,
)

reporting.registrar_estado(df, "undersample_benign_class", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "undersample_benign_class")

# ============ CODIFICAR ETIQUETA LABEL ================
print("Codificando etiqueta LABEL...")

prev_rows, prev_cols = len(df), len(df.columns)

df, mapping = preprocessing.codificar_etiqueta_label(df, LABEL_COL)

reporting.registrar_estado(df, "encode_label", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "encode_label")

# ============ ELIMINAR CLASE ERRONEA ================
print(f"Eliminando clase erronea {ERROR_CLASS_LABEL}...")

prev_rows, prev_cols = len(df), len(df.columns)

df = df[df[LABEL_COL] != ERROR_CLASS_LABEL].reset_index(drop=True)
mapping = {
    original_label: encoded_label
    for original_label, encoded_label in mapping.items()
    if encoded_label != ERROR_CLASS_LABEL
}

reporting.registrar_estado(df, "remove_error_class", prev_rows, prev_cols)
reporting.registrar_distribucion_clases(df, "remove_error_class")

# Guardar mapping de etiquetas
mapping_df = pd.DataFrame(
    [
        {
            "dataset": DATASET_ORIGINAL,
            "original_label": original_label,
            "encoded_label": encoded_label,
        }
        for original_label, encoded_label in mapping.items()
    ]
)

mapping_df.to_csv(
    RUTA_SALIDA_REPORTES / f"{DATASET_ORIGINAL}__label_mapping.csv",
    index=False,
)

# ============ GUARDAR DATASET ===============
print("Guardando dataset limpio...")

preprocessing.guardar_dataset_csv(
    df=df,
    nombre_archivo=NOMBRE_DATASET_LIMPIO,
    ruta=RUTA_SALIDA_DATASET,
)

# ============ GUARDAR REPORTES ===============
print("Guardando reportes de limpieza...")

reporting.guardar_reportes()

print("Proceso de limpieza finalizado correctamente.")
print(f"Dataset limpio guardado en: {RUTA_SALIDA_DATASET / NOMBRE_DATASET_LIMPIO}")
print(f"Reportes guardados en: {RUTA_SALIDA_REPORTES}")
