"""Funciones auxiliares para la limpieza inicial de datasets."""

from pathlib import Path

import numpy as np
import pandas as pd


CSV_READ_OPTIONS = {
    "low_memory": False,
    "on_bad_lines": "warn",
}


def cargar_dataset(nombre_dataset="*", ruta_base="../../02_datasets/raw"):
    """
    Carga un CSV concreto o concatena todos los CSV de una carpeta.

    Parameters
    ----------
    nombre_dataset : str
        Nombre del dataset, por ejemplo ``"CIC17.csv"``. Si vale ``"*"``,
        se leen todos los CSV disponibles en ``ruta_base``.
    ruta_base : str or pathlib.Path
        Ruta base donde estan los datasets.

    Returns
    -------
    pandas.DataFrame
        Dataset cargado.
    """
    ruta_base = Path(ruta_base)
    directorio_dataset = ruta_base if ruta_base.is_absolute() else Path.cwd() / ruta_base

    if nombre_dataset == "*":
        archivos = sorted(directorio_dataset.glob("*.csv"))

        if not archivos:
            raise FileNotFoundError(f"No se encontraron CSV en: {directorio_dataset}")

        return pd.concat(
            (pd.read_csv(archivo, **CSV_READ_OPTIONS) for archivo in archivos),
            ignore_index=True,
        )

    archivo_dataset = directorio_dataset / nombre_dataset
    return pd.read_csv(archivo_dataset, **CSV_READ_OPTIONS)


def homogeneizar_columnas(df):
    """Normaliza los nombres de columnas al formato MAYUSCULAS_CON_GUIONES."""
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.upper()
    )

    return df


def eliminar_columnas(df, to_drop):
    """Elimina las columnas indicadas."""
    return df.drop(columns=to_drop)


def limpiar_infinitos_y_vacios(df):
    """
    Convierte infinitos y strings vacios en ``NaN``.

    No elimina filas; solo prepara el dataset para decidir despues como tratar los nulos.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.replace(r"^\s*$", np.nan, regex=True)


def eliminar_filas_duplicadas(df):
    """Elimina filas duplicadas."""
    return df.drop_duplicates()


def separar_filas_con_y_sin_nulos(df):
    """Separa el dataset entre filas completas y filas con al menos un nulo."""
    mask_null = df.isnull().any(axis=1)
    df_sin_nulos = df.loc[~mask_null].copy()
    df_con_nulos = df.loc[mask_null].copy()

    return df_sin_nulos, df_con_nulos


def imputar_o_eliminar_nulos_por_clase(
    df_sin_nulos,
    df_con_nulos,
    label_col="LABEL",
    min_class_impute=50,
):
    """
    Imputa o descarta filas con nulos segun el tamaño de su clase.

    Si una fila con nulos pertenece a una clase con menos de
    ``min_class_impute`` muestras completas, se intentan imputar sus columnas numericas con la mediana de esa misma clase. Si aun quedan nulos, o si la clase supera el umbral, la fila se descarta.
    """
    df_sin_nulos = df_sin_nulos.copy()

    class_counts = df_sin_nulos[label_col].value_counts().to_dict()
    numeric_cols = [
        col
        for col in df_sin_nulos.select_dtypes(include=[np.number]).columns
        if col != label_col
    ]

    filas_recuperadas = []
    imputadas = 0
    eliminadas = 0

    for _, row in df_con_nulos.iterrows():
        label = row[label_col]
        class_size = class_counts.get(label, 0)

        if class_size >= min_class_impute:
            eliminadas += 1
            continue

        row_copy = row.copy()
        subset_clase = df_sin_nulos[df_sin_nulos[label_col] == label]

        for col in numeric_cols:
            if pd.isna(row_copy[col]):
                row_copy[col] = subset_clase[col].median()

        if row_copy.isnull().any():
            eliminadas += 1
        else:
            filas_recuperadas.append(row_copy)
            imputadas += 1

    if filas_recuperadas:
        df_sin_nulos = pd.concat(
            [df_sin_nulos, pd.DataFrame(filas_recuperadas)],
            ignore_index=True,
        )

    return df_sin_nulos, imputadas, eliminadas


def eliminar_columnas_constantes(df):
    """Elimina columnas con un unico valor distinto."""
    return df.loc[:, df.nunique() > 1]


def codificar_columnas_string(df, label_col="LABEL"):
    """Factoriza columnas string, exceptuando la columna de etiquetas."""
    for col in df.columns:
        if col == label_col:
            continue

        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col], _ = pd.factorize(df[col])

    return df


def eliminar_columnas_altamente_correlacionadas(df, threshold, label_col="LABEL"):
    """Elimina columnas con correlacion absoluta superior al umbral indicado."""
    X = df.drop(columns=label_col)
    y = df[label_col]

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    X = X.drop(columns=to_drop)
    return pd.concat([X, y], axis=1)


def codificar_etiqueta_label(df, etiqueta):
    """
    Codifica la etiqueta como enteros, dejando ``Benign`` como clase 0 si existe.

    El resto de clases se ordenan por frecuencia descendente.
    """
    labels = df[etiqueta].value_counts().index.tolist()

    if "Benign" in labels:
        labels.remove("Benign")
        labels = ["Benign"] + labels

    mapping = {label: i for i, label in enumerate(labels)}
    df[etiqueta] = df[etiqueta].map(mapping)

    return df, mapping


def guardar_dataset_csv(df, nombre_archivo, ruta):
    """Guarda un DataFrame en CSV, creando la carpeta de salida si hace falta."""
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)

    df.to_csv(ruta / nombre_archivo, index=False)
