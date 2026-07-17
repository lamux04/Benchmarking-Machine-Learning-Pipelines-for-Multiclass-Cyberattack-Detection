"""Funciones auxiliares para generar reportes de preprocesamiento."""

import inspect
from pathlib import Path

import pandas as pd


DATASET_ORIGINAL = None
LABEL_COL = "LABEL"
RUTA_SALIDA_REPORTES = None

cleaning_summary = []
columns_removed = []
label_distribution_by_step = []
null_handling_summary = []
split_summary = []
split_label_distribution = []
split_parameters = []
experiment_parameters = []
pca_summary = []
pca_explained_variance = []
cv_fold_results = []
cv_summary = []


def configurar_reportes(
    dataset_original,
    ruta_salida_reportes,
    label_col="LABEL",
    resumen_limpieza=None,
    columnas_eliminadas=None,
    distribucion_clases=None,
    resumen_nulos=None,
    resumen_split=None,
    distribucion_split=None,
    parametros_split=None,
    parametros_experimento=None,
    resumen_pca=None,
    varianza_pca=None,
    resultados_cv_fold=None,
    resumen_cv=None,
):
    """
    Configura el contexto de reporting de forma explicita.

    Los parametros de listas son opcionales para mantener compatibilidad con
    scripts que quieran gestionar esas estructuras fuera del modulo.
    """
    global DATASET_ORIGINAL
    global LABEL_COL
    global RUTA_SALIDA_REPORTES
    global cleaning_summary
    global columns_removed
    global label_distribution_by_step
    global null_handling_summary
    global split_summary
    global split_label_distribution
    global split_parameters
    global experiment_parameters
    global pca_summary
    global pca_explained_variance
    global cv_fold_results
    global cv_summary

    DATASET_ORIGINAL = dataset_original
    LABEL_COL = label_col
    RUTA_SALIDA_REPORTES = Path(ruta_salida_reportes)

    if resumen_limpieza is not None:
        cleaning_summary = resumen_limpieza
    if columnas_eliminadas is not None:
        columns_removed = columnas_eliminadas
    if distribucion_clases is not None:
        label_distribution_by_step = distribucion_clases
    if resumen_nulos is not None:
        null_handling_summary = resumen_nulos
    if resumen_split is not None:
        split_summary = resumen_split
    if distribucion_split is not None:
        split_label_distribution = distribucion_split
    if parametros_split is not None:
        split_parameters = parametros_split
    if parametros_experimento is not None:
        experiment_parameters = parametros_experimento
    if resumen_pca is not None:
        pca_summary = resumen_pca
    if varianza_pca is not None:
        pca_explained_variance = varianza_pca
    if resultados_cv_fold is not None:
        cv_fold_results = resultados_cv_fold
    if resumen_cv is not None:
        cv_summary = resumen_cv


def _context_value(name, default=None):
    """Obtiene valores configurados o, por compatibilidad, del script llamador."""
    value = globals().get(name)

    if value is not None:
        return value

    frame = inspect.currentframe().f_back

    while frame is not None:
        if frame.f_globals is globals():
            frame = frame.f_back
            continue

        value = frame.f_globals.get(name)

        if value is not None:
            return value

        frame = frame.f_back

    return default


def _context_list(name):
    """Obtiene una lista de reporting configurada o declarada en el script."""
    local_list = globals().get(name)

    if local_list:
        return local_list

    frame = inspect.currentframe().f_back

    while frame is not None:
        if frame.f_globals is globals():
            frame = frame.f_back
            continue

        value = frame.f_globals.get(name)

        if value is not None:
            return value

        frame = frame.f_back

    return local_list


def _dataset_original():
    return _context_value("DATASET_ORIGINAL", "dataset")


def _ruta_salida_reportes():
    ruta = _context_value("RUTA_SALIDA_REPORTES")

    if ruta is None:
        raise ValueError(
            "RUTA_SALIDA_REPORTES no esta configurada. Usa configurar_reportes(...) "
            "o define la variable antes de llamar a guardar_reportes()."
        )

    return Path(ruta)


def registrar_estado(df, step, previous_rows=None, previous_cols=None):
    """Registra el numero de filas y columnas del dataset en un paso concreto."""
    rows = len(df)
    cols = len(df.columns)

    removed_rows = None if previous_rows is None else previous_rows - rows
    removed_cols = None if previous_cols is None else previous_cols - cols

    _context_list("cleaning_summary").append(
        {
            "dataset": _dataset_original(),
            "step": step,
            "rows": rows,
            "columns": cols,
            "removed_rows": removed_rows,
            "removed_columns": removed_cols,
        }
    )


def registrar_distribucion_clases(df, step, label_col=None):
    """Registra la distribucion de clases en un paso concreto."""
    label_col = label_col or _context_value("LABEL_COL", "LABEL")

    if label_col not in df.columns:
        return

    total = len(df)
    counts = df[label_col].value_counts(dropna=False)

    for label, count in counts.items():
        _context_list("label_distribution_by_step").append(
            {
                "dataset": _dataset_original(),
                "step": step,
                "label": label,
                "count": int(count),
                "proportion": count / total if total > 0 else 0,
            }
        )


def registrar_parametros_split(test_size, random_state, label_col=None):
    """Registra los parametros usados para dividir el dataset."""
    _context_list("split_parameters").append(
        {
            "dataset": _dataset_original(),
            "test_size": test_size,
            "train_size": 1 - test_size,
            "random_state": random_state,
            "label_col": label_col or _context_value("LABEL_COL", "LABEL"),
        }
    )


def registrar_resumen_split(df_original, train_df, test_df):
    """Registra el tamano total y relativo de cada particion."""
    total_rows = len(df_original)

    particiones = {
        "original": df_original,
        "train": train_df,
        "test": test_df,
    }

    for partition, df_partition in particiones.items():
        rows = len(df_partition)

        _context_list("split_summary").append(
            {
                "dataset": _dataset_original(),
                "partition": partition,
                "rows": rows,
                "columns": len(df_partition.columns),
                "proportion": rows / total_rows if total_rows > 0 else 0,
            }
        )


def registrar_distribucion_clases_split(
    df_original,
    train_df,
    test_df,
    label_col=None,
):
    """Registra la distribucion de clases del dataset original, train y test."""
    label_col = label_col or _context_value("LABEL_COL", "LABEL")

    particiones = {
        "original": df_original,
        "train": train_df,
        "test": test_df,
    }

    for partition, df_partition in particiones.items():
        if label_col not in df_partition.columns:
            continue

        total = len(df_partition)
        counts = df_partition[label_col].value_counts(dropna=False)

        for label, count in counts.items():
            _context_list("split_label_distribution").append(
                {
                    "dataset": _dataset_original(),
                    "partition": partition,
                    "label": label,
                    "count": int(count),
                    "proportion": count / total if total > 0 else 0,
                }
            )


def registrar_columnas_eliminadas(columnas, step, reason, extra_info=None):
    """Registra las columnas eliminadas en un paso concreto."""
    for col in columnas:
        _context_list("columns_removed").append(
            {
                "dataset": _dataset_original(),
                "step": step,
                "column": col,
                "reason": reason,
                "extra_info": extra_info,
            }
        )


def registrar_parametros_experimento(experiment_name, parameters):
    """Registra los parametros generales de un experimento."""
    registro = {
        "dataset": _dataset_original(),
        "experiment": experiment_name,
    }
    registro.update(parameters)

    _context_list("experiment_parameters").append(registro)


def registrar_resumen_pca(
    experiment_name,
    variance_threshold,
    selected_components,
    selected_cumulative_variance,
    n_samples,
    n_features,
    n_components_total,
    figure_path=None,
):
    """Registra el resumen del PCA y el numero de componentes elegido."""
    _context_list("pca_summary").append(
        {
            "dataset": _dataset_original(),
            "experiment": experiment_name,
            "variance_threshold": variance_threshold,
            "selected_components": selected_components,
            "selected_cumulative_variance": selected_cumulative_variance,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components_total": n_components_total,
            "figure_path": figure_path,
        }
    )


def registrar_varianza_pca(experiment_name, explained_variance, cumulative_variance):
    """Registra la varianza explicada y acumulada por componente principal."""
    for i, (explained, cumulative) in enumerate(
        zip(explained_variance, cumulative_variance),
        start=1,
    ):
        _context_list("pca_explained_variance").append(
            {
                "dataset": _dataset_original(),
                "experiment": experiment_name,
                "component": i,
                "explained_variance_ratio": explained,
                "cumulative_variance": cumulative,
            }
        )


def registrar_resultado_cv_fold(experiment_name, row):
    """Registra los resultados de un fold de validacion cruzada."""
    registro = {
        "dataset": _dataset_original(),
        "experiment": experiment_name,
    }
    registro.update(row)

    _context_list("cv_fold_results").append(registro)


def registrar_resumen_cv(experiment_name, row):
    """Registra el resumen agregado de validacion cruzada para un valor de n."""
    registro = {
        "dataset": _dataset_original(),
        "experiment": experiment_name,
    }
    registro.update(row)

    _context_list("cv_summary").append(registro)


def detectar_columnas_eliminadas(cols_before, cols_after):
    """Devuelve las columnas que estaban antes y ya no estan despues."""
    return sorted(set(cols_before) - set(cols_after))


def guardar_reportes():
    """Guarda todos los reportes generados durante la limpieza."""
    ruta_salida = _ruta_salida_reportes()
    ruta_salida.mkdir(parents=True, exist_ok=True)
    dataset = _dataset_original()

    reportes = {
        "cleaning_summary": _context_list("cleaning_summary"),
        "columns_removed": _context_list("columns_removed"),
        "label_distribution_by_step": _context_list("label_distribution_by_step"),
        "null_handling_summary": _context_list("null_handling_summary"),
    }

    for nombre_reporte, registros in reportes.items():
        pd.DataFrame(registros).to_csv(
            ruta_salida / f"{dataset}__{nombre_reporte}.csv",
            index=False,
        )


def guardar_reportes_split():
    """Guarda los reportes generados durante la division train/test."""
    ruta_salida = _ruta_salida_reportes()
    ruta_salida.mkdir(parents=True, exist_ok=True)
    dataset = _dataset_original()

    reportes = {
        "split_summary": _context_list("split_summary"),
        "split_label_distribution": _context_list("split_label_distribution"),
        "split_parameters": _context_list("split_parameters"),
    }

    for nombre_reporte, registros in reportes.items():
        pd.DataFrame(registros).to_csv(
            ruta_salida / f"{dataset}__{nombre_reporte}.csv",
            index=False,
        )


def guardar_reportes_experimento(experiment_name):
    """Guarda los reportes generados durante un experimento."""
    ruta_salida = _ruta_salida_reportes()
    ruta_salida.mkdir(parents=True, exist_ok=True)
    dataset = _dataset_original()

    reportes = {
        "experiment_parameters": _context_list("experiment_parameters"),
        "pca_summary": _context_list("pca_summary"),
        "pca_explained_variance": _context_list("pca_explained_variance"),
    }

    for nombre_reporte, registros in reportes.items():
        pd.DataFrame(registros).to_csv(
            ruta_salida / f"{dataset}__{experiment_name}__{nombre_reporte}.csv",
            index=False,
        )


def guardar_reportes_cv_experimento(experiment_name):
    """Guarda los reportes de un experimento basado en validacion cruzada."""
    ruta_salida = _ruta_salida_reportes()
    ruta_salida.mkdir(parents=True, exist_ok=True)
    dataset = _dataset_original()

    reportes = {
        "experiment_parameters": _context_list("experiment_parameters"),
        "cv_fold_results": _context_list("cv_fold_results"),
        "cv_summary": _context_list("cv_summary"),
    }

    for nombre_reporte, registros in reportes.items():
        pd.DataFrame(registros).to_csv(
            ruta_salida / f"{dataset}__{experiment_name}__{nombre_reporte}.csv",
            index=False,
        )
