"""Metricas y resúmenes para experimentos de clasificacion."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_CLASSIFICATION_METRICS = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "fpr_weighted",
    "fnr_weighted",
    "fpr_macro",
    "fnr_macro",
    "mcc",
    "roc_auc",
    "detection_latency_seconds",
    "fit_time",
    "score_time",
]


def calcular_roc_auc_multiclase_seguro(modelo, X_val, y_val, labels_globales):
    """Calcula AUC-ROC multiclase ponderado de forma robusta."""
    try:
        if not hasattr(modelo, "predict_proba"):
            return np.nan

        y_val = pd.Series(y_val).reset_index(drop=True)
        y_proba_original = modelo.predict_proba(X_val)
        clases_modelo = np.array(modelo.classes_)
        clases_presentes_y = np.array(sorted(pd.unique(y_val)))

        if len(clases_presentes_y) < 2:
            return np.nan

        proba_por_clase = {
            clase: y_proba_original[:, idx]
            for idx, clase in enumerate(clases_modelo)
        }

        aucs = []
        pesos = []

        for clase in labels_globales:
            if clase not in clases_presentes_y or clase not in proba_por_clase:
                continue

            y_binaria = (y_val == clase).astype(int).values

            if len(np.unique(y_binaria)) < 2:
                continue

            aucs.append(roc_auc_score(y_binaria, proba_por_clase[clase]))
            pesos.append(int(y_binaria.sum()))

        if not aucs:
            return np.nan

        return float(np.average(aucs, weights=pesos))
    except Exception:
        return np.nan


def calcular_fpr_fnr_multiclase(y_true, y_pred, labels_globales):
    """Calcula FPR y FNR macro/weighted a partir de la matriz de confusion."""
    cm = confusion_matrix(y_true, y_pred, labels=labels_globales)
    total = cm.sum()
    soporte = cm.sum(axis=1)

    fpr_por_clase = []
    fnr_por_clase = []

    for i in range(len(labels_globales)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        fpr_por_clase.append(fpr)
        fnr_por_clase.append(fnr)

    fpr_por_clase = np.array(fpr_por_clase, dtype=float)
    fnr_por_clase = np.array(fnr_por_clase, dtype=float)
    soporte = np.array(soporte, dtype=float)

    if soporte.sum() > 0:
        fpr_weighted = np.average(fpr_por_clase, weights=soporte)
        fnr_weighted = np.average(fnr_por_clase, weights=soporte)
    else:
        fpr_weighted = 0.0
        fnr_weighted = 0.0

    return {
        "fpr_macro": float(np.mean(fpr_por_clase)),
        "fnr_macro": float(np.mean(fnr_por_clase)),
        "fpr_weighted": float(fpr_weighted),
        "fnr_weighted": float(fnr_weighted),
    }


def calcular_metricas_clasificacion(
    modelo,
    X_eval,
    y_eval,
    y_pred,
    labels_globales,
    score_time,
):
    """Calcula las metricas de clasificacion usadas en los experimentos."""
    metricas = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision_weighted": precision_score(
            y_eval,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            y_eval,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_eval,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "precision_macro": precision_score(
            y_eval,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_eval,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_eval,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(y_eval, y_pred),
        "roc_auc": calcular_roc_auc_multiclase_seguro(
            modelo=modelo,
            X_val=X_eval,
            y_val=y_eval,
            labels_globales=labels_globales,
        ),
        "detection_latency_seconds": float(score_time / len(y_eval))
        if len(y_eval) > 0
        else 0.0,
    }

    metricas.update(
        calcular_fpr_fnr_multiclase(
            y_true=y_eval,
            y_pred=y_pred,
            labels_globales=labels_globales,
        )
    )

    return metricas


def resumir_metricas_cv(
    df_folds,
    metadata,
    metrics=None,
    metric_prefix="B",
    group_cols=None,
    sort_by=None,
):
    """Calcula media, desviacion tipica y varianza por metrica y grupo."""
    metrics = metrics or DEFAULT_CLASSIFICATION_METRICS
    group_cols = group_cols or ["n"]
    resumenes = []

    for group_values, df_group in df_folds.groupby(group_cols):
        if len(group_cols) == 1 and not isinstance(group_values, tuple):
            group_values = (group_values,)

        fila = dict(metadata)

        for col, value in zip(group_cols, group_values):
            if isinstance(value, np.integer):
                value = int(value)
            fila[col] = value

        for metrica in metrics:
            col = f"{metric_prefix}_{metrica}"

            if col not in df_group.columns:
                continue

            fila[f"{metric_prefix}_{metrica}_mean"] = float(df_group[col].mean())
            fila[f"{metric_prefix}_{metrica}_std"] = float(df_group[col].std(ddof=1))
            fila[f"{metric_prefix}_{metrica}_var"] = float(df_group[col].var(ddof=1))

        resumenes.append(fila)

    df_resumen = pd.DataFrame(resumenes)

    sort_by = sort_by or group_cols
    cols_orden = [col for col in sort_by if col in df_resumen.columns]

    if cols_orden:
        df_resumen = df_resumen.sort_values(cols_orden)

    return df_resumen.reset_index(drop=True)


def crear_tabla_publicacion_cv(df_resumen, metric_prefix="B"):
    """Crea una tabla reducida de metricas CV para consulta rapida."""
    columnas = {
        "n": "n",
        f"{metric_prefix}_accuracy_mean": "mean_accuracy",
        f"{metric_prefix}_accuracy_std": "std_accuracy",
        f"{metric_prefix}_precision_weighted_mean": "mean_precision_weighted",
        f"{metric_prefix}_precision_weighted_std": "std_precision_weighted",
        f"{metric_prefix}_recall_weighted_mean": "mean_recall_weighted",
        f"{metric_prefix}_recall_weighted_std": "std_recall_weighted",
        f"{metric_prefix}_f1_weighted_mean": "mean_f1_weighted",
        f"{metric_prefix}_f1_weighted_std": "std_f1_weighted",
        f"{metric_prefix}_precision_macro_mean": "mean_precision_macro",
        f"{metric_prefix}_precision_macro_std": "std_precision_macro",
        f"{metric_prefix}_recall_macro_mean": "mean_recall_macro",
        f"{metric_prefix}_recall_macro_std": "std_recall_macro",
        f"{metric_prefix}_f1_macro_mean": "mean_f1_macro",
        f"{metric_prefix}_f1_macro_std": "std_f1_macro",
        f"{metric_prefix}_fpr_weighted_mean": "mean_fpr_weighted",
        f"{metric_prefix}_fpr_weighted_std": "std_fpr_weighted",
        f"{metric_prefix}_fnr_weighted_mean": "mean_fnr_weighted",
        f"{metric_prefix}_fnr_weighted_std": "std_fnr_weighted",
        f"{metric_prefix}_fpr_macro_mean": "mean_fpr_macro",
        f"{metric_prefix}_fpr_macro_std": "std_fpr_macro",
        f"{metric_prefix}_fnr_macro_mean": "mean_fnr_macro",
        f"{metric_prefix}_fnr_macro_std": "std_fnr_macro",
        f"{metric_prefix}_mcc_mean": "mean_mcc",
        f"{metric_prefix}_mcc_std": "std_mcc",
        f"{metric_prefix}_roc_auc_mean": "mean_roc_auc",
        f"{metric_prefix}_roc_auc_std": "std_roc_auc",
        f"{metric_prefix}_detection_latency_seconds_mean": (
            "mean_detection_latency_seconds"
        ),
        f"{metric_prefix}_detection_latency_seconds_std": (
            "std_detection_latency_seconds"
        ),
        f"{metric_prefix}_fit_time_mean": "mean_fit_time",
        f"{metric_prefix}_score_time_mean": "mean_score_time",
    }

    cols_existentes = [col for col in columnas if col in df_resumen.columns]
    df_tabla = df_resumen[cols_existentes].rename(columns=columnas)

    cols_num = df_tabla.select_dtypes(include=[np.number]).columns
    df_tabla[cols_num] = df_tabla[cols_num].round(6)

    return df_tabla
