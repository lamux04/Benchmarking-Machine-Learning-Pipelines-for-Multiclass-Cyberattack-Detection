"""Construye una tabla comparativa global de resultados A2."""

from pathlib import Path
import os
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ORIGINAL = "CIC17"
PHASE_DIR_NAME = "fase_A2_sampling_size"

RESULTS_DIR = PROJECT_ROOT / "05_results" / DATASET_ORIGINAL / PHASE_DIR_NAME
OUTPUT_DIR = RESULTS_DIR / "_comparison"
PLOTS_DIR = OUTPUT_DIR / "graficas"

SUMMARY_SUFFIX = "__cv_summary.csv"
FOLDS_SUFFIX = "__cv_fold_results.csv"

METRIC_AGG_RE = re.compile(r"^B_.+_(mean|std)$")
METRIC_VAR_RE = re.compile(r"^B_.+_var$")

VD_METRICS = {
    "B_f1_macro_std": {
        "label": "std F1 macro",
        "ylabel": "VD normalizado de desviacion tipica F1 macro",
        "title": "Beneficio marginal de la desviacion tipica F1 macro",
        "file": "vd_std_f1_macro",
    },
    "B_f1_macro_mean": {
        "label": "mean F1 macro",
        "ylabel": "VD normalizado de F1 macro medio",
        "title": "Beneficio marginal de F1 macro medio",
        "file": "vd_media_f1_macro",
    },
    "B_accuracy_mean": {
        "label": "mean accuracy",
        "ylabel": "VD normalizado de accuracy medio",
        "title": "Beneficio marginal de accuracy medio",
        "file": "vd_media_accuracy",
    },
}

VD_TIMES = {
    "fit_time": {
        "column": "B_fit_time_mean",
        "label": "fit time",
        "file": "fit_time",
    },
    "score_time": {
        "column": "B_score_time_mean",
        "label": "score time",
        "file": "score_time",
    },
    "total_time": {
        "column": "B_total_time_mean",
        "label": "fit time + score time",
        "file": "fit_score_time",
    },
}

MEAN_METRIC_PLOTS = {
    "B_f1_macro_mean": {
        "ylabel": "F1 macro medio",
        "file": "media_f1_macro_medio_sin_vd",
    },
    "B_f1_macro_std": {
        "ylabel": "Desviacion tipica F1 macro media",
        "file": "std_f1_macro_medio_sin_vd",
    },
}


def _read_csv(path):
    """Lee un CSV y anade informacion de procedencia."""
    df = pd.read_csv(path)
    relative_parts = path.relative_to(RESULTS_DIR).parts

    if len(relative_parts) >= 2 and "model_family" not in df.columns:
        df["model_family"] = relative_parts[0]

    df["source_file"] = str(path.relative_to(PROJECT_ROOT))
    return df


def _find_report_files(suffix):
    """Localiza reportes A2 dentro de cualquier familia/modelo."""
    return sorted(RESULTS_DIR.glob(f"*/*/reports/*{suffix}"))


def _find_experiment_dirs():
    """Localiza directorios de experimento A2."""
    return sorted(
        path
        for path in RESULTS_DIR.glob("*/*")
        if path.is_dir() and path.name != "_comparison"
    )


def _build_missing_reports_message():
    """Construye un diagnostico cuando aun no hay reportes A2."""
    experiment_dirs = _find_experiment_dirs()

    if not experiment_dirs:
        return (
            "No se han encontrado experimentos A2 en resultados. "
            f"Ruta esperada: {RESULTS_DIR}/*/*"
        )

    lines = [
        "No se han encontrado reportes A2 de resumen.",
        f"Ruta esperada: {RESULTS_DIR}/*/*/reports/*{SUMMARY_SUFFIX}",
        "Directorios de experimento encontrados, pero sin cv_summary:",
    ]
    lines.extend(
        f"- {path.relative_to(PROJECT_ROOT)}" for path in experiment_dirs
    )
    lines.append(
        "Ejecuta primero los experimentos A2; este script solo agrega los CSV "
        "que generan."
    )

    return "\n".join(lines)


def _ordered_columns(df):
    """Ordena metadatos primero y metricas despues, sin perder columnas."""
    metadata_cols = [
        "dataset",
        "experiment",
        "descripcion_experimento",
        "model_family",
        "model",
        "model_display_name",
        "scaler",
        "use_pca",
        "n_components_pca",
        "rebalanceo",
        "rebalanceo_short",
        "n",
    ]
    metric_cols = sorted(col for col in df.columns if METRIC_AGG_RE.match(col))
    other_cols = [
        col
        for col in df.columns
        if col not in metadata_cols
        and col not in metric_cols
        and not METRIC_VAR_RE.match(col)
    ]

    return (
        [col for col in metadata_cols if col in df.columns]
        + metric_cols
        + other_cols
    )


def _sort_table(df):
    """Aplica una ordenacion estable para comparar experimentos y tamanos."""
    sort_cols = [
        "model_family",
        "model",
        "scaler",
        "use_pca",
        "n_components_pca",
        "rebalanceo_short",
        "experiment",
        "n",
    ]
    sort_cols = [col for col in sort_cols if col in df.columns]

    if not sort_cols:
        return df

    return df.sort_values(sort_cols).reset_index(drop=True)


def _build_summary_table():
    summary_files = _find_report_files(SUMMARY_SUFFIX)

    if not summary_files:
        raise FileNotFoundError(_build_missing_reports_message())

    df = pd.concat((_read_csv(path) for path in summary_files), ignore_index=True)
    df = _sort_table(df)
    return df[_ordered_columns(df)]


def _build_fold_table():
    fold_files = _find_report_files(FOLDS_SUFFIX)

    if not fold_files:
        return pd.DataFrame()

    df = pd.concat((_read_csv(path) for path in fold_files), ignore_index=True)
    return _sort_table(df)


def _build_mean_metrics_table(df_summary):
    """Crea una vista compacta con medias y desviaciones tipicas."""
    metric_cols = sorted(
        col
        for col in df_summary.columns
        if col.startswith("B_") and col.endswith(("_mean", "_std"))
    )
    metadata_cols = [
        "dataset",
        "experiment",
        "descripcion_experimento",
        "model",
        "model_display_name",
        "scaler",
        "use_pca",
        "n_components_pca",
        "rebalanceo_short",
        "n",
    ]
    cols = [col for col in metadata_cols if col in df_summary.columns] + metric_cols
    return df_summary[cols].copy()


def _normalize_series(series):
    """Normaliza una serie a [0, 1] manteniendo nulos."""
    series = series.astype(float)
    minimum = series.min(skipna=True)
    maximum = series.max(skipna=True)

    if pd.isna(minimum) or pd.isna(maximum) or np.isclose(maximum, minimum):
        return pd.Series(np.where(series.notna(), 0.0, np.nan), index=series.index)

    return (series - minimum) / (maximum - minimum)


def _validate_vd_columns(df):
    """Comprueba que existen las columnas necesarias para calcular VD."""
    required_cols = {"n", *VD_METRICS.keys()}
    required_cols.update(cfg["column"] for cfg in VD_TIMES.values() if cfg["column"] != "B_total_time_mean")
    missing_cols = sorted(col for col in required_cols if col not in df.columns)

    if missing_cols:
        raise ValueError(
            "No se pueden generar las tablas/graficas VD. "
            "Faltan columnas en el resumen A2: " + ", ".join(missing_cols)
        )


def _prepare_vd_base(df_summary):
    """Prepara la tabla base con metricas y tiempos usados por las graficas VD."""
    _validate_vd_columns(df_summary)

    vd_cols = ["n", *VD_METRICS.keys(), "B_fit_time_mean", "B_score_time_mean"]
    metadata_cols = [
        "experiment",
        "descripcion_experimento",
        "model_family",
        "model",
        "scaler",
        "use_pca",
        "n_components_pca",
        "rebalanceo_short",
    ]
    cols = [col for col in metadata_cols if col in df_summary.columns] + vd_cols
    df = df_summary[cols].copy()

    for col in vd_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["B_total_time_mean"] = df["B_fit_time_mean"] + df["B_score_time_mean"]
    return df


def _build_vd_table(df_base, group_cols):
    """Calcula el beneficio marginal VD por grupo y tamano de muestra."""
    rows = []
    groups = (
        df_base.groupby(group_cols, sort=False, dropna=False)
        if group_cols
        else [((), df_base)]
    )

    for group_keys, df_group in groups:
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        group_values = dict(zip(group_cols, group_keys))
        df_group = df_group.sort_values("n")

        for metric_col, metric_cfg in VD_METRICS.items():
            metric_delta = df_group[metric_col].diff()

            for time_name, time_cfg in VD_TIMES.items():
                time_col = time_cfg["column"]
                time_delta = df_group[time_col].diff()
                vd = metric_delta / time_delta.replace(0, np.nan)
                vd = vd.where(time_delta > 0)
                normalized_vd = _normalize_series(vd)

                for idx, row in df_group.iterrows():
                    rows.append(
                        {
                            **group_values,
                            "n": row["n"],
                            "metrica": metric_cfg["label"],
                            "metrica_columna": metric_col,
                            "tipo_tiempo": time_name,
                            "tiempo_columna": time_col,
                            "tiempo": row[time_col],
                            "valor_metrica": row[metric_col],
                            "delta_metrica": metric_delta.loc[idx],
                            "delta_tiempo": time_delta.loc[idx],
                            "VD": vd.loc[idx],
                            "VD_normalizado": normalized_vd.loc[idx],
                        }
                    )

    return pd.DataFrame(rows)


def _plot_vd_by_experiment(tabla_vd, n_values, metric_col, time_name):
    """Genera una grafica VD con una linea por experimento."""
    metric_cfg = VD_METRICS[metric_col]
    time_cfg = VD_TIMES[time_name]
    data = tabla_vd[
        (tabla_vd["metrica_columna"] == metric_col)
        & (tabla_vd["tipo_tiempo"] == time_name)
    ]
    x_pos = np.arange(len(n_values))
    x_map = dict(zip(n_values, x_pos))

    plt.figure(figsize=(14, 7))

    label_col = (
        "descripcion_experimento"
        if "descripcion_experimento" in data.columns
        else "experiment"
    )
    for experiment, df_exp in data.groupby(label_col, sort=False):
        df_exp = df_exp.sort_values("n")
        plt.plot(
            df_exp["n"].map(x_map),
            df_exp["VD_normalizado"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=experiment,
        )

    plt.xticks(x_pos, [str(int(n)) for n in n_values])
    plt.ylim(-0.05, 1.05)
    plt.xlabel("N")
    plt.ylabel(metric_cfg["ylabel"])
    plt.grid(True, alpha=0.3)
    plt.legend(title="Experimento", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / f"{metric_cfg['file']}_con_{time_cfg['file']}_por_experimento.pdf"
    )
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def _plot_vd_means(tabla_vd_medias, n_values, metric_col, time_name):
    """Genera una grafica VD de medias entre experimentos."""
    metric_cfg = VD_METRICS[metric_col]
    time_cfg = VD_TIMES[time_name]
    data = tabla_vd_medias[
        (tabla_vd_medias["metrica_columna"] == metric_col)
        & (tabla_vd_medias["tipo_tiempo"] == time_name)
    ].sort_values("n")
    x_pos = np.arange(len(n_values))
    x_map = dict(zip(n_values, x_pos))

    plt.figure(figsize=(8, 5))
    plt.plot(
        data["n"].map(x_map),
        data["VD_normalizado"],
        marker="o",
        linewidth=2,
        markersize=5,
    )

    plt.xticks(x_pos, [str(int(n)) for n in n_values])
    plt.ylim(-0.05, 1.05)
    plt.xlabel("N")
    plt.ylabel(metric_cfg["ylabel"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOTS_DIR / f"{metric_cfg['file']}_medio_con_{time_cfg['file']}.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def _plot_metric_means(df_medias, n_values, metric_col):
    """Genera una grafica de la media de una metrica entre experimentos."""
    metric_cfg = MEAN_METRIC_PLOTS[metric_col]
    data = df_medias.sort_values("n")
    x_pos = np.arange(len(n_values))
    x_map = dict(zip(n_values, x_pos))

    plt.figure(figsize=(8, 5))
    plt.plot(
        data["n"].map(x_map),
        data[metric_col],
        marker="o",
        linewidth=2,
        markersize=5,
    )

    plt.xticks(x_pos, [str(int(n)) for n in n_values])
    plt.xlabel("N")
    plt.ylabel(metric_cfg["ylabel"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOTS_DIR / f"{metric_cfg['file']}.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def _clean_previous_plots():
    """Elimina PDFs antiguos para que la carpeta refleje la ejecucion actual."""
    for path in PLOTS_DIR.glob("*.pdf"):
        path.unlink()


def _build_vd_outputs(df_summary):
    """Genera las tablas VD y las graficas principales para A2."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _clean_previous_plots()

    df_base = _prepare_vd_base(df_summary)
    n_values = sorted(df_base["n"].dropna().unique())
    group_cols = [
        col
        for col in [
            "experiment",
            "descripcion_experimento",
            "model_family",
            "model",
            "scaler",
            "use_pca",
            "n_components_pca",
            "rebalanceo_short",
        ]
        if col in df_base.columns
    ]

    tabla_vd_experimentos = _build_vd_table(df_base, group_cols)

    mean_cols = [
        "n",
        *VD_METRICS.keys(),
        "B_fit_time_mean",
        "B_score_time_mean",
        "B_total_time_mean",
    ]
    df_medias = df_base[mean_cols].groupby("n", as_index=False).mean()
    tabla_vd_medias = _build_vd_table(df_medias, [])

    vd_experiments_path = (
        OUTPUT_DIR / f"{DATASET_ORIGINAL}__A2__tabla_vd_por_experimento.csv"
    )
    vd_means_path = (
        OUTPUT_DIR / f"{DATASET_ORIGINAL}__A2__tabla_vd_medias_experimentos.csv"
    )
    tabla_vd_experimentos.to_csv(vd_experiments_path, index=False)
    tabla_vd_medias.to_csv(vd_means_path, index=False)

    plot_paths = [
        _plot_metric_means(df_medias, n_values, "B_f1_macro_mean"),
        _plot_vd_means(
            tabla_vd_medias,
            n_values,
            "B_f1_macro_mean",
            "total_time",
        ),
        _plot_metric_means(df_medias, n_values, "B_f1_macro_std"),
    ]

    return vd_experiments_path, vd_means_path, plot_paths


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df_summary = _build_summary_table()
    except FileNotFoundError as exc:
        print(exc)
        return 0

    df_means = _build_mean_metrics_table(df_summary)
    df_folds = _build_fold_table()
    vd_experiments_path, vd_means_path, plot_paths = _build_vd_outputs(df_summary)

    summary_path = (
        OUTPUT_DIR
        / f"{DATASET_ORIGINAL}__A2__super_table_all_experiments_all_metrics.csv"
    )
    means_path = (
        OUTPUT_DIR
        / f"{DATASET_ORIGINAL}__A2__super_table_all_experiments_mean_metrics.csv"
    )
    excel_path = (
        OUTPUT_DIR / f"{DATASET_ORIGINAL}__A2__super_table_all_experiments.xlsx"
    )

    df_summary.to_csv(summary_path, index=False)
    df_means.to_csv(means_path, index=False)

    try:
        with pd.ExcelWriter(excel_path) as writer:
            df_summary.to_excel(writer, sheet_name="all_metrics", index=False)
            df_means.to_excel(writer, sheet_name="mean_metrics", index=False)
            if not df_folds.empty:
                df_folds.to_excel(writer, sheet_name="fold_results", index=False)
    except ImportError:
        excel_path = None

    print("Tabla comparativa A2 generada.")
    print(f"Filas resumen: {len(df_summary)}")
    print(f"CSV completo: {summary_path}")
    print(f"CSV medias: {means_path}")
    print(f"CSV VD por experimento: {vd_experiments_path}")
    print(f"CSV VD medias experimentos: {vd_means_path}")
    print(f"Graficas: {len(plot_paths)} PDFs en {PLOTS_DIR}")
    if excel_path is not None:
        print(f"Excel: {excel_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
