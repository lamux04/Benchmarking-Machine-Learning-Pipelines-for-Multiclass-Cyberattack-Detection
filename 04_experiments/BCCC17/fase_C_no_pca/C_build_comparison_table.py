"""Construye tablas comparativas globales de resultados C."""

from pathlib import Path
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ORIGINAL = "BCCC17"
PHASE_DIR_NAME = "fase_C_no_pca"
PHASE_LABEL = "C"

RESULTS_DIR = PROJECT_ROOT / "05_results" / DATASET_ORIGINAL / PHASE_DIR_NAME
OUTPUT_DIR = RESULTS_DIR / "_comparison"

BEST_CONFIG_SUFFIX = "__best_config.csv"
CV_SUMMARY_SUFFIX = "__cv_summary.csv"
HOLDOUT_SUFFIX = "__holdout_metrics.csv"

METRIC_AGG_RE = re.compile(r"^B_.+_(mean|std)$")
METRIC_VAR_RE = re.compile(r"^B_.+_var$")
HOLDOUT_METRIC_RE = re.compile(r"^H_.+")
PARAM_RE = re.compile(r"^param_.+")


def _add_provenance(df, path):
    """Anade informacion de procedencia al dataframe."""
    relative_parts = path.relative_to(RESULTS_DIR).parts

    if len(relative_parts) >= 3 and "model_family" not in df.columns:
        df["model_family"] = relative_parts[0]

    df["source_file"] = str(path.relative_to(PROJECT_ROOT))
    return df


def _read_csv(path):
    """Lee un CSV y anade informacion de procedencia."""
    return _add_provenance(pd.read_csv(path), path)


def _find_report_files(suffix):
    """Localiza reportes C dentro de cualquier familia/modelo."""
    return sorted(RESULTS_DIR.glob(f"*/*/reports/*{suffix}"))


def _find_experiment_dirs():
    """Localiza directorios de experimento C."""
    return sorted(
        path
        for path in RESULTS_DIR.glob("*/*")
        if path.is_dir() and path.name != "_comparison"
    )


def _build_missing_reports_message(suffix):
    """Construye un diagnostico cuando aun no hay reportes C."""
    experiment_dirs = _find_experiment_dirs()

    if not experiment_dirs:
        return (
            "No se han encontrado experimentos C en resultados. "
            f"Ruta esperada: {RESULTS_DIR}/*/*"
        )

    lines = [
        "No se han encontrado reportes C para construir la comparativa.",
        f"Ruta esperada: {RESULTS_DIR}/*/*/reports/*{suffix}",
        "Directorios de experimento encontrados:",
    ]
    lines.extend(f"- {path.relative_to(PROJECT_ROOT)}" for path in experiment_dirs)
    lines.append(
        "Ejecuta primero los experimentos C; este script solo agrega los CSV "
        "que generan."
    )

    return "\n".join(lines)


def _report_dir_from_file(path):
    """Devuelve el directorio reports asociado a un CSV."""
    return path.parent


def _find_single_report(report_dir, suffix):
    """Busca un unico reporte auxiliar dentro del mismo experimento."""
    matches = sorted(report_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _merge_selection_columns(df_cv, df_best):
    """Incorpora metadatos de seleccion de best_config a la fila CV."""
    selection_cols = [
        "selection_metric",
        "selection_column",
        "best_param_id",
        "best_params",
    ]
    for col in selection_cols:
        if col in df_best.columns and col not in df_cv.columns:
            df_cv[col] = df_best.iloc[0][col]
    return df_cv


def _build_best_cv_row(best_config_path):
    """Construye la fila CV ganadora de un experimento."""
    report_dir = _report_dir_from_file(best_config_path)
    cv_summary_path = _find_single_report(report_dir, CV_SUMMARY_SUFFIX)
    df_best = _read_csv(best_config_path)

    if cv_summary_path is None:
        return df_best

    df_cv = _read_csv(cv_summary_path)
    best_param_id = None

    if "best_param_id" in df_best.columns:
        best_param_id = df_best.iloc[0]["best_param_id"]
    elif "param_id" in df_best.columns:
        best_param_id = df_best.iloc[0]["param_id"]

    if best_param_id is not None and "param_id" in df_cv.columns:
        df_cv = df_cv[df_cv["param_id"] == best_param_id]

    if df_cv.empty:
        df_cv = df_best
    else:
        df_cv = _merge_selection_columns(df_cv.copy(), df_best)

    return df_cv


def _ordered_columns(df, metric_re):
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
        "selection_metric",
        "selection_column",
        "best_param_id",
        "best_params",
        "param_id",
        "params",
    ]
    param_cols = sorted(
        col for col in df.columns if PARAM_RE.match(col) and col not in metadata_cols
    )
    metric_cols = sorted(col for col in df.columns if metric_re.match(col))
    other_cols = [
        col
        for col in df.columns
        if col not in metadata_cols
        and col not in param_cols
        and col not in metric_cols
        and not METRIC_VAR_RE.match(col)
    ]

    return (
        [col for col in metadata_cols if col in df.columns]
        + [col for col in param_cols if col in df.columns]
        + metric_cols
        + other_cols
    )


def _sort_table(df):
    """Aplica una ordenacion estable para comparar experimentos."""
    sort_cols = [
        "model_family",
        "model",
        "scaler",
        "use_pca",
        "n_components_pca",
        "rebalanceo_short",
        "experiment",
    ]
    sort_cols = [col for col in sort_cols if col in df.columns]

    if not sort_cols:
        return df.reset_index(drop=True)

    return df.sort_values(sort_cols).reset_index(drop=True)


def _build_best_cv_table():
    best_config_files = _find_report_files(BEST_CONFIG_SUFFIX)

    if not best_config_files:
        raise FileNotFoundError(_build_missing_reports_message(BEST_CONFIG_SUFFIX))

    df = pd.concat(
        (_build_best_cv_row(path) for path in best_config_files),
        ignore_index=True,
    )
    df = _sort_table(df)
    return df[_ordered_columns(df, METRIC_AGG_RE)]


def _build_holdout_table():
    holdout_files = _find_report_files(HOLDOUT_SUFFIX)

    if not holdout_files:
        raise FileNotFoundError(_build_missing_reports_message(HOLDOUT_SUFFIX))

    df = pd.concat((_read_csv(path) for path in holdout_files), ignore_index=True)
    df = _sort_table(df)
    return df[_ordered_columns(df, HOLDOUT_METRIC_RE)]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df_best_cv = _build_best_cv_table()
        df_holdout = _build_holdout_table()
    except FileNotFoundError as exc:
        print(exc)
        return 0

    best_cv_path = (
        OUTPUT_DIR
        / f"{DATASET_ORIGINAL}__{PHASE_LABEL}__super_table_best_cv_results.csv"
    )
    holdout_path = (
        OUTPUT_DIR
        / f"{DATASET_ORIGINAL}__{PHASE_LABEL}__super_table_holdout_results.csv"
    )
    excel_path = OUTPUT_DIR / f"{DATASET_ORIGINAL}__{PHASE_LABEL}__super_table.xlsx"

    df_best_cv.to_csv(best_cv_path, index=False)
    df_holdout.to_csv(holdout_path, index=False)

    try:
        with pd.ExcelWriter(excel_path) as writer:
            df_best_cv.to_excel(writer, sheet_name="best_cv_results", index=False)
            df_holdout.to_excel(writer, sheet_name="holdout_results", index=False)
    except ImportError:
        excel_path = None

    print("Tabla comparativa C generada.")
    print(f"Filas CV mejores configs: {len(df_best_cv)}")
    print(f"Filas hold-out: {len(df_holdout)}")
    print(f"CSV CV mejores configs: {best_cv_path}")
    print(f"CSV hold-out: {holdout_path}")
    if excel_path is not None:
        print(f"Excel: {excel_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
