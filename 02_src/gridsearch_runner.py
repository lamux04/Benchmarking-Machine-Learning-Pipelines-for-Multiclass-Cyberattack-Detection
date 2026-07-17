"""Runner comun para experimentos con GridSearch en CV."""

from itertools import count
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import experiment_utils
import matplotlib.pyplot as plt
import metrics
import models
import preprocessing
import reporting
import sampling


def _stringify_config_value(value):
    """Convierte valores de configuracion a una representacion estable para CSV."""
    if isinstance(value, (list, tuple)):
        return ",".join(map(str, value))

    return value


def _params_to_text(params):
    """Serializa parametros para que se puedan comparar y guardar en CSV."""
    return json.dumps(params, sort_keys=True, default=str)


def _build_experiment_description(
    scaler_name,
    use_pca,
    estrategia_rebalanceo,
    model_name,
):
    pca_label = "PCA" if use_pca else "sin PCA"
    return f"{scaler_name} + {pca_label} + {estrategia_rebalanceo} + {model_name}"


def _merge_model_params(base_params, grid_params):
    """Une parametros fijos del modelo con los candidatos del GridSearch."""
    model_params = dict(base_params or {})
    model_params.update(grid_params)
    return model_params


def _calcular_metricas_modelo(
    modelo,
    X_eval,
    y_eval,
    labels_globales,
    return_predictions=False,
):
    """Predice y calcula metricas midiendo tambien el tiempo de prediccion."""
    t0 = time.time()
    y_pred = modelo.predict(X_eval)
    score_time = time.time() - t0

    metricas = metrics.calcular_metricas_clasificacion(
        modelo=modelo,
        X_eval=X_eval,
        y_eval=y_eval,
        y_pred=y_pred,
        labels_globales=labels_globales,
        score_time=score_time,
    )
    metricas["score_time"] = score_time

    if return_predictions:
        return metricas, y_pred

    return metricas


def _guardar_matriz_confusion_holdout(
    y_true,
    y_pred,
    labels_globales,
    csv_path,
    figure_path,
):
    """Guarda la matriz de confusion del hold-out como tabla y grafica."""
    cm = confusion_matrix(y_true, y_pred, labels=labels_globales)

    df_cm = pd.DataFrame(cm, index=labels_globales, columns=labels_globales)
    df_cm.index.name = "true_label"
    df_cm.columns.name = "predicted_label"
    df_cm.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels_globales,
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title("Matriz de confusion - hold-out")
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(figure_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _guardar_predicciones_holdout(y_true, y_pred, output_path):
    """Guarda el detalle de prediccion de cada fila del hold-out."""
    df_predicciones = pd.DataFrame(
        {
            "holdout_index": pd.Series(y_true).index,
            "true_label": pd.Series(y_true).to_numpy(),
            "predicted_label": np.asarray(y_pred),
        }
    )
    df_predicciones["is_correct"] = (
        df_predicciones["true_label"] == df_predicciones["predicted_label"]
    )
    df_predicciones.to_csv(output_path, index=False)


def _guardar_grafica_gridsearch(df_summary, selection_col, output_path):
    """Guarda una grafica sencilla con el score medio de cada configuracion."""
    if selection_col not in df_summary.columns:
        return

    df_plot = df_summary.sort_values(selection_col, ascending=False).copy()

    plt.figure(figsize=(9, 5))
    plt.bar(df_plot["param_id"].astype(str), df_plot[selection_col])
    plt.xlabel("Configuracion de hiperparametros")
    plt.ylabel(selection_col)
    plt.title("GridSearch CV")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def ejecutar_experimento_gridsearch(config):
    """Ejecuta un experimento con scaler, PCA opcional, sampler y GridSearch."""
    project_root = Path(config["project_root"])
    dataset_original = config["dataset_original"]
    model_family = config["model_family"]
    experiment_name = config["experiment_name"]
    phase = config.get("phase", "B_pca")
    results_phase_dir = config.get("results_phase_dir", "fase_B_pca")

    label_col = config["label_col"]
    n_splits = config["n_splits"]
    shuffle = config["shuffle"]
    random_state = config["random_state"]

    scaler_name = config["scaler_name"]
    use_pca = config.get("use_pca", True)
    n_components_pca = config.get("n_components_pca")
    estrategia_rebalanceo = config["estrategia_rebalanceo"]
    estrategia_rebalanceo_corta = config["estrategia_rebalanceo_corta"]
    target_n = config["target_n"]

    model_name = config["model_name"]
    model_display_name = config.get("model_display_name", model_name)
    model_base_params = dict(config.get("model_base_params", {}))
    param_grid = dict(config["param_grid"])
    selection_metric = config.get("selection_metric", "f1_macro")

    nearmiss_version = config["nearmiss_version"]
    smote_k_neighbors = config["smote_k_neighbors"]
    enn_n_neighbors = config["enn_n_neighbors"]

    cv_fold_results = config.get("cv_fold_results", [])
    cv_summary = config.get("cv_summary", [])
    experiment_parameters = config.get("experiment_parameters", [])

    nombre_dataset_entrada = f"{dataset_original}__split"
    nombre_train = f"{nombre_dataset_entrada}__train.csv"
    nombre_test = f"{nombre_dataset_entrada}__test.csv"

    ruta_dataset_entrada = project_root / "01_datasets" / dataset_original / "split"
    ruta_salida_resultados = (
        project_root
        / "05_results"
        / dataset_original
        / results_phase_dir
        / model_family
        / experiment_name
    )
    ruta_salida_reportes = ruta_salida_resultados / "reports"
    ruta_salida_graficas = ruta_salida_resultados / "figures"

    ruta_salida_reportes.mkdir(parents=True, exist_ok=True)
    ruta_salida_graficas.mkdir(parents=True, exist_ok=True)

    reporting.configurar_reportes(
        dataset_original=dataset_original,
        ruta_salida_reportes=ruta_salida_reportes,
        label_col=label_col,
        parametros_experimento=experiment_parameters,
        resultados_cv_fold=cv_fold_results,
        resumen_cv=cv_summary,
    )

    print("Cargando particiones train/test...")
    df_train = preprocessing.cargar_dataset(
        nombre_dataset=nombre_train,
        ruta_base=ruta_dataset_entrada,
    )
    df_test = preprocessing.cargar_dataset(
        nombre_dataset=nombre_test,
        ruta_base=ruta_dataset_entrada,
    )

    experiment_utils.validar_dataset_numerico(df_train, label_col, "train")
    experiment_utils.validar_dataset_numerico(df_test, label_col, "test")

    X_train = df_train.drop(columns=[label_col])
    y_train = df_train[label_col]
    y_test = df_test[label_col]
    labels_globales = np.array(sorted(pd.unique(pd.concat([y_train, y_test]))))

    descripcion_experimento = _build_experiment_description(
        scaler_name=scaler_name,
        use_pca=use_pca,
        estrategia_rebalanceo=estrategia_rebalanceo,
        model_name=model_name,
    )

    parameter_report = {
        "phase": phase,
        "model_family": model_family,
        "model": model_name,
        "model_display_name": model_display_name,
        "scaler": scaler_name,
        "use_pca": use_pca,
        "n_components_pca": n_components_pca,
        "rebalanceo": estrategia_rebalanceo,
        "rebalanceo_short": estrategia_rebalanceo_corta,
        "target_n": target_n,
        "selection_metric": selection_metric,
        "label_col": label_col,
        "n_splits": n_splits,
        "shuffle": shuffle,
        "random_state": random_state,
        "input_train": nombre_train,
        "input_test": nombre_test,
        "rows_train": len(df_train),
        "rows_test": len(df_test),
        "features_train": X_train.shape[1],
        "nearmiss_version": nearmiss_version,
        "smote_k_neighbors": smote_k_neighbors,
        "enn_n_neighbors": enn_n_neighbors,
        "param_grid": _params_to_text(param_grid),
    }
    parameter_report.update(
        {
            f"model_base_param_{key}": _stringify_config_value(value)
            for key, value in model_base_params.items()
        }
    )

    reporting.registrar_parametros_experimento(
        experiment_name=experiment_name,
        parameters=parameter_report,
    )

    # Cacula la distribución de clases de train
    train_label_distribution = (
        y_train.value_counts(dropna=False)
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )
    train_label_distribution["proportion"] = (
        train_label_distribution["count"] / len(df_train)
    )

    # Calcula la distribución de clases de test
    test_label_distribution = (
        y_test.value_counts(dropna=False)
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )
    test_label_distribution["proportion"] = (
        test_label_distribution["count"] / len(df_test)
    )

    # Creamos el StrtifiedKFold para crear las particiones del cross-validation
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    # Creamos la grilla de parámetros y le asignamos un id a cada combinación para poder usarla en los reports
    grid = list(ParameterGrid(param_grid))
    param_ids = {
        _params_to_text(params): param_id
        for params, param_id in zip(grid, count(start=1))
    }

    print("Iniciando GridSearch dentro de validacion cruzada...")

    # Cada combinacion de hiperparametros se evalua con los mismos folds. En
    # cada fold se ajusta scaler/PCA solo con A y se rebalancea solo A.
    for grid_params in grid: # Por cada combinación de hiperparámetros
        param_key = _params_to_text(grid_params)
        param_id = param_ids[param_key]
        model_params = _merge_model_params(model_base_params, grid_params)

        print("=" * 80)
        print(f"Configuracion {param_id}/{len(grid)}: {param_key}")
        print("=" * 80)

        # Creamos las particiones del cross-validation y entramos en el bucle del CV
        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_train, y_train),
            start=1,
        ):
            print(f"Fold {fold}/{n_splits}")

            df_train_fold = df_train.iloc[train_idx].copy()
            df_val_fold = df_train.iloc[val_idx].copy()

            # Aplica escalado y PCA
            # Calcula parámetros escalado y PCA solo con A y los aplica a A y B
            X_A_transformed, X_B = experiment_utils.transformar_features(
                df_train_fold=df_train_fold,
                df_val_fold=df_val_fold,
                label_col=label_col,
                scaler_name=scaler_name,
                use_pca=use_pca,
                n_components_pca=n_components_pca,
                random_state=random_state,
            )
            y_A_original = df_train_fold[label_col]
            y_B = df_val_fold[label_col]

            df_train_fold_transformed = X_A_transformed.copy()
            df_train_fold_transformed[label_col] = y_A_original

            # Rebalancea A
            df_train_fold_balanceado = sampling.rebalancear_train_fold(
                df_fold_train=df_train_fold_transformed,
                label_col=label_col,
                estrategia_rebalanceo=estrategia_rebalanceo,
                target_n=target_n,
                random_state=random_state + fold,
                nearmiss_version=nearmiss_version,
                smote_k_neighbors=smote_k_neighbors,
                enn_n_neighbors=enn_n_neighbors,
            )

            X_A = df_train_fold_balanceado.drop(columns=[label_col])
            y_A = df_train_fold_balanceado[label_col]

            # Crea el modelo con los parámetros de la configuración actual
            modelo = models.crear_modelo(
                model_name=model_name,
                model_params=model_params,
                random_state=random_state,
            )

            # Entrena el modelo
            t0 = time.time()
            modelo.fit(X_A, y_A)
            fit_time = time.time() - t0

            # Valida el modelo con la partición de validación
            metricas_B = _calcular_metricas_modelo(
                modelo=modelo,
                X_eval=X_B,
                y_eval=y_B,
                labels_globales=labels_globales,
            )
            metricas_B["fit_time"] = fit_time

            fila = {
                "descripcion_experimento": descripcion_experimento,
                "scaler": scaler_name,
                "use_pca": use_pca,
                "n_components_pca": n_components_pca,
                "rebalanceo": estrategia_rebalanceo,
                "rebalanceo_short": estrategia_rebalanceo_corta,
                "model": model_name,
                "model_display_name": model_display_name,
                "n": int(target_n),
                "param_id": int(param_id),
                "params": param_key,
                "fold": int(fold),
                "train_original_rows_A": int(df_train_fold.shape[0]),
                "train_balanceado_rows_A": int(df_train_fold_balanceado.shape[0]),
                "val_rows_B": int(df_val_fold.shape[0]),
            }
            fila.update({f"param_{key}": value for key, value in grid_params.items()})

            for nombre, valor in metricas_B.items():
                fila[f"B_{nombre}"] = valor

            reporting.registrar_resultado_cv_fold(experiment_name, fila)

            print(
                f"B -> acc={metricas_B['accuracy']:.6f}, "
                f"f1_macro={metricas_B['f1_macro']:.6f}, "
                f"mcc={metricas_B['mcc']:.6f}, "
                f"auc={metricas_B['roc_auc']:.6f}"
            )

    # Obtenemos en df_folds los resultados obtenidos durante las diferentes configuraciones
    df_folds = pd.DataFrame(cv_fold_results)
    summary_metadata = {
        "descripcion_experimento": descripcion_experimento,
        "scaler": scaler_name,
        "use_pca": use_pca,
        "n_components_pca": n_components_pca,
        "rebalanceo": estrategia_rebalanceo,
        "rebalanceo_short": estrategia_rebalanceo_corta,
        "model": model_name,
        "model_display_name": model_display_name,
    }

    param_cols = [col for col in df_folds.columns if col.startswith("param_")]
    group_cols = ["n", "param_id", "params"] + param_cols

    df_resumen = metrics.resumir_metricas_cv(
        df_folds=df_folds,
        metadata=summary_metadata,
        group_cols=group_cols,
        sort_by=["param_id"],
    )

    selection_col = f"B_{selection_metric}_mean"
    selection_std_col = f"B_{selection_metric}_std"

    if selection_col not in df_resumen.columns:
        raise ValueError(f"No existe la metrica de seleccion: {selection_col}")

    # Ordena las diferentes configuraciones
    df_ranking = df_resumen.sort_values(
        by=[selection_col, selection_std_col, "B_fit_time_mean"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    df_ranking.insert(0, "rank", range(1, len(df_ranking) + 1))

    # Localiza la mejor configuración y los mejores parámetros de esa configuración
    best_row = df_ranking.iloc[0].to_dict()
    best_params = grid[int(best_row["param_id"]) - 1]
    best_model_params = _merge_model_params(model_base_params, best_params)

    for _, row in df_resumen.iterrows():
        reporting.registrar_resumen_cv(experiment_name, row.to_dict())

    print("Reentrenando el mejor pipeline con el 80% completo...")

    # Calcula parámetros de escalado y PCA solo con el 80% y lo aplica a train y test
    X_train_final, X_test_final = experiment_utils.transformar_features(
        df_train_fold=df_train,
        df_val_fold=df_test,
        label_col=label_col,
        scaler_name=scaler_name,
        use_pca=use_pca,
        n_components_pca=n_components_pca,
        random_state=random_state,
    )

    df_train_final_transformed = X_train_final.copy()
    df_train_final_transformed[label_col] = y_train

    # Rebalancea train
    df_train_final_balanceado = sampling.rebalancear_train_fold(
        df_fold_train=df_train_final_transformed,
        label_col=label_col,
        estrategia_rebalanceo=estrategia_rebalanceo,
        target_n=target_n,
        random_state=random_state,
        nearmiss_version=nearmiss_version,
        smote_k_neighbors=smote_k_neighbors,
        enn_n_neighbors=enn_n_neighbors,
    )

    X_final = df_train_final_balanceado.drop(columns=[label_col])
    y_final = df_train_final_balanceado[label_col]

    # Crea el modelo elegido en la grilla
    modelo_final = models.crear_modelo(
        model_name=model_name,
        model_params=best_model_params,
        random_state=random_state,
    )

    # Entrena con train
    t0 = time.time()
    modelo_final.fit(X_final, y_final)
    final_fit_time = time.time() - t0

    # Valida el modelo con el test
    metricas_holdout, y_pred_holdout = _calcular_metricas_modelo(
        modelo=modelo_final,
        X_eval=X_test_final,
        y_eval=y_test,
        labels_globales=labels_globales,
        return_predictions=True,
    )
    metricas_holdout["fit_time"] = final_fit_time

    holdout_row = {
        "dataset": dataset_original,
        "experiment": experiment_name,
        "descripcion_experimento": descripcion_experimento,
        "scaler": scaler_name,
        "use_pca": use_pca,
        "n_components_pca": n_components_pca,
        "rebalanceo": estrategia_rebalanceo,
        "rebalanceo_short": estrategia_rebalanceo_corta,
        "model": model_name,
        "model_display_name": model_display_name,
        "n": int(target_n),
        "best_param_id": int(best_row["param_id"]),
        "best_params": _params_to_text(best_params),
        "train_original_rows": int(df_train.shape[0]),
        "train_balanceado_rows": int(df_train_final_balanceado.shape[0]),
        "test_rows": int(df_test.shape[0]),
    }
    holdout_row.update({f"best_param_{key}": value for key, value in best_params.items()})
    holdout_row.update({f"H_{key}": value for key, value in metricas_holdout.items()})

    best_config_row = {
        "dataset": dataset_original,
        "experiment": experiment_name,
        "selection_metric": selection_metric,
        "selection_column": selection_col,
        "best_param_id": int(best_row["param_id"]),
        "best_params": _params_to_text(best_params),
    }
    best_config_row.update(
        {
            col: best_row[col]
            for col in df_ranking.columns
            if col.startswith("B_") or col.startswith("param_")
        }
    )

    print("Guardando reportes...")
    reporting.guardar_reportes_cv_experimento(experiment_name)

    train_label_distribution.to_csv(
        ruta_salida_reportes
        / f"{dataset_original}__{experiment_name}__train_label_distribution.csv",
        index=False,
    )
    test_label_distribution.to_csv(
        ruta_salida_reportes
        / f"{dataset_original}__{experiment_name}__test_label_distribution.csv",
        index=False,
    )
    df_ranking.to_csv(
        ruta_salida_reportes
        / f"{dataset_original}__{experiment_name}__gridsearch_ranking.csv",
        index=False,
    )
    pd.DataFrame([best_config_row]).to_csv(
        ruta_salida_reportes
        / f"{dataset_original}__{experiment_name}__best_config.csv",
        index=False,
    )
    pd.DataFrame([holdout_row]).to_csv(
        ruta_salida_reportes
        / f"{dataset_original}__{experiment_name}__holdout_metrics.csv",
        index=False,
    )
    _guardar_matriz_confusion_holdout(
        y_true=y_test,
        y_pred=y_pred_holdout,
        labels_globales=labels_globales,
        csv_path=(
            ruta_salida_reportes
            / f"{dataset_original}__{experiment_name}__holdout_confusion_matrix.csv"
        ),
        figure_path=(
            ruta_salida_graficas
            / f"{dataset_original}__{experiment_name}__holdout_confusion_matrix.pdf"
        ),
    )
    _guardar_predicciones_holdout(
        y_true=y_test,
        y_pred=y_pred_holdout,
        output_path=(
            ruta_salida_reportes
            / f"{dataset_original}__{experiment_name}__holdout_predictions.csv"
        ),
    )

    print("Guardando graficas...")
    _guardar_grafica_gridsearch(
        df_summary=df_ranking,
        selection_col=selection_col,
        output_path=(
            ruta_salida_graficas
            / f"{dataset_original}__{experiment_name}__gridsearch_{selection_metric}.pdf"
        ),
    )

    print("Experimento finalizado correctamente.")
    print(f"Mejor configuracion: {best_params}")
    print(f"Reportes guardados en: {ruta_salida_reportes}")
    print(f"Graficas guardadas en: {ruta_salida_graficas}")


def ejecutar_experimento_b_pca_gridsearch(config):
    """Alias compatible para los scripts de fase B existentes."""
    config = dict(config)
    config.setdefault("phase", "B_pca")
    config.setdefault("results_phase_dir", "fase_B_pca")
    config.setdefault("use_pca", True)
    return ejecutar_experimento_gridsearch(config)


def ejecutar_experimento_c_no_pca_gridsearch(config):
    """Ejecuta fase C: mismo flujo que B, pero con escalado sin PCA."""
    config = dict(config)
    config.setdefault("phase", "C_no_pca")
    config.setdefault("results_phase_dir", "fase_C_no_pca")
    config["use_pca"] = False
    config["n_components_pca"] = None
    return ejecutar_experimento_gridsearch(config)
