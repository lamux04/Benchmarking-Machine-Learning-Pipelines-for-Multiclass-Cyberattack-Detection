"""Runner comun para experimentos A2 de sampling size."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import experiment_utils
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


def _build_experiment_description(
    scaler_name,
    use_pca,
    estrategia_rebalanceo,
    model_name,
):
    pca_label = "PCA" if use_pca else "no PCA"
    return f"{scaler_name} + {pca_label} + {estrategia_rebalanceo} + {model_name}"


def ejecutar_experimento_a2_sampling_size(config):
    """Ejecuta un experimento A2 basado en scaler/PCA/sampling/modelo."""
    # El runner recibe toda la configuracion desde el script de experimento.
    # Asi se puede reutilizar para cambiar scaler, PCA, sampler o modelo sin
    # duplicar el bucle de validacion cruzada.
    project_root = Path(config["project_root"])
    dataset_original = config["dataset_original"]
    model_family = config["model_family"]
    experiment_name = config["experiment_name"]

    label_col = config["label_col"]
    n_splits = config["n_splits"]
    shuffle = config["shuffle"]
    random_state = config["random_state"]
    n_values = config["n_values"]

    scaler_name = config["scaler_name"]
    use_pca = config["use_pca"]
    n_components_pca = config["n_components_pca"]
    estrategia_rebalanceo = config["estrategia_rebalanceo"]
    estrategia_rebalanceo_corta = config["estrategia_rebalanceo_corta"]

    model_name = config["model_name"]
    model_display_name = config.get("model_display_name", model_name)
    model_params = dict(config.get("model_params", {}))

    if not model_params:
        raise ValueError("El experimento debe definir model_params explicitamente.")

    nearmiss_version = config["nearmiss_version"]
    smote_k_neighbors = config["smote_k_neighbors"]
    enn_n_neighbors = config["enn_n_neighbors"]

    cv_fold_results = config.get("cv_fold_results", [])
    cv_summary = config.get("cv_summary", [])
    experiment_parameters = config.get("experiment_parameters", [])

    nombre_dataset_entrada = f"{dataset_original}__split"
    nombre_train = f"{nombre_dataset_entrada}__train.csv"

    ruta_dataset_entrada = project_root / "01_datasets" / dataset_original / "split"
    ruta_salida_resultados = (
        project_root
        / "05_results"
        / dataset_original
        / "fase_A2_sampling_size"
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

    # Solo se carga train. Esta fase no evalua hold-out: todo el analisis se
    # hace con validacion cruzada sobre el 80% de entrenamiento ya generado.
    print("Cargando dataset de entrenamiento...")
    df_train = preprocessing.cargar_dataset(
        nombre_dataset=nombre_train,
        ruta_base=ruta_dataset_entrada,
    )

    # Asegura que LABEL existe y que todas las features son numericas antes de
    # entrar en escalado, PCA o modelos de scikit-learn.
    experiment_utils.validar_dataset_numerico(df_train, label_col, "train")

    X_train = df_train.drop(columns=[label_col])
    y_train = df_train[label_col]
    labels_globales = np.array(sorted(pd.unique(y_train)))

    # StratifiedKFold mantiene la proporcion de clases en cada fold.
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    # Primer reporte: fotografia completa de la configuracion usada. Incluye
    # parametros del modelo con prefijo model_param_* para que sea comparable
    # entre KNN, logreg y MLP.
    parameter_report = {
        "phase": "A2_sampling_size",
        "model_family": model_family,
        "model": model_name,
        "model_display_name": model_display_name,
        "scaler": scaler_name,
        "use_pca": use_pca,
        "n_components_pca": n_components_pca,
        "rebalanceo": estrategia_rebalanceo,
        "rebalanceo_short": estrategia_rebalanceo_corta,
        "label_col": label_col,
        "n_splits": n_splits,
        "shuffle": shuffle,
        "random_state": random_state,
        "n_values": ",".join(map(str, n_values)),
        "input_dataset": nombre_train,
        "rows_train": len(df_train),
        "features_train": X_train.shape[1],
        "nearmiss_version": nearmiss_version,
        "smote_k_neighbors": smote_k_neighbors,
        "enn_n_neighbors": enn_n_neighbors,
    }
    parameter_report.update(
        {
            f"model_param_{key}": _stringify_config_value(value)
            for key, value in model_params.items()
        }
    )

    reporting.registrar_parametros_experimento(
        experiment_name=experiment_name,
        parameters=parameter_report,
    )

    # Se guarda tambien la distribucion original de clases en train para
    # interpretar cuanto cambia cada valor de n tras el rebalanceo.
    train_label_distribution = (
        df_train[label_col]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )
    train_label_distribution["proportion"] = (
        train_label_distribution["count"] / len(df_train)
    )

    print("Iniciando validacion cruzada...")

    # Bucle externo: cada target_n define el tamano objetivo por clase que
    # intentaran alcanzar las estrategias de under/over-sampling.
    for target_n in n_values:
        print("=" * 80)
        print(f"n = {target_n}")
        print("=" * 80)

        # Bucle interno: en cada fold se separan A (train del fold) y B
        # (validacion). B nunca se rebalancea ni se usa para ajustar scaler/PCA.
        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_train, y_train),
            start=1,
        ):
            print(f"Fold {fold}/{n_splits}")

            df_train_fold = df_train.iloc[train_idx].copy()
            df_val_fold = df_train.iloc[val_idx].copy()

            # Escalado y PCA opcional se ajustan solo con A. Despues se aplica
            # la transformacion aprendida a B para evitar leakage.
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

            # El rebalanceo se aplica solo sobre A ya transformado. B queda con
            # su distribucion real para medir rendimiento de validacion.
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

            # La factoria de modelos permite usar knn/logreg/mlp con el mismo
            # runner. Los parametros concretos vienen del script de experimento.
            modelo = models.crear_modelo(
                model_name=model_name,
                model_params=model_params,
                random_state=random_state,
            )

            # Medimos por separado tiempo de entrenamiento y tiempo de
            # prediccion para poder reportar tambien latencia por muestra.
            t0 = time.time()
            modelo.fit(X_A, y_A)
            fit_time = time.time() - t0

            t0 = time.time()
            y_pred_B = modelo.predict(X_B)
            score_time = time.time() - t0

            # Las metricas se calculan solo sobre B: accuracy, precision,
            # recall, F1, MCC, AUC-ROC, FPR/FNR y latencia.
            metricas_B = metrics.calcular_metricas_clasificacion(
                modelo=modelo,
                X_eval=X_B,
                y_eval=y_B,
                y_pred=y_pred_B,
                labels_globales=labels_globales,
                score_time=score_time,
            )

            descripcion_experimento = _build_experiment_description(
                scaler_name=scaler_name,
                use_pca=use_pca,
                estrategia_rebalanceo=estrategia_rebalanceo,
                model_name=model_name,
            )

            # Una fila por fold y por n. Este es el reporte mas granular y el
            # origen del resumen agregado posterior.
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
                "fold": int(fold),
                "train_original_rows_A": int(df_train_fold.shape[0]),
                "train_balanceado_rows_A": int(df_train_fold_balanceado.shape[0]),
                "val_rows_B": int(df_val_fold.shape[0]),
                "B_fit_time": float(fit_time),
                "B_score_time": float(score_time),
            }

            for nombre, valor in metricas_B.items():
                fila[f"B_{nombre}"] = valor

            reporting.registrar_resultado_cv_fold(experiment_name, fila)

            print(
                f"B -> acc={metricas_B['accuracy']:.6f}, "
                f"f1_macro={metricas_B['f1_macro']:.6f}, "
                f"mcc={metricas_B['mcc']:.6f}, "
                f"fpr_macro={metricas_B['fpr_macro']:.6f}, "
                f"fnr_macro={metricas_B['fnr_macro']:.6f}, "
                f"auc={metricas_B['roc_auc']:.6f}, "
                f"latency={metricas_B['detection_latency_seconds']:.9f}s"
            )

    # Al terminar todos los folds, se agregan metricas por n: media,
    # desviacion tipica y varianza entre folds.
    df_folds = pd.DataFrame(cv_fold_results)
    summary_metadata = {
        "descripcion_experimento": _build_experiment_description(
            scaler_name=scaler_name,
            use_pca=use_pca,
            estrategia_rebalanceo=estrategia_rebalanceo,
            model_name=model_name,
        ),
        "scaler": scaler_name,
        "use_pca": use_pca,
        "n_components_pca": n_components_pca,
        "rebalanceo": estrategia_rebalanceo,
        "rebalanceo_short": estrategia_rebalanceo_corta,
        "model": model_name,
        "model_display_name": model_display_name,
    }

    df_resumen = metrics.resumir_metricas_cv(df_folds, metadata=summary_metadata)
    df_tabla_cv = metrics.crear_tabla_publicacion_cv(df_resumen)

    # Se registra el resumen agregado en el sistema comun de reporting.
    for _, row in df_resumen.iterrows():
        reporting.registrar_resumen_cv(experiment_name, row.to_dict())

    # Los CSV principales salen desde reporting.py; los auxiliares especificos
    # de esta fase se guardan aqui junto al resto de reportes.
    print("Guardando reportes...")
    reporting.guardar_reportes_cv_experimento(experiment_name)

    train_label_distribution.to_csv(
        ruta_salida_reportes
        / f"{dataset_original}__{experiment_name}__train_label_distribution.csv",
        index=False,
    )
    df_tabla_cv.to_csv(
        ruta_salida_reportes / f"{dataset_original}__{experiment_name}__cv_table.csv",
        index=False,
    )

    # Graficas compactas para comparar como evoluciona el rendimiento al variar
    # n. Se guardan en PDF para incluirlas facilmente en memoria/articulo.
    print("Guardando graficas...")
    graficas = [
        ("B_f1_macro_mean", "CV - Mean F1 macro", "Mean F1 macro", "cv_mean_f1_macro"),
        ("B_f1_macro_var", "CV - F1 macro variance", "F1 macro variance", "cv_var_f1_macro"),
        ("B_accuracy_mean", "CV - Mean accuracy", "Mean accuracy", "cv_mean_accuracy"),
        ("B_mcc_mean", "CV - Mean MCC", "Mean MCC", "cv_mean_mcc"),
        ("B_roc_auc_mean", "CV - Mean AUC-ROC", "Mean AUC-ROC", "cv_mean_roc_auc"),
    ]

    for y_col, title, ylabel, suffix in graficas:
        experiment_utils.guardar_grafica_metrica(
            df_resumen,
            y_col=y_col,
            title=title,
            ylabel=ylabel,
            output_path=(
                ruta_salida_graficas
                / f"{dataset_original}__{experiment_name}__{suffix}.pdf"
            ),
        )

    print("Experimento finalizado correctamente.")
    print(f"Reportes guardados en: {ruta_salida_reportes}")
    print(f"Graficas guardadas en: {ruta_salida_graficas}")
