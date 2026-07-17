"""Utilidades comunes para scripts de experimentos."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler


def crear_scaler(nombre_scaler):
    """Crea un escalador por nombre."""
    if nombre_scaler == "StandardScaler":
        return StandardScaler()
    if nombre_scaler == "RobustScaler":
        return RobustScaler()

    raise ValueError(f"Scaler no soportado: {nombre_scaler}")


def validar_dataset_numerico(df, label_col, dataset_name):
    """Comprueba que el dataset tiene etiqueta y variables numericas."""
    if label_col not in df.columns:
        raise ValueError(f"No se encontro la columna {label_col} en {dataset_name}.")

    X = df.drop(columns=[label_col])
    columnas_no_numericas = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if columnas_no_numericas:
        raise ValueError(
            f"Hay columnas no numericas en {dataset_name}: {columnas_no_numericas}"
        )


def transformar_features(
    df_train_fold,
    df_val_fold,
    label_col,
    scaler_name,
    use_pca,
    n_components_pca=None,
    random_state=42,
):
    """Aplica escalado y PCA opcional ajustando solo con el fold de train."""
    feature_cols = [col for col in df_train_fold.columns if col != label_col]

    X_A_original = df_train_fold[feature_cols]
    X_B_original = df_val_fold[feature_cols]

    scaler = crear_scaler(scaler_name)

    X_A_scaled = pd.DataFrame(
        scaler.fit_transform(X_A_original),
        columns=feature_cols,
        index=df_train_fold.index,
    )
    X_B_scaled = pd.DataFrame(
        scaler.transform(X_B_original),
        columns=feature_cols,
        index=df_val_fold.index,
    )

    if not use_pca:
        return X_A_scaled, X_B_scaled

    pca = PCA(n_components=n_components_pca, random_state=random_state)
    pca_cols = [f"PC{i + 1}" for i in range(n_components_pca)]

    X_A = pd.DataFrame(
        pca.fit_transform(X_A_scaled),
        columns=pca_cols,
        index=df_train_fold.index,
    )
    X_B = pd.DataFrame(
        pca.transform(X_B_scaled),
        columns=pca_cols,
        index=df_val_fold.index,
    )

    return X_A, X_B


def guardar_grafica_metrica(df, y_col, title, ylabel, output_path):
    """Guarda una grafica de una metrica frente al tamano de muestreo."""
    if y_col not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df["n"], df[y_col], marker="o", linewidth=1.8)
    plt.xticks(df["n"])
    plt.xlabel("N instancias por clase")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
