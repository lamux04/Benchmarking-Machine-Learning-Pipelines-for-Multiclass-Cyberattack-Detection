"""Funciones auxiliares para dividir datasets."""

import numpy as np


def dividir_train_test_stratified(df, label_col="LABEL", test_size=0.2, random_state=42):
    """Divide un dataset en train/test manteniendo la proporcion de clases."""
    if label_col not in df.columns:
        raise KeyError(f"La columna de etiquetas no existe: {label_col}")

    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    class_counts = df[label_col].value_counts()
    clases_con_una_muestra = class_counts[class_counts < 2]

    if not clases_con_una_muestra.empty:
        clases = ", ".join(map(str, clases_con_una_muestra.index.tolist()))
        raise ValueError(
            "No se puede hacer un split estratificado con clases de una sola "
            f"muestra. Clases afectadas: {clases}"
        )

    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    for _, group in df.groupby(label_col, sort=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)

        n_test = round(len(indices) * test_size)
        n_test = min(max(n_test, 1), len(indices) - 1)

        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df
