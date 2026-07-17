"""Estrategias de rebalanceo propias de la fase A2."""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, RandomUnderSampler


VALID_SAMPLING_STRATEGIES = {
    "NONE",
    "RUS_SMOTE",
    "RUS_SMOTE_ENN",
    "NearMiss_SMOTE",
    "NearMiss_SMOTE_ENN",
}


def rebalancear_train_fold(
    df_fold_train,
    label_col,
    estrategia_rebalanceo,
    target_n,
    random_state=42,
    nearmiss_version=1,
    smote_k_neighbors=5,
    enn_n_neighbors=3,
):
    """Rebalancea solo el train de un fold con la estrategia indicada."""
    if estrategia_rebalanceo not in VALID_SAMPLING_STRATEGIES:
        raise ValueError(f"Estrategia no valida: {estrategia_rebalanceo}")

    df_fold_train = df_fold_train.copy()

    if estrategia_rebalanceo == "NONE":
        return df_fold_train

    X = df_fold_train.drop(columns=[label_col])
    y = df_fold_train[label_col]

    conteos = y.value_counts()
    sampling_under = {
        clase: target_n
        for clase, n in conteos.items()
        if n > target_n
    }

    if sampling_under:
        if estrategia_rebalanceo.startswith("RUS"):
            undersampler = RandomUnderSampler(
                sampling_strategy=sampling_under,
                random_state=random_state,
            )
        else:
            undersampler = NearMiss(
                sampling_strategy=sampling_under,
                version=nearmiss_version,
                n_jobs=-1,
            )

        X_res, y_res = undersampler.fit_resample(X, y)
    else:
        X_res, y_res = X.copy(), y.copy()

    conteos_res = pd.Series(y_res).value_counts()
    sampling_over = {
        clase: target_n
        for clase, n in conteos_res.items()
        if n < target_n and n >= 2
    }

    if sampling_over:
        min_clase_smote = min(conteos_res[clase] for clase in sampling_over)
        k_neighbors_real = min(smote_k_neighbors, min_clase_smote - 1)

        smote = SMOTE(
            sampling_strategy=sampling_over,
            k_neighbors=k_neighbors_real,
            random_state=random_state,
        )
        X_res, y_res = smote.fit_resample(X_res, y_res)

    if estrategia_rebalanceo.endswith("_ENN"):
        enn = EditedNearestNeighbours(
            n_neighbors=enn_n_neighbors,
            n_jobs=-1,
        )
        X_res, y_res = enn.fit_resample(X_res, y_res)

    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res[label_col] = y_res

    return df_res
