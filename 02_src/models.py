"""Factorias de modelos para experimentos."""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def crear_modelo_knn(n_neighbors=5, weights="distance", metric="minkowski", p=2):
    """Crea un clasificador KNN."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p,
    )


def crear_modelo_logreg(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    class_weight=None,
    n_jobs=-1,
    random_state=42,
):
    """Crea un modelo de regresion logistica."""
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def crear_modelo_mlp(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size="auto",
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=False,
    random_state=42,
):
    """Crea un MLPClassifier."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        verbose=verbose,
        random_state=random_state,
    )


def crear_modelo(model_name, model_params=None, random_state=42):
    """Crea un modelo por nombre a partir de un diccionario de parametros."""
    model_params = dict(model_params or {})

    if model_name == "knn":
        return crear_modelo_knn(**model_params)

    if model_name == "logreg":
        model_params.setdefault("random_state", random_state)
        return crear_modelo_logreg(**model_params)

    if model_name == "mlp":
        model_params.setdefault("random_state", random_state)
        return crear_modelo_mlp(**model_params)

    raise ValueError(f"Modelo no soportado: {model_name}")
