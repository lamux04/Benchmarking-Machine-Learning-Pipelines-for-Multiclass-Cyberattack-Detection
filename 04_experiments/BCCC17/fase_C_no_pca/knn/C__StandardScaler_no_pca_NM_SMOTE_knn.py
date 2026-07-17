"""Experimento C: StandardScaler + sin PCA + NearMiss/SMOTE + GridSearch KNN."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "02_src"
sys.path.append(str(SRC_DIR))
from gridsearch_runner import ejecutar_experimento_c_no_pca_gridsearch


# ============== DATASET Y EXPERIMENTO ================
DATASET_ORIGINAL = "BCCC17"
MODEL_FAMILY = "knn"
EXPERIMENT_NAME = "C__StandardScaler_no_pca_NM_SMOTE_knn"

# ============== VALIDACION CRUZADA ================
LABEL_COL = "LABEL"
N_SPLITS = 5
SHUFFLE = True
RANDOM_STATE = 42

# ============== TRANSFORMACION DE FEATURES ================
SCALER_NAME = "StandardScaler"
N_COMPONENTS_PCA = None

# ============== REBALANCEO ================
# TARGET_N viene de la fase A2: tamano objetivo por clase para el rebalanceo.
# Si ESTRATEGIA_REBALANCEO = "NONE", este valor se reporta pero no se aplica.
TARGET_N = 10000
ESTRATEGIA_REBALANCEO = "NearMiss_SMOTE"
ESTRATEGIA_REBALANCEO_CORTA = "NM_SMOTE"

# ============== MODELO KNN ================
MODEL_NAME = "knn"
MODEL_DISPLAY_NAME = "KNeighborsClassifier"

# Parametros fijos del modelo. Los hiperparametros candidatos van en PARAM_GRID.
MODEL_BASE_PARAMS = {
    "weights": "distance",
}

# GridSearch del experimento: se prueba cada combinacion dentro del CV.
PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "metric": ["euclidean", "manhattan"],
}

# Metrica que decide la mejor configuracion tras promediar los folds.
SELECTION_METRIC = "f1_macro"

# ============== PARAMETROS INTERNOS DE REBALANCEO ================
NEARMISS_VERSION = 1
SMOTE_K_NEIGHBORS = 5
ENN_N_NEIGHBORS = 3

# ============== ESTRUCTURAS PARA REPORTES ================
experiment_parameters = []
cv_fold_results = []
cv_summary = []


ejecutar_experimento_c_no_pca_gridsearch(
    {
        "project_root": PROJECT_ROOT,
        "dataset_original": DATASET_ORIGINAL,
        "model_family": MODEL_FAMILY,
        "experiment_name": EXPERIMENT_NAME,
        "label_col": LABEL_COL,
        "n_splits": N_SPLITS,
        "shuffle": SHUFFLE,
        "random_state": RANDOM_STATE,
        "scaler_name": SCALER_NAME,
        "n_components_pca": N_COMPONENTS_PCA,
        "target_n": TARGET_N,
        "estrategia_rebalanceo": ESTRATEGIA_REBALANCEO,
        "estrategia_rebalanceo_corta": ESTRATEGIA_REBALANCEO_CORTA,
        "model_name": MODEL_NAME,
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_base_params": MODEL_BASE_PARAMS,
        "param_grid": PARAM_GRID,
        "selection_metric": SELECTION_METRIC,
        "nearmiss_version": NEARMISS_VERSION,
        "smote_k_neighbors": SMOTE_K_NEIGHBORS,
        "enn_n_neighbors": ENN_N_NEIGHBORS,
        "experiment_parameters": experiment_parameters,
        "cv_fold_results": cv_fold_results,
        "cv_summary": cv_summary,
    }
)
