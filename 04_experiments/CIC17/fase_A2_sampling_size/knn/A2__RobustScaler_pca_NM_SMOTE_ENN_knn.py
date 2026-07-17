"""Experimento A2: RobustScaler + PCA + NearMiss/SMOTE/ENN + KNN."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "02_src"
PHASE_DIR = Path(__file__).resolve().parents[1]

sys.path.append(str(SRC_DIR))
sys.path.append(str(PHASE_DIR))

from a2_sampling_size_runner import ejecutar_experimento_a2_sampling_size


# ============== DATASET Y EXPERIMENTO ================
DATASET_ORIGINAL = "CIC17"
MODEL_FAMILY = "knn"
EXPERIMENT_NAME = "A2__RobustScaler_pca_NM_SMOTE_ENN_knn"

# ============== VALIDACION CRUZADA ================
LABEL_COL = "LABEL"
N_SPLITS = 5
SHUFFLE = True
RANDOM_STATE = 42
N_VALUES = [500, 1000, 2000, 5000, 10000, 20000, 30000, 40000]

# ============== TRANSFORMACION DE FEATURES ================
SCALER_NAME = "RobustScaler"
USE_PCA = True
N_COMPONENTS_PCA = 3

# ============== REBALANCEO ================
ESTRATEGIA_REBALANCEO = "NearMiss_SMOTE_ENN"
ESTRATEGIA_REBALANCEO_CORTA = "NM_SMOTE_ENN"

# ============== MODELO KNN ================
MODEL_NAME = "knn"
MODEL_DISPLAY_NAME = "KNeighborsClassifier"
MODEL_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "minkowski",
    "p": 2,
}

# ============== PARAMETROS INTERNOS DE REBALANCEO ================
NEARMISS_VERSION = 1
SMOTE_K_NEIGHBORS = 5
ENN_N_NEIGHBORS = 3

# ============== ESTRUCTURAS PARA REPORTES ================
experiment_parameters = []
cv_fold_results = []
cv_summary = []


ejecutar_experimento_a2_sampling_size(
    {
        "project_root": PROJECT_ROOT,
        "dataset_original": DATASET_ORIGINAL,
        "model_family": MODEL_FAMILY,
        "experiment_name": EXPERIMENT_NAME,
        "label_col": LABEL_COL,
        "n_splits": N_SPLITS,
        "shuffle": SHUFFLE,
        "random_state": RANDOM_STATE,
        "n_values": N_VALUES,
        "scaler_name": SCALER_NAME,
        "use_pca": USE_PCA,
        "n_components_pca": N_COMPONENTS_PCA,
        "estrategia_rebalanceo": ESTRATEGIA_REBALANCEO,
        "estrategia_rebalanceo_corta": ESTRATEGIA_REBALANCEO_CORTA,
        "model_name": MODEL_NAME,
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_params": MODEL_PARAMS,
        "nearmiss_version": NEARMISS_VERSION,
        "smote_k_neighbors": SMOTE_K_NEIGHBORS,
        "enn_n_neighbors": ENN_N_NEIGHBORS,
        "cv_fold_results": cv_fold_results,
    }
)
