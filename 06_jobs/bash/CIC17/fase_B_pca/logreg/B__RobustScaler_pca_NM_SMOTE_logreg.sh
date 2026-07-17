#!/bin/bash

set -euo pipefail

# ========= RUTAS =========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

PYTHON_SCRIPT="$BASE_DIR/04_experiments/CIC17/fase_B_pca/logreg/B__RobustScaler_pca_NM_SMOTE_logreg.py"
NOMBRE_SCRIPT="$(basename "$PYTHON_SCRIPT" .py)"
LOG_OUT_DIR="$BASE_DIR/06_jobs/logs/CIC17/fase_B_pca/logreg"

JOB_ID="$(date +"%Y%m%d_%H%M%S")"
LOG_OUT_FILE="$LOG_OUT_DIR/output_${NOMBRE_SCRIPT}_${JOB_ID}.log"

mkdir -p "$LOG_OUT_DIR"

# ========= CONDA =========
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate tfgClean
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate tfgClean
else
    echo "No se ha encontrado conda.sh. Se usara el Python activo."
fi

export PYTHONNOUSERSITE=1

# ========= EJECUCION =========
cd "$BASE_DIR"

echo "Ejecutando experimento B RobustScaler PCA NM_SMOTE logreg..."
echo "Script: $PYTHON_SCRIPT"
echo "Log: $LOG_OUT_FILE"

python "$PYTHON_SCRIPT" 2>&1 | tee "$LOG_OUT_FILE"

echo "Proceso finalizado correctamente."
