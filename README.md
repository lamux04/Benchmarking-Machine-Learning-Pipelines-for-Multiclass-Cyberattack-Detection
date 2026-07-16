# Cyberattack Detection with Classical Machine Learning

A reproducible benchmark of classical machine learning pipelines for **multiclass network intrusion detection** on the CIC-IDS-2017 dataset.

The project evaluates how feature scaling, class rebalancing and PCA affect predictive performance and computational cost under extreme class imbalance. The strongest configuration combined **KNN + StandardScaler**, without PCA or class rebalancing, and achieved a **0.876 Macro F1** and **0.988 MCC** on an untouched hold-out set.

---

## Key Results

| Metric | Result |
|---|---:|
| Macro F1 | **0.8762** |
| MCC | **0.9879** |
| Accuracy | **99.64%** |
| Processed network flows | **2,520,798** |
| Predictive features | **47** |
| Traffic classes | **15** |
| Pipeline variants benchmarked | **60** |

### Selected pipeline

```text
47 cleaned network-flow features
                ↓
        StandardScaler
                ↓
             KNN
   k = 5 · Manhattan distance
                ↓
      15-class prediction
```

| Component | Selected configuration |
|---|---|
| Scaler | StandardScaler |
| Dimensionality reduction | None |
| Class rebalancing | None |
| Classifier | KNN |
| Number of neighbours | 5 |
| Distance metric | Manhattan |

---

## Problem

CIC-IDS-2017 is highly imbalanced. Benign traffic contains more than two million flows, while some attack classes contain fewer than 40 examples.

Under this distribution, accuracy can be misleading: a model may classify most flows correctly while still failing to detect rare attacks. The project therefore focuses on **class-balanced performance**, leakage-safe validation and security-relevant errors such as malicious traffic classified as benign.

The benchmark addresses three questions:

1. Does class rebalancing improve detection on the original traffic distribution?
2. Can PCA reduce computational cost without losing important predictive information?
3. How do different model families respond to the same preprocessing decisions?

---

## Dataset

This project uses the machine-learning-ready CSV version of **CIC-IDS-2017**.

### Initial dataset

- 2,830,743 network flows
- 79 columns
- 15 classes
- 1 benign class
- 14 attack classes

### Dataset after cleaning

- 2,520,798 network flows
- 47 predictive features
- 1 target column

The original dataset is **not included** in this repository because of its size and distribution terms.

Place the downloaded CSV files inside:

```text
01_datasets/
└── CIC17/
    └── raw/
```

The data-preparation pipeline handles:

- duplicate removal;
- missing and infinite values;
- constant features;
- highly correlated features;
- target encoding;
- stratified train/hold-out splitting.

---

## Experimental Design

The cleaned dataset was split into:

- **80% development set** for model selection and cross-validation;
- **20% untouched hold-out set** for final evaluation.

Candidate pipelines combined the following dimensions:

| Dimension | Alternatives |
|---|---|
| Scaling | StandardScaler, RobustScaler |
| Dimensionality | Original features, PCA |
| Rebalancing | None, RUS + SMOTE, RUS + SMOTE + ENN, NearMiss + SMOTE, NearMiss + SMOTE + ENN |
| Model | KNN, Logistic Regression, MLP |

This produced **60 main pipeline variants**, followed by model-specific hyperparameter search.

### Leakage prevention

During each cross-validation fold:

1. the scaler was fitted only on the training partition;
2. PCA was fitted only on the scaled training partition;
3. rebalancing was applied only to training data;
4. validation data preserved its original class distribution;
5. the hold-out set remained untouched until final evaluation.

---

## Evaluation Metrics

**Macro F1** was the primary model-selection metric because it assigns equal importance to every class.

Additional metrics included:

- Matthews Correlation Coefficient;
- accuracy;
- macro precision and recall;
- weighted precision, recall and F1;
- fit time;
- validation time;
- mean detection latency.

---

## Model Comparison

The following table shows the performance-oriented configuration selected through cross-validation for each model.

| Model | PCA | Accuracy | Macro F1 | MCC |
|---|---|---:|---:|---:|
| **KNN** | No | **99.64%** | **87.62%** | **0.9879** |
| MLP | No | 98.80% | 71.60% | 0.9599 |
| Logistic Regression | No | 97.79% | 60.08% | 0.9295 |

Although all three models achieved high accuracy, their Macro F1 scores differed substantially. This confirms that accuracy alone was not sufficient for evaluating performance under severe class imbalance.

---

## Impact of Class Rebalancing

The following comparison isolates the effect of rebalancing for KNN with StandardScaler and without PCA.

| Rebalancing strategy | Accuracy | Macro F1 | MCC |
|---|---:|---:|---:|
| NearMiss + SMOTE | 15.10% | 22.18% | 0.1633 |
| NearMiss + SMOTE + ENN | 15.08% | 22.19% | 0.1644 |
| **No rebalancing** | **99.64%** | **87.62%** | **0.9879** |
| RUS + SMOTE | 96.45% | 63.33% | 0.8972 |
| RUS + SMOTE + ENN | 96.68% | 64.04% | 0.9020 |

The original training distribution produced the strongest result. RUS-based strategies were more competitive than NearMiss-based alternatives, while ENN produced only marginal changes.

---

## Impact of PCA

| Model | Macro F1 without PCA | Macro F1 with PCA | Difference | Fit time without PCA | Fit time with PCA | Time reduction |
|---|---:|---:|---:|---:|---:|---:|
| KNN | **87.62%** | 87.31% | −0.31 pp | 0.185 s | 0.180 s | 3% |
| Logistic Regression | **60.08%** | 54.53% | −5.54 pp | 345.38 s | 259.98 s | 25% |
| MLP | 71.60% | **71.80%** | +0.20 pp | 244.61 s | 89.34 s | **63%** |

PCA was not universally beneficial:

- it provided no useful trade-off for KNN;
- it reduced Logistic Regression performance;
- it reduced MLP fit time substantially while preserving similar predictive performance.

KNN's low fit time should not be interpreted as low end-to-end cost. Because it is instance-based, much of its computational workload occurs during inference.

---

## Error Analysis

The strongest KNN pipeline still struggled with several underrepresented classes.

Security-relevant false negatives included:

- **Bot:** 121 of 390 flows classified as benign;
- **Infiltration:** 4 of 7 flows classified as benign;
- **SQL Injection:** only 2 of 4 flows classified correctly.

These results highlight an important limitation: strong aggregate metrics do not guarantee reliable detection for rare attacks.

<!-- Add the English-labelled normalized confusion matrix here. -->

```markdown
![Normalized confusion matrix](05_results/CIC17/figures/confusion-matrix-knn.png)
```

---

## Repository Structure

The repository is organized by project stage. Every top-level directory contains a dedicated `CIC17` folder so that additional datasets can be incorporated later without mixing their code, experiments or results.

```text
.
├── 01_datasets/
│   └── CIC17/                  # Raw, cleaned and partitioned CIC-IDS-2017 data
├── 02_src/
│   └── CIC17/                  # Shared Python modules and reusable utilities
├── 03_preprocessing/
│   └── CIC17/                  # Data cleaning, feature preparation and dataset splitting
├── 04_experiments/
│   └── CIC17/                  # PCA, rebalancing, model selection and hold-out experiments
├── 05_results/
│   └── CIC17/                  # Metrics, tables, figures and confusion matrices
├── 06_jobs/
│   └── CIC17/                  # Local or cluster job scripts used to launch experiments
├── environment.yml             # Conda environment, when available
├── requirements.txt            # Python dependencies, when available
├── LICENSE
└── README.md
```

---

## Installation

### Option 1: Conda

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>

conda env create -f environment.yml
conda activate cyberattack-detection
```

### Option 2: pip

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Reproducing the Experiments

The experiments must be executed **sequentially** because the results of the preliminary phases determine the configuration used in the following stages.

All commands below are intended to be run from the repository root.

### 1. Clean and split the dataset

First, clean the original CIC-IDS-2017 files and create the stratified development and hold-out partitions:

```bash
python 03_preprocessing/CIC17/01_clean_dataset.py
python 03_preprocessing/CIC17/02_split_dataset.py
```

The cleaning script prepares the final feature set, while the split script reserves the untouched hold-out partition used for final evaluation.

### 2. Phase A1 — Select the number of PCA components

Run the PCA component-selection experiment independently for each scaler:

```bash
python 04_experiments/CIC17/fase_A1_pca_components/A1_RobustScaler.py
python 04_experiments/CIC17/fase_A1_pca_components/A1_StandardScaler.py
```

Review the generated results and select the minimum number of components required to retain at least 99% of cumulative explained variance for each scaler.

In this project, the selected values were:

- **RobustScaler:** 3 components
- **StandardScaler:** 33 components

### 3. Phase A2 — Select the rebalancing size

Run all KNN experiments under:

```text
04_experiments/CIC17/fase_A2_sampling_size/knn/
```

The phase evaluates combinations of:

- StandardScaler and RobustScaler;
- NearMiss + SMOTE;
- NearMiss + SMOTE + ENN;
- RUS + SMOTE;
- RUS + SMOTE + ENN.

Each script compares the predefined sampling sizes and records predictive performance and computational cost.

To run every Python experiment in the folder:

```bash
find 04_experiments/CIC17/fase_A2_sampling_size/knn \
  -maxdepth 1 \
  -type f \
  -name "*.py" \
  -print0 | sort -z | xargs -0 -n1 python
```

Review the results before continuing. The selected target in this project was **10,000 samples per class**.

### 4. Phase B — Experiments with PCA

Run the model experiments stored under:

```text
04_experiments/CIC17/fase_B_pca/
├── knn/
├── logreg/
└── mlp/
```

Each model folder contains the combinations of scaler and rebalancing strategy evaluated with PCA.

```bash
for model in knn logreg mlp; do
  find "04_experiments/CIC17/fase_B_pca/$model" \
    -maxdepth 1 \
    -type f \
    -name "*.py" \
    -print0 | sort -z | xargs -0 -n1 python
done
```

After all experiments finish, build the phase comparison table:

```bash
python 04_experiments/CIC17/fase_B_pca/B_build_comparison_table.py
```

### 5. Phase C — Experiments without PCA

Run the equivalent experiments without dimensionality reduction:

```text
04_experiments/CIC17/fase_C_no_pca/
├── knn/
├── logreg/
└── mlp/
```

```bash
for model in knn logreg mlp; do
  find "04_experiments/CIC17/fase_C_no_pca/$model" \
    -maxdepth 1 \
    -type f \
    -name "*.py" \
    -print0 | sort -z | xargs -0 -n1 python
done
```

After all experiments finish, build the comparison table:

```bash
python 04_experiments/CIC17/fase_C_no_pca/C_build_comparison_table.py
```

### 6. Review the generated results

Outputs from every phase should be stored under:

```text
05_results/CIC17/
```

Review the cross-validation results from phases B and C to identify the strongest configuration for each model. The selected pipelines can then be retrained on the complete development set and evaluated on the untouched hold-out partition.

Each experiment should preserve:

- the pipeline configuration;
- hyperparameters;
- random seed, where applicable;
- cross-validation metrics;
- fit and scoring times;
- hold-out metrics;
- per-class classification reports;
- confusion matrices.

> The workflow is intentionally sequential. Do not run phases B or C before completing A1 and A2, because their selected parameters are required by the later experiments.

---

## Reproducibility

To make the benchmark reproducible:

- use fixed random seeds where supported;
- store experiment parameters in configuration files;
- record Python and dependency versions;
- keep the hold-out set separate from model selection;
- save raw result files before generating summary tables;
- document the hardware used for timing measurements.

Computational times are environment-dependent and should not be interpreted as universal benchmarks.

---

## Main Findings

- StandardScaler appeared in the strongest configuration for all three evaluated models.
- Rebalancing did not improve performance on the original traffic distribution.
- NearMiss-based undersampling caused the largest performance losses.
- PCA was model-dependent rather than universally beneficial.
- KNN achieved the strongest class-balanced performance.
- Rare attack categories remained the main limitation.

---

## Limitations

- Results were obtained on a hold-out partition of CIC-IDS-2017, not live production traffic.
- Some attack classes contained too few examples for robust per-class conclusions.
- The benchmark did not evaluate temporal drift or generalization across datasets.
- KNN inference cost may become significant at higher traffic volumes.
- The project evaluates a classification pipeline, not a complete production NIDS.

---

## Future Work

- validate the methodology on CSE-CIC-IDS-2018 and BCCC-CIC-IDS-2017;
- investigate cost-sensitive learning and class-weighted approaches;
- evaluate Random Forest and gradient-boosting models;
- compare deep learning architectures;
- test streaming inference, throughput and memory usage;
- evaluate performance on temporally separated traffic;
- add experiment tracking and automated reproducibility checks.

---

## Technologies

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn` · `Matplotlib` · `Git` · `GNU/Linux`

---

## Documentation

- [Professional case study](<YOUR_PORTFOLIO_CASE_STUDY_URL>)
- [Complete technical report](<YOUR_TECHNICAL_REPORT_URL>)
- [Experiment results](./05_results/CIC17/)
- [Experiment code](./04_experiments/CIC17/)
- [Preprocessing pipeline](./03_preprocessing/CIC17/)

---
## License

Copyright © 2026 Javier Labrador Muñoz.

The source code in this repository is licensed under the [PolyForm Noncommercial License 1.0.0](./LICENSE).

You may use, study, modify and redistribute the code for permitted noncommercial purposes, provided that the license terms and required copyright notice are preserved.

Commercial use is not permitted without prior written permission from the author.

The CIC-IDS-2017 dataset is not included in this repository and remains subject to the terms established by its original provider.
