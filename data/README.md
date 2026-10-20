# Data

Datasets, reference snapshots, and labels are **not committed to Git** by
default. This directory documents the expected local layout.

## Demo dataset

Training materializes the scikit-learn breast cancer dataset at
`data/breast_cancer.csv` when no dataset is configured. It contains 569 rows,
30 numeric features, and a binary `target` column:

| Value | Class |
|---|---|
| 0 | malignant |
| 1 | benign |

This dataset is bundled with scikit-learn and is used only to demonstrate the
workflow. It is not a clinical system and must not be used for medical
decisions.

## Generated files

| Path | Purpose |
|---|---|
| `data/breast_cancer.csv` | Demo dataset created by `scripts/train.py` |
| `data/reference.csv` | Training split written by the training pipeline |
| `data/labels.jsonl` | Optional ground-truth labels for monitoring |

`data/reference.csv` is the baseline for drift detection. Regenerate it whenever
a new model version is promoted.

## Using a custom dataset

A custom dataset must be a CSV with a target column. Classification targets may
be binary or multiclass. Numeric and categorical features are supported.

1. Set `DATASET_PATH` and `TARGET_COLUMN` in `.env`.
2. Optionally adjust `TEST_SIZE` and `RANDOM_STATE`.
3. Run `python scripts/train.py`.
4. Review `reports/evaluation.json` and the configured model gates.

The training workflow rejects candidates that fail the configured F1, recall,
ROC AUC, or baseline comparison gates.

## Ground-truth labels

Performance monitoring requires labels joined to predictions by
`prediction_id`. Labels can be supplied as JSONL or CSV:

```jsonl
{"prediction_id": "3f2c...", "label": 1}
```

```csv
prediction_id,label
3f2c...,1
```

Delayed labels are expected in production. Append them when they become
available and rerun `python scripts/run_monitoring.py`.

## Privacy

Inference payloads are written to `artifacts/predictions.jsonl` to support
monitoring. Do not send sensitive personal or regulated data unless you have
reviewed retention, access control, and redaction requirements. Set
`PREDICTION_LOG_ENABLED=false` when payload logging is not appropriate.
