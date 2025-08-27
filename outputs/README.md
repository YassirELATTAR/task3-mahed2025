# Outputs

This folder contains model predictions on test datasets.

## Contents

Prediction files are generated in CSV format with naming convention:
```
predictions_{model_type}.csv
```

Examples:
- `predictions_bilinear.csv`: Bilinear fusion predictions
- `predictions_concatenate.csv`: Concatenation fusion predictions  
- `predictions_cross_attention.csv`: Cross-attention fusion predictions
- `predictions_ensemble.csv`: Ensemble model predictions
- `predictions_transformer.csv`: Transformer fusion predictions

## Format

Each CSV file contains:
- Sample IDs
- Predicted labels (0: non-hate, 1: hate)
- Confidence scores (when available)

These files can be used for further analysis, error analysis, or submission to evaluation platforms.