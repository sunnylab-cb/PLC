# Random Forest Binding Site Prediction PoC Results
**Date**: 2025-12-27

## Experiment Overview
We aimed to build a Random Forest model to predict protein-ligand binding sites using residue-level and ligand-level features.
Two datasets were compared:
1.  **Baseline**: 46 features (Residue Identity, Charge, Hydrophobicity, Protein AAC, Ligand Props).
2.  **Enhanced**: 466 features (Baseline + Dipeptide Composition + Amphiphilic Pseudo-AA Composition).

## Results Summary

| Model Config | Data Size | ROC AUC | Accuracy | Recall (Binding) | Precision (Binding) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (46 dims)** | 5 Batches (~200k samples) | **0.8013** | 0.8330 | 0.54 | 0.20 |
| **Enhanced (466 dims)** | 5 Batches (~227k samples) | 0.7248 | **0.8687** | 0.36 | 0.15 |
| **Enhanced (466 dims)** | **Full Data** (~569k samples) | **0.7961** | 0.8387 | **0.55** | 0.19 |

## distinct Findings
1.  **Baseline Effectiveness**: The simple 46-dim feature set is robust and performs well (AUC ~0.80) even with limited data.
2.  **Curse of Dimensionality**: When trained on limited data (5 batches), the Enhanced model significantly underperformed (AUC 0.72 vs 0.80), likely due to overfitting the large feature space (466 dims).
3.  **Data Scale Matters**: scaling up to the full dataset (13 batches) allowed the Enhanced model to recover performance (AUC ~0.80), matching the baseline.
4.  **Performance Ceiling**: More features did not strictly *outperform* the baseline yet, suggesting either the additional features (Dipeptide/Pseudo-AA) are not highly discriminative for this specific task, or the model needs even more data/complexity (e.g., Deep Learning) to leverage them.

## Next Steps
- Consider feature selection on the Enhanced set to remove noise.
- Explore Gradient Boosting (XGBoost/LightGBM) which might handle high-dimensional features better than Random Forest.
