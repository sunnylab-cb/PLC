# Regularization & Hyperparameter References for Protein-Ligand ML

Collected during PLC v10m4 over-regularization analysis (2026-04-11).
Used to benchmark PLC's regularization choices against published models.

---

## Protein Binding Site / Structure Prediction Models

### LM-GVP
**Full name**: LM-GVP: An Extensible Sequence and Structure Informed Deep Learning Framework for Protein Property Prediction
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9046255/
**Relevance**: GVP + language model fusion for protein property prediction. Learning rate range 1e-4 ~ 1e-5. Demonstrates that separate projection paths for LM and GVP are important to prevent language model from dominating geometric pathway.

### DeepProSite
**Full name**: DeepProSite: Structure-Aware Protein Binding Site Prediction Using ESMFold and Pretrained Graph Transformer
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC10723037/
**Relevance**: ESMFold + Graph Transformer for binding site prediction. Uses dropout=0.2, classifier_dropout=0.2. Confirms these values are standard for ESM-based binding site models.

### ESMBind
**Full name**: ESMBind and QBind: LoRA, QLoRA, and ESM-2 for Predicting Binding Sites and Residue Interactions
**URL**: https://www.biorxiv.org/content/10.1101/2023.11.13.566930v1.full
**HuggingFace blog**: https://huggingface.co/blog/AmelieSchreiber/esmbind
**Relevance**: LoRA fine-tuning of ESM-2 for binding site prediction. lora_dropout=0.2, weight_decay=0.2, lr=5.7e-4. Confirms dropout=0.2 is normal for ESM fine-tuning contexts.

### ScanNet
**Full name**: ScanNet: An Interpretable Geometric Deep Learning Model for Structure-Based Protein Binding Site Prediction
**URL**: https://www.nature.com/articles/s41592-022-01490-7
**Journal**: Nature Methods, 2022
**Relevance**: Geometric deep learning for protein binding site prediction. Does NOT use sample-level modality dropout. Confirms that multi-modality simultaneous dropout is not standard practice in protein binding site ML.

---

## Protein Structure / Design Models

### ProteinMPNN
**Full name**: Robust Deep Learning-Based Protein Sequence Design Using ProteinMPNN
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9997061/
**Journal**: Science, 2022
**Training details**: https://github.com/dauparas/ProteinMPNN/blob/main/training/README.md
**Relevance**: Coordinate noise ablation. Trained variants: v_48_002 (0.02 Å), v_48_010 (0.10 Å), v_48_020 (0.20 Å), v_48_030 (0.30 Å). **Maximum coordinate noise used = 0.30 Å**. dropout=0.1. Key reference: PLC's coord_noise_std=0.5 Å exceeds ProteinMPNN's maximum tested value.

### LigandMPNN
**Full name**: Atomic Context-Conditioned Protein Sequence Design Using LigandMPNN
**URL**: https://www.nature.com/articles/s41592-025-02626-1
**Journal**: Nature Methods, 2025
**Relevance**: Extension of ProteinMPNN for protein-ligand contexts. Discusses backbone noise in the context of protein-ligand design. Confirms 0.2–0.3 Å as the practical upper bound for coordinate noise in structure-based models.

### ThermoMPNN
**Full name**: Transfer Learning to Leverage Larger Datasets for Improved Prediction of Protein Stability Changes
**URL**: https://www.pnas.org/doi/10.1073/pnas.2314853121
**Journal**: PNAS, 2024
**Relevance**: Fine-tuning ProteinMPNN for stability prediction. AdamW with weight_decay=0.01. Shows that moderate weight decay (0.01) is standard for MPNN-style protein models.

### ProstaNet
**Full name**: ProstaNet: A GVP-GNN Framework for Protein Stability Prediction
**URL**: https://spj.science.org/doi/10.34133/research.0674
**Journal**: Research (SPJ), 2024
**Relevance**: GVP-GNN for protein stability. Uses dropout=0.6 during pretraining, 0.4 during fine-tuning — but in a large pretraining regime, not comparable to PLC's from-scratch training.

---

## Loss Functions

### Tversky Loss
**Full name**: Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks
**URL**: https://arxiv.org/abs/1706.05721
**Relevance**: Original Tversky loss paper. **Recommends alpha=0.3, beta=0.7 as the standard configuration** for emphasizing recall (reducing false negatives). PLC's Tversky settings are exactly aligned with this recommendation. Note: the recall bias is a feature, not a bug — the question is whether this is the right objective for precise pocket localization.

### Unified Focal Loss
**Full name**: Unified Focal Loss: Generalising Dice and Cross Entropy-Based Losses to Handle Class Imbalanced Medical Image Segmentation
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC8785124/
**Relevance**: Supports combining Focal + Tversky as a joint objective. Shows this combination is valid in medical segmentation. However, notes that the combination can amplify recall bias when both components are recall-oriented.

---

## Regularization Techniques

### Stochastic Depth (Drop Path)
**Full name**: Deep Networks with Stochastic Depth
**Reference implementation**: ConvNeXt (A ConvNet for the 2020s)
**URL**: https://arxiv.org/pdf/2201.03545
**Relevance**: ConvNeXt-T uses stochastic_depth=0.1, ConvNeXt-S uses 0.2. Original paper recommends linearly increasing from 0 to max over layers. PLC's 0.15 is conservative and safe.

### Modality Dropout (Multi-modal)
**Full name**: Modality Dropout for Improved Performance-Driven Talking Faces
**URL**: https://dl.acm.org/doi/10.1145/3382507.3418840
**Conference**: ACM MM, 2020
**Relevance**: One of the earliest uses of modality dropout (randomly zeroing an entire input modality). Designed to improve robustness to missing modalities at inference time. Key context: only useful when the model might receive missing modalities at inference — PLC always has all modalities at inference, reducing the justification for this technique.

### Modality Bias from Dropout (CVPR 2024)
**Full name**: A Study of Dropout-Induced Modality Bias on Robustness to Missing Video Frames
**URL**: https://openaccess.thecvf.com/content/CVPR2024/papers/Dai_A_Study_of_Dropout-Induced_Modality_Bias_on_Robustness_to_Missing_CVPR_2024_paper.pdf
**Conference**: CVPR, 2024
**Relevance**: Shows that modality dropout introduces modality bias — the model learns to over-rely on the modality that is more often present during training. When multiple modalities are simultaneously dropped at different rates, the model learns an asymmetric reliance on the more "reliable" modality, which can hurt performance when all modalities are present at inference.

---

## Key Takeaways for PLC v10m4

| Finding | Evidence |
|---------|----------|
| coord_noise_std=0.5 Å is above literature maximum | ProteinMPNN max = 0.30 Å (LigandMPNN confirms) |
| Tversky α=0.3, β=0.7 is correct per literature | Tversky 2017 recommends exactly these values |
| Sample-level GVP dropout has no protein ML precedent | ScanNet, DeepProSite, LM-GVP do not use it |
| Simultaneous multi-modality dropout causes modality bias | CVPR 2024 — especially harmful when inference always has all modalities |
| Individual dropout values (0.1–0.2) are normal | ProteinMPNN, DeepProSite, ESMBind all confirm |
| Weight decay 0.05 is acceptable | ESMBind uses 0.2; ThermoMPNN uses 0.01 |
| Dataset size (15k training) is NOT small for protein ML | scPDB ≈ 17k, PDBbind refined ≈ 4k; 15k is above average |

---

## Dataset Size Context

For reference, major protein-ligand datasets:

| Dataset | Size | Notes |
|---------|------|-------|
| scPDB | ~17,000 complexes | Common binding site training set |
| PDBbind refined | ~4,000–5,000 | High-quality curated |
| PDBbind general | ~19,000+ | Broader, variable quality |
| COACH420 | 420 | Standard benchmark test set |
| HOLO4K | ~4,000 | Larger benchmark test set |
| **PLC training set** | **~15,230** | Above average; "small dataset" argument does not strongly apply |

At 15k+ training samples, the over-regularization concern is more about **gradient signal quality** than sample count — simultaneous modality dropouts reduce the effective signal per batch even with large datasets.
