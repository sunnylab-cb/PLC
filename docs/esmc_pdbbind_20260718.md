# ESM-C PDBbind integration (2026-07-18)

## 9977123 selector control

Both seed-44 ESM3-global-prior runs completed 42 epochs.  The experiment changes
only whether `local_delta_logit_scale` is fixed or trainable.

| variant | selector | epoch | val AUPRC | val AUROC | global F1 @ 0.5 | global F1 @ val gate | effective delta scale |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed | best F1 | 32 | 0.62176 | 0.95710 | 0.60213 | 0.60294 | 0.46212 |
| fixed | best AUPRC | 37 | **0.62959** | 0.96295 | **0.60640** | **0.60648** | 0.46212 |
| fixed | best AUROC | 16 | 0.59600 | 0.96303 | 0.57568 | 0.57940 | 0.46212 |
| trainable | best F1 | 27 | 0.61553 | 0.95886 | 0.59927 | 0.59944 | 0.37728 |
| trainable | best AUPRC | 39 | 0.62490 | 0.96170 | 0.60585 | 0.60639 | 0.36973 |
| trainable | best AUROC | 32 | 0.62165 | **0.96353** | 0.59276 | 0.59770 | 0.37164 |

Interpretation:

- The trainable scale consistently shrinks the local-delta multiplier from
  `tanh(0.5)=0.4621` to about `0.37`.
- It does not improve calibrated validation F1 and loses about 0.0047 AUPRC at
  the best-AUPRC checkpoint.  Fixed scale remains the default pending test
  real/shuffle/zero evaluation.
- AUROC-only selection favors an earlier/differently calibrated model and gives
  materially worse F1.  Keep AUROC checkpoints for reporting, not as the sole
  deployment selector.
- The observed validation-loss pattern is reproducible when non-finite padded
  logits enter loss arithmetic as `NaN * 0`; contact ranking metrics use valid
  residues and therefore remain finite.  The loss now uses
  `masked_fill`/valid-only tensors so invalid padding cannot poison early
  stopping.  A short rerun will verify this was the only NaN source.

Machine-readable checkpoint metadata is in
`tmp/selector_9977123_summary.json` (not intended for version control).

## PDBbind ESM-C feature extraction

Model and representation:

- `biohub/esmc-600m-2024-12`
- revision `e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c`
- transformer blocks 12, 24, and 36
- each selected residual stream passes the pretrained final LayerNorm
- inference BF16; original extraction payload `[L,3456]` FP32
- PLC package payload `[L,3,1152]` FP16

Alignment and QC:

- PDB count: 19,037
- exact unique complex sequences: 9,284
- unique residues: 5,708,124
- duplicate PDB occurrences removed by exact sequence SHA-256: 9,753
- ESM2 cache files validated: 19,037/19,037
- ESM-C extraction/package failures: 0
- maximum complex sequence length: 5,400
- longest-sequence smoke peak VRAM: 1.99 GiB
- PDBbind full-complex hashes already present in the PPC chain cache: 6
- representative FP32-to-FP16 packaging relative RMSE: 0.000207
- representative FP32-to-FP16 cosine similarity: 1.000000

The split-generation FASTA is not residue-identical to the cache consumed by
training for 13,585/19,037 PDB IDs (often terminal residues differ).  Therefore
the ESM-C manifest deliberately uses `data/esm2_latents/*/*/*_protein.pt`'s
`sequence` as ground truth.  This preserves the exact residue order and length
used by the current model instead of silently trusting an older split artifact.

Star paths:

```text
/data/husrcf/ProtBind/PLC/features/
  esmc_600m_2024_12_mlc_l12_l24_l36/                 # original FP32 extraction
  esmc_600m_2024_12_mlc_l12_l24_l36_layered_fp16/   # PLC training cache
```

The training cache uses one feature file per exact unique sequence plus
`pdb_manifest.tsv`.  `PocketDatasetV8` resolves PDB IDs through this manifest,
avoiding duplicated feature storage.  For ESM-C + ESM3 runs,
`esm_alignment_features_dir=esm2_latents` supplies only PDB-specific residue
keys for ESM3 alignment; ESM2 embeddings are not fed to the model.

## Initial experiment matrix

`scripts/slurm/run_v11m13_esmc_probes.slurm` keeps the V11m13
group-rescompat/co-attention/local54 architecture and changes only the protein
language-model representation:

1. scratch block-36-initialized learned weighted sum;
2. scratch uniform learned weighted sum;
3. scratch block-36-anchored concat residual;
4. scratch block-36-initialized weighted sum plus weak ESM3 global prior;
5. shape-compatible V11m13 warm-start followed by full fine-tuning;
6. shape-compatible V11m13 warm-start with the established base frozen, training
   only the new ESM-C layer fusion and 1152D input projection.

All probes use seed 44.  The scratch and warm-start arms prevent initialization
regime from being confounded with the protein representation.  Promotion
requires a gain in test real F1 while preserving the real-shuffle
ligand-dependency gap.
