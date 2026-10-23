# Strict VAL Full-versus-Chain Diagnostic

## Contract

Continue the chain-context diagnostic on the independent joint global-SI30
validation split, without new TEST/COACH/HOLO inference. The existing GFGF
seed44 VAL-best epoch25 checkpoint is frozen; no optimization, gate search,
checkpoint comparison or automatic inference-policy promotion occurs.

Validation membership:17,395 pairs, comprising1,697 PDBbind and15,698 BioLiP.
The cross-set global-SI audit maximum is0.2998324958123953 and passing edge
count is0. No assertion of external benchmark isolation follows from this.
This VAL also selected the checkpoint, so it is development evidence, not a
new independent final test or an unbiased performance certificate.

Both inference modes retain the production VAL configuration:

- Protein residue cap1800 and ligand atom cap256, alternating-ends crop.
  Unlike the previous uncapped-ligand HOLO run, this diagnostic retains the
  original internal VAL cap in BOTH modes to change only chain context.
- TRAIN-only normalization already materialized in the dataset; do not
  normalize a second time. Keep the original spatial clamp and labels <5 A.
- BF16, microbatch2, model.eval(), no NMA augmentation or torch.compile.
- Full mode: existing production graph/attention context.
- Chain mode: infer every observed chain, including zero-contact chains,
  and stitch onto the exact same original capped residue axis.
- Cached ESM/V4 descriptors are sliced, not re-extracted on isolated chains.
- Primary Top20 remains global per original protein-ligand pair, never20 per
  chain. Retain additional fixed budgets and tie-aware expectations.
- F1/MCC/P/R at0.5 and fixed VAL gate0.5535550117492676, separately
  sample-macro, residue-pooled and PDB-balanced; also pooled AUPRC/AUROC.

## Stratification

Report all/PDBbind/BioLiP, single/multichain, and source-by-chain-count groups.
BioLiP is already chain-sliced and must not dilute conclusions about the
multichain PDBbind subset. Also report contacts on one versus multiple chains
(label-based diagnostic strata, never criteria for selecting inference chains).
Multiple contact chains is not necessarily a biologically validated interface.
Report zero-contact-chain false positives under both thresholds and independent
single-chain numerical controls. There is no assumption that every subgroup
will favor chain-context inference.

## Artifacts And Execution

- Runner: `scripts/diagnostics/eval_joint_val_chain_context.py`.
- Slurm: `scripts/slurm/run_joint_val_chain_context.slurm`.
- Remote root: `/hpc2hdd/home/shuang886/PLC/experiments/joint_val_chain_context_20261023`.
- CPU `i64m512ue` preparation freezes VAL-only membership and axes from the
  existing versioned labels, verifies SI/training artifacts/checkpoint hashes.
- One GPU `i64m1tga800ue` performs the paired inference; no training.
- Dependent CPU `i64m512ue` summarizes saved NPZ predictions and output hashes.
- Raw cache stores both probability arrays on the same keys/labels/coordinates.
- Source snapshot and manifest hashes prevent accidental changes after launch.

Local verification:71 tests pass, including VAL-only membership, threshold MCC
versus Top20 MCC, RSA field slicing, no-contact chain retention, exact-axis
checks, and existing benchmark/reference regressions. Slurm syntax check passed.

No jobs or validation scores at protocol creation; submission details will
be appended after successful remote submission.
