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

## Submission

Immutable code commit: `04703f4`, pushed to `plc/master`.
Archive SHA256: `70f7a30fb59e4d3c82780bac9d7b607ea3b38132bf14b4524fbbc165def87e8e`.
Remote af3 CPU import/splitting/fixed-threshold-MCC smoke passed; that
environment has no pytest, while the71-test local BIO suite passed.

| Phase | Job | Partition | Initial State |
| --- | --- | --- | --- |
| VAL manifest and integrity audit | 13281235 | i64m512ue | Pending/Priority |
| Paired full/chain inference | 13281236 | i64m1tga800ue, one GPU | Pending/afterok:13281235 |
| Source/chain-stratified summary | 13281237 | i64m512ue | Pending/afterok:13281236 |

## Recovered Results (2026-10-30)

All three stages completed. The summary was recovered from
`results/summary.json`; SHA256
`8bdac6dde5df7abdba14232efedde9d2f9e8e6443ea52bc4bf4225b3c8bde962`.
Protocol SHA256:
`0478c2056eb745c072d508e51cf988e762241d598e30863345b8c81c91e66476`.
The GPU pass processed17,395 original pairs and19,165 chains in1,636.90s;
peak allocated/reserved memory1.260/2.525GiB. This is the OLD full-width joint
epoch25 checkpoint, not the newer width256 G0 epoch52 checkpoint.

| Strict VAL stratum | Pairs | Full Top20 F1 | Chain Top20 F1 | Full MCC | Chain MCC | Full gate F1 | Chain gate F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 17395 | .550023 | .538660 | .536137 | .524268 | .566515 | .556240 |
| PDBbind | 1697 | .602390 | .486098 | .587711 | .466215 | .617072 | .511768 |
| BioLiP | 15698 | .544362 | .544342 | .530562 | .530544 | .561049 | .561048 |
| Multichain | 950 | .555496 | .347695 | .548136 | .331018 | .572993 | .384767 |
| Multiple contact chains | 356 | .484277 | .323728 | .473041 | .304234 | .503939 | .358217 |

F1/MCC in this table are sample means; MCC columns use the Top20 rule, NOT a
fixed probability threshold. Full versus chain pooled AUPRC is .625319/.572673
overall, .667141/.394072 for PDBbind and .619736/.619735 for BioLiP.

Conclusion: independently inferring chains is not a universal improvement on
the original per-copy task. The already-chain BioLiP numerical control is
unchanged; multichain PDBbind loses substantial signal. Do not impose universal
full/chain consistency on interface-containing cases. The subsequently approved
chain/exact-chemical union protocol is a NEW task/label definition, not evidence
that the old model becomes universally better when chain context is removed.
