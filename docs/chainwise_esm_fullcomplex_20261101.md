# Chainwise ESM2, Full-Complex G0 Control

## Approved Scope

Execute directions 1 and 3 only. Do NOT create or restart pure single-chain
model training/inference. Independently encode chains inside the frozen ESM2
extractor, then restore the original complete protein-complex residue axis.
The model still receives the same complete complex, geometry, chain metadata,
ligand input and original labels. Existing BioLiP single-chain samples remain
unchanged; no samples are discarded because they originally contain one chain.

The CPU background audit compares isolated-chain and supplied-complex geometry
only. It does not feed isolated chains into the prediction model.

## Frozen Control

- Baseline: `G0_small_global`, seed44, best-F1 checkpoint at epoch52.
- Baseline VAL: F1=0.602434, AUPRC=0.633018, MCC@0.5=0.601489.
- Saved VAL gate: 0.5325314998626709. Frozen diagnostic does not refit it.
- Original experiment: `experiments/local_spatial_scan_ampfix_u_20261024`.
- Dataset: `experiments/joint_global_si30_gfgf_20261017/dataset`.
- Preserve strict TRAIN/VAL/TEST membership, original 5A contact labels,
  TRAIN-only normalization statistics, 1800-residue cap, 256-atom cap and
  alternating-end atom truncation. This study does not remove existing caps.
- Scratch training uses the original G0 `code_v1` model and trainer, fully
  trainable, seed44, batch16, accumulation1, BF16, LR1e-4, weight decay0.05,
  F1 early stopping with patience30. No resume, freezing, or compile.
- Selection remains the original joint VAL F1 rule for a controlled comparison.
  Additional source/chain-stratified VAL metrics are diagnostics, not a new
  checkpoint-selection objective. TEST/HOLO4K do not guide development.

## Feature-Only Change

Replace only multichain PDBbind ESM2 caches. Independently encode each observed
chain sequence with ESM2-650M L33, FP32, without artificial linkers or additional
normalization. Preserve the original cached sequence/token mapping. Reassemble
by original residue keys, order and count, including interleaved chain rows.
Original single-chain PDBbind and BioLiP caches are linked unchanged as controls.

Deduplicate sequences by SHA256. Short-chain LiBRe caches may be reused only
after metadata validation and fresh-extraction parity checks on up to eight
cache hits. Any parity failure disables reuse. Chains over1022 residues use
1022-residue windows, overlap768, uniform overlap averaging, without truncation.
Long single-chain PDBbind inputs remain unchanged in this narrow experiment;
it is not a claim of harmonized windowing for every historical cache.

Checks include frozen config/split/stats/source hashes, sequence and residue-key
alignment, finite `[N,1280]` tensors, complete reconstruction, and real-data GPU
training/VAL smoke before scratch training. New features live separately from
original caches. Extraction may materialize TEST features, but no TEST forward
pass or selection is performed.

## Jobs And Outputs

Experiment root on HPC2:
`/hpc2hdd/home/shuang886/PLC/experiments/chainwise_esm_fullcomplex_20261101`.

| Stage | Job | Resources | Depends On | Purpose |
|---|---|---|---|---|
| Inventory | 13445710 | i64m512u,16 CPU | None | Validate frozen inputs; deduplicate chains; prepare unchanged G0 config |
| ESM extraction/reassembly | 13445711 | i64m1tga800u,1 GPU,8 CPU | Inventory | Chainwise ESM2 and complete-complex reconstruction |
| Frozen VAL cache swap | 13445712 | i64m1tga800u,1 GPU,8 CPU | Extraction | Two complete-complex forwards differing only in ESM features |
| Scratch G0 | 13445713 | i64m1tga800u,1 GPU,8 CPU | Extraction | 100-step smoke, then unchanged G0 scratch recipe |
| Background audit | 13445714 | i64m512u,32 CPU | Inventory | Structural-background analysis of existing full-complex predictions |

The frozen swap is a distribution-shift diagnostic, not evidence of retrained
performance. Scratch does not depend on the direction of that diagnostic's F1.
Report PDBbind single-chain, PDBbind multichain and BioLiP strata separately:
F1/MCC@0.5, F1/MCC at the joint VAL gate, AUPRC/AUROC and Top20/Top-L metrics.
Training writes `ValStrata/*` TensorBoard scalars and `fullcomplex_val_strata.jsonl`.

CPU audit covers all strict-VAL multichain PDBbind queries plus deterministic
50-query single-chain controls from each source. It reconstructs the same
cached protein heavy atoms, compares SASA with/without other chains, measures
nearest other-chain heavy-atom distance and relates these to existing G0
**full-complex** probabilities at0.5 and the saved VAL gate. The 6A proximity and
10A^2 delta-SASA bins are descriptive, not new model gates. Failures are retained.
No native ligand coordinates become model inputs.

The supplied context is not automatically a verified biological assembly.
Missing symmetry mates/atoms are not reconstructed. SASA association is not
causal proof that removing context caused prediction errors. The initial pooled
audit weights residues/queries; per-query records are retained for balanced
follow-up summaries.

## Verification And Submission Ledger

Local regression: 37 tests passed before deployment, including full-complex
reassembly, feature/config preservation, cache validation, long-chain window
coverage and synthetic two-chain SASA occlusion. Remote GPU validation is a
separate required stage, not implied by CPU tests.

Old chain-union training `13423577_[0-2]` failed and is excluded from this plan.
The inspected task0 failed during DataLoader worker creation with
`OSError: [Errno12] Cannot allocate memory`. No restart was requested or performed.
Other already running LocalScan/sequence-context jobs are left untouched.

Submitted on 2026-11-01 from immutable commit `1d76d97`. Initial queue check:
inventory pending Priority; all four descendants pending Dependency. No new
GPU result is available yet. All descendants use `afterok` and fail closed if
their dependency fails; unrelated jobs are not cancelled.

Deployment archive SHA256:
`6218ccb59c5b0508af4081cba719becee6014f3d3f183f5972b24e2f7a62640b`.
Remote `deployment.json` and `jobs.json` retain deployment and submission
provenance. The remote shell prints a startup warning that breaks rsync/SFTP;
the same archive was transferred through SSH stdin and verified before unpacking.
