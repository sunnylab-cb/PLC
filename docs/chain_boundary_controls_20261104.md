# G0 chain-boundary controls, 2026-11-04

Execution-policy update, 2026-11-06: the three already-running controls may
finish, but no new/restarted HPC jobs are permitted. All future compute moves
to Star. See [acceptance and cutover](hpc_acceptance_star_cutover_20261106.md).

## Purpose and locked protocol

The user approved repairing two chain-boundary defects and retraining controls
on HPC2. Preserve existing Star2843 and HPC13468092 as ESM-only references.
No changes to their code, data, jobs or selection rule.

All new runs retain the G0 width256 model, chainwise ESM2, full-complex inputs,
strict joint split, 5 A contact labels, TRAIN-only normalization, batch16, BF16,
LR1e-4, seed44, complete scratch training and F1_c0 early stopping. No resume,
backbone freezing, new architecture capacity, TEST or HOLO-driven selection.

| Array job | Variant | GVP boundary fix | Co-attention chain positions |
| --- | --- | --- | --- |
| 13573439_0 | gvp_boundary | on | off |
| 13573439_1 | coattn_position | off | on |
| 13573439_2 | both | on | on |
| Existing13468092 | ESM-only reference | off | off |

Each new array element requests one GPU,8 CPUs,100G RAM,7 days on
`i64m1tga800ue`. It first runs a100-step real TRAIN smoke, a cap1800 memory stress
step, and a small VAL check. Smoke weights are discarded before fresh training.
Initial queue check: all new tasks pending Priority; original13468092 running.

Remote root:
`/hpc2hdd/home/shuang886/PLC/experiments/chain_boundary_controls_20261104`.
`model_code` is an independent copy of the frozen G0 source with three model
files overlaid. `code_v1` contains the orchestration/audit scripts. Training
asserts the model imports from `model_code`, not the active repository. The
current V11 file also has an older, inactive optional sequence-encoder branch;
the configuration explicitly retains the original hybrid encoder.

Preparation verifies the upstream data/split/stats/config artifacts and freezes
source hashes and all three YAMLs. Subsequent starts verify source/config and
small metadata hashes. Large PT assets are checked during preparation, not
rehash-scanned at every epoch. ESM extraction weights are not required.

## Exact repairs

`gvp_chain_boundary_fix` defaults false for historical checkpoints. When true,
explicit backbone edges are created only within the same chain. Cross-chain
neighbors must pass the normal spatial-radius filter, and cross-chain edges no
longer carry array-index-based backbone/short/medium/long sequence flags. Real
spatial cross-chain edges remain. This is not a same-chain-only graph.

`coattn_chain_position_fix` also defaults false. When true, the already-computed
chain-gap RoPE positions are explicitly passed into the co-attention joint
forward call, including non-reentrant checkpoint replay. No mutable position
cache is used. This does not enable a new distance bias or forbid cross-chain
attention. Chain-gap RoPE does not mathematically guarantee monotonic distance
attenuation or permutation invariance; those are not claimed.

The fixes add no parameters and preserve seeded initialization. Same-chain
missing-residue discontinuities are not addressed by this first control.

## Validation diagnostics: job13573440

One independent GPU job loops through baseline/GVP/position/both on the same
frozen old G0 epoch52 best-F1 checkpoint. Every pass uses chainwise ESM and the
full complex. Thus this is a frozen architectural-intervention diagnostic, not
an estimate of retrained model performance.

- Population: all950 multichain strict-VAL rows and the first50 single-chain
  rows of each source, selected without prediction scores.
- Fixed thresholds:0.5 and the checkpoint's existing VAL gate; no threshold fit.
- Metrics: fixed global Top20, sample/protein-balanced F1/MCC, pooled AUPRC/AUROC,
  and false positives on zero-contact chains. No per-chain Top20 budget increase.
- Graph audit: count artificial directed backbone bridges and their distances,
  including how many bypass the15 A radius.
- Chain-order reversal: permute every residue-aligned model input consistently,
  retain the same ligand/geometry/labels, unpermute output scores, then measure
  mean absolute probability change and Top20 Jaccard.
- Save all per-residue scores/labels/keys plus per-row audit records. No TEST or
  external benchmark inference. Single-chain permutation is an identity control.

Outputs go to `val_boundary_audit/{protocol,progress,summary,rows}.json` and
`val_boundary_audit/predictions/`. Training logs are under `logs/train_%A_%a.*`.

## Checks performed

- Local CPU:31 tests passed (new boundary controls plus existing GFGF/core-bundle
  checks), including FP32/BF16 checkpoint-gradient parity.
- The96.2 A artificial cross-chain bridge is removed; true close cross-chain
  neighbors remain; same-chain graphs and full-model outputs match controls.
- Seeded parameter tensors and RNG state are unchanged by the two switches.
- Chain positions reach every co-attention self-attention call and replay.
- Shell syntax and Python compilation checks passed.
- HPC's af3 environment lacks pytest. No dependency was installed into it;
  a standalone CPU import/graph/model preflight is used there, followed by the
  scheduled real-data GPU smoke.
- That HPC preflight passed: corrected distant-edge behavior, isolated model
  import path, exactly23,650,742 parameters, both flags enabled, all trainable.

Further preprocessing inspection did not find the same cross-chain continuity
error in V4 backbone torsions: `protein_features_v4.py` checks chain identity
before phi/psi/omega calculation. Dataset `_compute_gvp_vectors` operates inside
each residue rather than using the adjacent chain's atoms. These observations
do not certify missing-residue handling, biological assembly validity, or atom
ordering in every historical feature file; those remain separate audits.

All deployment transfers were direct SSH/tar, not CPU/Slurm transfer jobs.
