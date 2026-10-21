# Joint Reference-Only TEST, 2026-10-21

## Scope and Decision Boundary

The user authorized this evaluation with an explicit restriction: TEST is
reference-only and must not drive subsequent iteration. Model selection,
architecture/loss changes, hyperparameter selection and threshold calibration
remain TRAIN/VAL-only. Do not select an experiment direction from TEST gaps,
compare checkpoints on TEST, or sweep a TEST threshold.

One checkpoint and one existing VAL-selected threshold were locked before
inference. This job launches no new training.

## Locked Protocol

| Item | Value |
| --- | --- |
| Training job | 13187469, completed |
| Model | V11m13 group_rescompat + GFGF, scratch seed44 |
| Selection | Highest training-logged VAL F1_c0, epoch25 |
| Checkpoint | `best_f1_c0_model.pt` |
| Fixed VAL gate | 0.5535550117492676 |
| TEST rows | 17,395: PDBbind 1,649; BioLiP 15,746 |
| Modes | real, shuffle, zero, sequentially on one GPU |
| Precision / batch | BF16 / 8, fixed before TEST |
| Length policy | Original max_seq_len=1800; max_atoms=256, alternating_ends |
| Shuffle | Seed42, within-source bijection, no self-index pairing |
| Invalid predictions | Fail on nonfinite valid-residue outputs; no silent row dropping |

The checkpoint/config/split/normalization assets and evaluation source hashes
are locked in the remote protocol. Model and dataset source hashes are checked
against the original training snapshot. All modes must retain the same sample
and label axes. Preparation verified provenance without running TEST inference.

Shuffle is a dependency diagnostic, not a certified biological nonbinder:
different rows can still represent equivalent ligands or sites. Zero removes
atom feature/coordinate values, bond attributes and UniMol values, while
retaining atom counts and edge indices. It is not a true protein-only model.

## Submission and Outputs

- Job: **13244903**, submitted October21 09:48:50, UTC+8.
- Partition: `i64m1tga800ue`, one GPU, 8 CPUs, 200G RAM, 12-hour limit.
- First queue receipt: PENDING(Priority). No TEST result was available then.
- At09:51 UTC+8: RUNNING on `gpu1-38`, elapsed2m29s; no logged error or
  completed inference result yet. Slurm start time is09:48:51.
- Evaluation code: commit `3e4dac1`.
- Protocol SHA256: `86ca9c922daef5f4d121c064b85b28c56ff5ca6f6f08bbc7568f41115f0313ec`.
- Deployment archive SHA256: `54c6ee6eaa2857d604eb3f5e6ff7657752663fbea7a6b607655aecdcd751403b`.

Remote evaluation root:

```text
/hpc2hdd/home/shuang886/PLC/experiments/joint_reference_test_20261021
```

The root retains immutable `code_v1`, `deployment.json`, `submission.json`,
logs and `results/protocol.json`. Evaluation refuses an already-started output
directory; submission also uses a lock and receipt to prevent duplicate jobs.

Expected results are `predictions_{real,shuffle,zero}.pt`, per-mode summary
JSONs, `progress.json` and `reference_test_summary.json`. Report all rows and
each source separately: AUPRC/AUROC, F1/MCC/precision/recall at 0.5 and the
fixed VAL gate, argmax metrics, sample-macro metrics, and Top L/20, L/10, L/5.
Real-minus-shuffle/zero gaps are descriptive only, not optimization targets.
Runtime and CUDA allocated/reserved peaks are recorded for execution auditing.

The evaluator does not call threshold-search helpers. Targeted evaluation,
compatibility and preparation tests: **23 passed**; Slurm shell syntax passed.

Training results and comparison caveats are recorded in
[the completed VAL report](joint_global_si30_results_20261021.md).
