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

## Completed Results, Verified 2026-10-22

Slurm reports COMPLETED, exit0:0, from2026-10-21 09:48:51 to10:25:41
(UTC+8), elapsed36m50s. Stderr is empty and stdout ends with
`REFERENCE_TEST_COMPLETE`. The user queue was empty at this inspection.
No new training, reevaluation, checkpoint selection or threshold sweep was
launched while collecting these results.

Protocol SHA256 matches the preregistered value. All three per-mode JSONs
match the final summary. Each mode covers17,395 rows and6,539,679 residues,
including262,533 class0 positives; nonfinite-residue count is0. Residue/label
axis SHA256 is identical across modes:
`b7055cdf39b9e0568e1995a9a82d411de350724d835f686b1e64786714c48b21`.

### Fixed VAL Gate Metrics

All F1/MCC/P/R below pool residues within the stated population and use the
locked VAL gate0.5535550117492676, not a threshold fitted on TEST. AUPRC and
AUROC are global ranking metrics. These are one seed and one shuffle draw;
no seed variance or significance claim is implied.

| Population | Mode | F1 | MCC | Precision | Recall | AUPRC | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | real | 0.595887 | 0.578981 | 0.564499 | 0.630972 | 0.615422 | 0.950212 |
| all | shuffle | 0.484521 | 0.462469 | 0.466491 | 0.504001 | 0.451392 | 0.911189 |
| all | zero | 0.336419 | 0.380408 | 0.696219 | 0.221797 | 0.471618 | 0.925836 |
| pdbbind | real | 0.621448 | 0.608982 | 0.611823 | 0.631380 | 0.660390 | 0.961091 |
| pdbbind | shuffle | 0.533368 | 0.517958 | 0.524057 | 0.543015 | 0.536703 | 0.934230 |
| pdbbind | zero | 0.365230 | 0.416565 | 0.751331 | 0.241252 | 0.514689 | 0.947750 |
| biolip | real | 0.592692 | 0.575037 | 0.558832 | 0.630919 | 0.609112 | 0.948306 |
| biolip | shuffle | 0.478276 | 0.455063 | 0.459298 | 0.498891 | 0.439933 | 0.907094 |
| biolip | zero | 0.332637 | 0.375509 | 0.688935 | 0.219248 | 0.465972 | 0.922114 |

| Population | Rows | Real-shuffle F1 | Real-zero F1 | Real sample-macro F1 |
| --- | --- | --- | --- | --- |
| all | 17395 | 0.111366 | 0.259468 | 0.551983 |
| pdbbind | 1649 | 0.088080 | 0.256218 | 0.611280 |
| biolip | 15746 | 0.114415 | 0.260054 | 0.545773 |

### Other Preregistered Metrics

| Population | Real F1@0.5 | Real MCC@0.5 | Real global argmax F1 | Real global argmax MCC |
| --- | --- | --- | --- | --- |
| all | 0.593724 | 0.577887 | 0.586024 | 0.570774 |
| pdbbind | 0.619732 | 0.607602 | 0.616483 | 0.604590 |
| biolip | 0.590479 | 0.573971 | 0.582261 | 0.566382 |

Class0 is one of three softmax classes: argmax(class0) is not the same
rule as p(class0)>=0.5. These global values are also not the legacy
batch-averaged VAL F1_c0. Sample-macro weights every ligand row equally,
not every protein equally, and is not interchangeable with pooled F1.

| Fraction | Real macro precision | Real macro recall |
| --- | --- | --- |
| L/20 (5%) | 0.522942 | 0.553765 |
| L/10 (10%) | 0.367705 | 0.718326 |
| L/5 (20%) | 0.221668 | 0.832494 |

### Descriptive Interpretation Only

- Real outperforms within-source shuffled ligand input in both sources,
  supporting sensitivity to the matched ligand under this fixed protocol.
  This does not prove correct same-protein multi-pocket assignment, nor
  certify that shuffled ligands are biologically incompatible.
- Zero reduces overall recall from0.630972 to0.221797 while precision rises
  from0.564499 to0.696219. Its F1 gap combines lost input information with
  an out-of-distribution intervention and threshold sensitivity; a larger
  zero gap is not intrinsically better ligand recognition. Overall zero
  AUPRC (.471618) exceeds shuffle AUPRC (.451392) despite lower gated F1,
  so ranking and fixed-threshold behavior should not be conflated.
- The same selected checkpoint's logged VAL global F1/MCC at its gate are
  .606903/.590500 versus TEST .595887/.578981; AUPRC is .625655 versus
  .615422. These are descriptive holdout differences, not a matched
  generalization experiment; VAL is selection-biased and its logged ranking
  metrics excluded a small nonfinite subset, unlike the complete TEST run.
- The fixed gate changes overall TEST pooled F1 from.593724 at0.5 to.595887,
  but sample-macro F1 from.553922 to.551983. Both rules were preregistered;
  this observation must not be used to choose a different threshold.
- The PDBbind and BioLiP subsets differ, and neither is the previous
  PDBbind-only holdout. These results cannot establish the causal gain
  from adding BioLiP or be ranked directly against historical split scores.
  The split certifies global SI<0.30, not local/domain independence or
  COACH420/HOLO4K dehomology.

No experimental recommendation is derived from these TEST results.
Any further development decisions must be justified independently on TRAIN/VAL.

### Resource Record and Archive

| Mode | Evaluation/summary seconds | Peak allocated GiB | Peak reserved GiB |
| --- | --- | --- | --- |
| real | 893.86 | 3.943 | 11.957 |
| shuffle | 538.33 | 3.943 | 11.959 |
| zero | 326.39 | 3.943 | 11.959 |

These are batch8 BF16 **inference** peaks, not training peaks or evidence
that a larger training batch fits. Slurm batch MaxRSS is47,085,004 KiB
(44.90 GiB). The total job includes startup, integrity checks and I/O.

The [raw aggregate JSON](results/joint_reference_test_20261021.json) preserves
all reported metrics and the reference-only policy. Per-sample predictions
and the detailed protocol remain at the remote evaluation root; they were
not downloaded or used for a new optimization pass.
