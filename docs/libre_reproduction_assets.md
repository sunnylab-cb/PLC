# LiBRe reproduction assets

## Status (2026-10-09)

- Requested destination: HPC2 HDD, not local storage or jhspoolers.
- SSH alias `hpc2` connects successfully to `mgmt-4`.
- Official repository: https://github.com/GoldRiver0/LiBRe
- GitHub metadata reports `models/model.pt` is 5,416,270 bytes, with git blob
  SHA-1 `f97a182b5fecc86e04d8789eb6be80043acf1412` at inspection time.
- Existing ESM2 weights and contact-regression weights are present on HPC2.
- After explicit user approval, all 10 selected official files were downloaded
  directly on HPC2 and checked against upstream Git blob hashes. SHA-256 digests
  are saved in the remote manifest. Completed at 2026-10-09 15:37 UTC.
- Pinned commit: `e9151a85bab2d05b67b953bc55a74c780aa2bf24`.
- Checkpoint SHA-256:
  `ba40bdf78ba4e596e7429e56d2c955c42b6debc67fb159a7438dfe6b9993c871`.
- Existing ESM2 files were hashed and symlinked, not downloaded or copied.
- Remote manifest, symlink targets and disk usage were read back successfully.
  Total allocated disk usage reported by `du -sh` is 6.1M.
- At download completion, no upstream code was executed and no inference job was
  submitted. Subsequent inference preparation is recorded separately below.

## Confirmed storage layout

```text
/hpc2hdd/home/shuang886/PLC/external_baselines/libre/
  source/<upstream_commit>/       # selected official source, README, example CSVs
  checkpoints/<upstream_commit>/model.pt
  checkpoints/esm2/              # symlinks to existing torch cache
  manifests/<upstream_commit>.json
  features/                     # future LiBRe-specific feature caches
  logs/                         # download and future execution logs
  results/                      # future predictions and benchmark summaries
```

Reuse without copying or downloading:

```text
/hpc2hdd/home/shuang886/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
/hpc2hdd/home/shuang886/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt
```

The helper `scripts/benchmark/prepare_libre_assets.py` resolves an upstream
commit, downloads only selected assets directly from HPC2, checks each Git blob
hash and file size, and records SHA-256 digests. It does not execute upstream
code, deserialize checkpoints, install packages, download ESM2, or submit jobs.
Re-running against the same revision verifies existing files instead of
overwriting them. Local syntax validation and remote asset preparation passed.

## Reuse and next steps

For repeat verification, run the helper on HPC2 using the root and ESM cache
paths above and `--revision e9151a85bab2d05b67b953bc55a74c780aa2bf24`.
Stream or copy only this small helper from the local workspace. Keep all large
assets and feature generation on HPC2. The completed download log is
`logs/assets_download_20261009.log`, and the manifest is
`manifests/e9151a85bab2d05b67b953bc55a74c780aa2bf24.json` under the remote root.

Subsequent inference requires a separate environment/import and state-dict
compatibility check. Feature preparation must use official ESM2 layer 33
representations and 40D ligand atom features rather than substituting PLC's
normalized ESM or local54 caches without equivalence checks.

Report both official pooled-residue F1 (default threshold 0.6) and PLC's
sample/protein-balanced metrics on matched samples. Explicitly audit chain,
residue, ligand and label alignment, length handling, and training overlap.

## Inference smoke preparation (2026-10-09)

- Existing Python: `/hpc2ssd/softwares/anaconda3/envs/af3/bin/python`.
- Imports passed: torch 2.9.1+cu128, torch_geometric 2.7.0, fair-esm 2.0.0,
  RDKit 2026.03.1, pandas, sklearn and tqdm. No packages installed.
- Official checkpoint loaded with `weights_only=True`, `strict=True`,
  `use_cnn_lstm=True`, `use_ligand=True`, `residue_input_dim=1280`.
  All 68 state-dict entries matched; the model has 1,348,481 parameters.
- Runner: `scripts/benchmark/smoke_libre_inference.py`.
- Slurm: `scripts/slurm/run_libre_inference_smoke.slurm`.
- Both scripts are deployed under the remote root's `runners/` directory.
  Local and remote SHA-256 digests match.
- Job **12959934**, `emergency_gpu`, one GPU, four CPUs, 32 GB host memory,
  30-minute limit. Initially PENDING (Priority); subsequently failed at the
  released CLI parsing stage, as documented below.
- Planned artifacts: `results/smoke_12959934/complete.json`, predictions,
  per-sample diagnostics, feature tensors, and official CLI log. Slurm stdout
  and stderr are `logs/smoke_12959934.out` and `.err`.

The smoke uses the first eight official test examples in released order. It
reproduces the upstream batch-8 ESM slicing and fixed 1500-position input, and
separately measures the effect of removing retained EOS tokens from short rows.
The original ESM code excludes padding but not necessarily EOS after a global
`[1:-1]` slice. This can affect the unmasked bidirectional LSTM and attention even
though metrics only cover true residue positions. No upstream source is patched.

Checks include input/label/graph alignment, pinned asset hashes, strict checkpoint
loading, finite and repeatable predictions, agreement with upstream `evaluate`,
and a separate released `evaluate.py` CLI run from saved inputs. Any reported
scores on these eight examples are smoke diagnostics, not benchmark evidence.

ESM2 is loaded exclusively from the hash-verified existing cache via the core
loader. Its legacy configuration requires explicit `weights_only=False`; the
newly downloaded LiBRe state dict continues to use `weights_only=True`.

Remote login emits a stray `proxy` diagnostic and breaks rsync/scp protocols.
The small scripts were transferred via tar streamed over SSH stdin instead;
no user shell configuration was changed. Slurm executables are addressed through
`/opt/slurm/bin`, which is absent from the current noninteractive login PATH.

## First smoke result and retry (2026-10-10)

- Job 12959934 ran on gpu1-25 for 40 seconds; FAILED 1:0, not OOM.
- Asset hash checks, strict model load, ESM2 extraction, finite/repeatable forward
  passes and in-memory upstream metric parity all passed.
- Eight-example diagnostics at threshold 0.6: pooled F1 0.6772727,
  precision 0.5246479, recall 0.9551282, MCC 0.6756954, AUROC 0.9770425,
  average precision 0.7566386; sample-mean F1 0.6547387. These are not full
  benchmark scores or evidence of superiority over PLC.
- Failure occurred in the separate upstream evaluate.py CLI: pandas inferred
  `label_sequence` as integers, and LiBReDataset's `list(lbl)` raised
  `TypeError: 'int' object is not iterable`. No complete.json was produced.
- Added `scripts/benchmark/libre_cli_compat.py`, which runs the unchanged official
  CLI with an explicit string dtype for the label column. Read-back assertions
  verified all eight label strings match csv.DictReader exactly, including
  leading zeros. No model, threshold, labels or upstream files were changed.
- Retained EOS diagnostic: seven of eight short rows retain one extra EOS
  embedding under upstream batching. Removing these changes residue probability
  by mean 0.0002543, maximum 0.0558305 on this smoke subset. Do not extrapolate
  this to a benchmark-wide effect.
- Submitted corrected retry **12960102** with the same one-GPU emergency_gpu
  resource specification. Its output uses `results/smoke_12960102/` and
  `logs/smoke_12960102.out` / `.err`. Original failed artifacts remain intact.

## Inference acceptance passed (2026-10-10)

- Retry **12960102 COMPLETED 0:0**, gpu1-39, 51 seconds wall time;
  runner elapsed 48.89 seconds. Slurm stderr is empty.
- Verified `results/smoke_12960102/complete.json`: `complete=true`, eight rows,
  1,895 real residues, no skipped rows; strict checkpoint loading, finite outputs
  and repeated-forward checks passed. Official CLI exit code is zero.
- Pooled F1 0.6772727, sample-mean F1 0.6547387, precision 0.5246479,
  recall 0.9551282, MCC 0.6756954, AUROC 0.9770425, AP 0.7566386.
  The official CLI reports F1 0.6773, matching at its printed precision.
  CLI AP prints 0.7567 versus in-memory AP 0.7566386; bitwise probability or
  all-metric equality across the two processes was not an acceptance condition.
- Removing the retained EOS embeddings leaves pooled F1 unchanged on these eight
  examples. It changes probabilities (mean absolute 0.0002543, maximum 0.0558307),
  so preprocessing policy still needs to be explicit for benchmark inference.
- End-to-end inference feasibility is established with the CSV dtype shim.
  This is not a full COACH420/HOLO4K evaluation or a reproduction of paper scores.

## Matched external benchmark (2026-10-10)

The user authorized the full evaluation on `i64m1tga800ue`. The new runner is
`scripts/benchmark/run_libre_benchmark.py`; the array launcher is
`scripts/slurm/run_libre_benchmark.slurm`. Each of the two tasks requests one
GPU, four CPUs, 48 GB host memory and a 12-hour wall limit (not an ETA).
The remote isolated code snapshot is `runners/benchmark_v1/` under the LiBRe
root. No PLC model code or upstream LiBRe source is changed.

Protocol fixed before results:

- Task 0 evaluates COACH420: 494 ligand rows / 299 proteins. Task 1 evaluates
  HOLO4K: 7,245 ligand rows / 3,430 proteins. These are the current PLC benchmark
  populations, not claims of full original-dataset coverage.
- Reference rows are frozen GFGF job 12956268 `val_gate` CSVs under
  `experiments/protein_hybrid_gdn_20261006/benchmark_bestf1_12956268/`.
  Reference row count, contact count and true-contact centroid are checked.
- Residue identity comes from the conservative protein V4 cache. Tags are
  independently verified against atom-H PDB/SDF heavy-atom distances and then
  joined by `(chain, residue number, insertion code)`, never by blind position.
- Inputs are freshly extracted raw ESM2 layer 33 FP32 chain representations,
  no PLC normalization, BOS/EOS excluded, and the released SMILES-to-40D graph
  function. Sanitized SDF chemistry supplies canonical isomeric SMILES. Native
  ligand coordinates do not enter the predictor. Chemical identity/bond-order
  correctness beyond existing SDF QC remains a limitation to audit separately.
- Every chain is evaluated against the current ligand; no ground-truth binding
  chain is selected. Observed chains longer than 1500 use overlapping windows
  (step 1250, tail anchored), merged by mean probability. Each LiBRe input retains
  its trained 1500-position zero padding. This is a documented external-protocol
  adaptation, not an exact reproduction of the original paper's chain protocol.
- Metrics use the same first `min(full protein length, 1800)` residues as PLC.
  Whole-chain sequence context can extend beyond this metric axis. The existing
  reference cap and any contacts beyond it are recorded; this does not introduce
  an additional silent 1500-residue evaluation cutoff. Ligand atoms are uncapped.
- Threshold is the official fixed `probability > 0.6`, not test-tuned. Report
  pooled contact F1/MCC/AP/AUROC, sample mean, protein mean of ligand means and
  protein mean of ligand maxima. Contact means heavy-atom distance below 5 A;
  LiBRe's binary output is NOT PLC's contact-plus-near (8 A) binary metric.
- PDR/LAA retain PLC's single-centroid diagnostics and BP is F1 conditional on
  localization success. Report their valid denominators; do not describe BP as
  unconditional pocket detection accuracy or these proxies as standard SOTA
  metrics. Reference-subset candidate centroids are recomputed after exclusions.
- Invalid inputs are logged in `failures.json`, never silently repaired or
  omitted from coverage reporting. Any missing row makes `complete=false` and
  the job exits nonzero after writing partial results. PLC metrics on exactly
  the successful subset are included if exclusions occur.
- Feature caches use exact sequence SHA-256; per-row predictions use protocol
  and input-file fingerprints. Atomic writes support restart without silently
  reusing changed inputs. Large features/predictions stay on HPC2.

Validation before submission: six local unit tests passed (window coverage,
invalid windows, strict threshold, undefined AUC, conditional assignment/BP,
protein balancing); shell syntax passed. Remote code hashes are checked before
submission. A training-overlap audit for the released LiBRe checkpoint is still
required before interpreting the comparison as leakage-controlled superiority.

Submitted array **12960178_[0-1]** on `i64m1tga800ue` after remote SHA-256,
Python compilation and shell syntax checks passed. Results are written to
`results/benchmark_12960178/{coach420,holo4k}/`; logs are
`logs/benchmark_12960178_{0,1}.{out,err}`. The 12-hour request is a safety limit;
actual ETA should be estimated from `progress.json`, not assumed from that limit.
Initial queue check: both tasks are `PENDING (Priority)`; inference has not yet
started and no GPU runtime or full-benchmark score is available.

Follow-up at 2026-10-10 01:06 CST: a lightweight remote preflight passed for
`1a26A_CNA_A200`, `1a2kC_GDP_C220`, `121p_GCP_A167`, and `12as_AMP_A332`.
Tag geometry, residue/contact counts and official 40D graph conversion passed.
Their residue counts are 351, 196, 166 and 656; graph atom counts are 27, 28,
32 and 23. This is input acceptance only, not GPU inference evidence.

Both live array tasks remain pending. To allow shorter backfill opportunities,
`scontrol update` reduced the existing task time limits to **2 hours for task 0**
and **6 hours for task 1**, with no cancellation, duplicate submission, partition
change or resource increase. The submitted launcher retains its conservative
12-hour default; the current job records are authoritative for this run.
Slurm temporarily predicts **2026-10-10 22:58 CST** for start; this is a mutable
scheduler estimate, not measured inference duration. After array materialization,
task 0 has internal JobId 12960205 but remains addressable as `12960178_0`;
task 1 remains `12960178_1`.

The next scheduler poll after the time-limit reduction moved the estimated
starts earlier: COACH task 0 to **2026-10-10 19:42 CST**, HOLO task 1 to
**2026-10-10 21:42 CST**, both scheduled on gpu1-29. Both are still
`PENDING (Priority)`; these estimates may change again.

## Startup failure and corrected resubmission (2026-10-10)

Both 12960178 tasks actually started on gpu1-11 at 03:12 CST, earlier than the
queue estimates, and failed within one second with `cd: too many arguments`.
The launcher accidentally concatenated `cd "${ROOT}"` and `nvidia-smi` on one
line. This was an authoring error, not OOM, an input failure or model behavior.
No benchmark prediction was produced. `bash -n` alone did not catch this valid
but incorrect shell command.

The launcher now uses separate commands and supports explicit environment path
overrides for testing. A new regression test executes both array branches with
stubbed GPU/Python commands, checking working directory and complete argument
construction (including paths containing spaces). All seven tests pass. The
same startup dry-run and pinned hashes are checked remotely before resubmission.
The corrected snapshot is `runners/benchmark_v2/`; the Python inference runner,
model checkpoint and protocol are unchanged. The default request is now six
hours, with COACH's pending task shortened to two hours. Original failed
scripts/logs remain under their original names.

Corrected array: **12960610_[0-1]**, still `i64m1tga800ue`, one GPU per task.
Remote hash checks and both actual shell startup dry-runs passed. New result root:
`results/benchmark_12960610/`; logs: `logs/benchmark_12960610_{0,1}.{out,err}`.

Both corrected tasks finished traversal on 2026-10-10 (03:21:59 and 03:49:21
CST). Strict coverage exits are nonzero because 3 COACH and 7 HOLO SDFs fail
chemical input validation. Successful predictions cover 491/494 and 7238/7245
requested rows; no OOM occurred. Full accounting, matched comparisons, bootstrap
intervals and the ten-row CCD audit are in
[the matched benchmark report](libre_benchmark_20261010.md).
This is not a claim of 100% successful input coverage.

## Final report acceptance (2026-10-10)

The user approved explicitly excluding the three COACH and seven HOLO chemistry
failures from both models. The finalized comparison uses 491 / 7238 identical
pair rows for LiBRe and the frozen PLC GFGF reference. The exact whitelist is
`docs/libre_benchmark_20261010_exclusions.json`; the verified machine-readable
report is `results/libre_benchmark_12960610/final_report.json` locally.
Original Slurm failure states, inference summaries, input coverage and chemistry
inputs are preserved. Report completion under this approved policy is distinct
from full-population inference success. No GPU rerun or input repair is needed.
