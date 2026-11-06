# Star chain-boundary continuation launch, 2026-11-06

## Final status: project paused before execution

The user subsequently requested cancelling this batch and pausing PLC.
Star jobs2862 and2863_[0-2] were explicitly cancelled; Slurm confirmed
`CANCELLED by 1001` with elapsed00:00:00 for both the smoke and the array.
Neither GPU smoke nor continuation ran. The dependency chain is no longer
pending. Other project jobs were not cancelled, and no files were deleted.
All checkpoints/data/configs/logs remain available for an explicitly authorized
future resumption. Do not automatically resubmit these jobs. The launch details
below are historical, not an active queue report.

## Scope and checkpoint selection

The user authorized starting continuation on Star after explicitly cancelling
all three HPC controls. No HPC training/evaluation/extraction/analysis was
launched. The only HPC access was direct migration and checksum acceptance.
The user-stopped Star ESM-only baseline2843 is NOT restarted.

Final HPC best-F1 and best-loss checkpoints, config, TB and logs were copied
to `/data1/PLC/archive/hpc_chain_boundary_final_20261106/experiments`.
The2,566,410,418-byte snapshot passed a content-checking rsync dry run with
exit0 and no changed-file records. No original artifacts were deleted.

| Star array task | Variant | Saved epoch | Start epoch | Saved VAL F1_c0 |
| --- | --- | ---: | ---: | ---: |
| 2863_0 | GVP boundary | 14 | 15 | 0.587773 |
| 2863_1 | Co-attention positions | 19 | 20 | 0.589223 |
| 2863_2 | Both fixes | 15 | 16 | 0.586951 |

Co-attention improved after the earlier preflight, so its final epoch19 state
supersedes the initially copied epoch16 state. The both-fixes arm has two
source runs because HPC requeued it: original best0.586951 at epoch15 versus
restarted best0.523089 at epoch3. Select the original best by existing VAL F1,
not file mtime. No TEST or external benchmark result was used for selection.

## Continuation implementation

Root: `/data1/PLC/experiments/chain_boundary_resume_20261106`.
Core data/runtime: `/data1/PLC/experiments/chainwise_esm_core_20261102`.

An independent source snapshot and explicit Star-only resume entrypoint are
used. The original scratch wrappers still reject resume. Preparation validates
the full existing core feature inventory, frozen fixed-model source, config,
split and normalization artifacts. Only paths and experiment naming relocate;
model, batch16, loss, LR schedule, seed44, F1 early stopping and checkpoint
selection remain unchanged. No features or ESM weights need re-extraction.

Resume restores model, AdamW, scheduler, AMP scaler, saved epoch, best metrics
and early-stop state. Missing/mismatched checkpoint state fails closed instead
of silently falling back to scratch. New outputs are separate from immutable
initial checkpoints. Existing best-F1 and best-loss weights are retained even
if continuation never improves them. No independent AUPRC/AUROC checkpoints
or periodic/latest checkpoints were enabled.

The stratified VAL/TB counter now accepts a completed-epoch offset, so resumed
validation is not mislabeled as epoch1. Existing scratch behavior defaults to
zero and is unchanged. A start marker prevents an accidental replay from
overwriting an already-started continuation.

Historical checkpoints have no RNG/sampler state and no latest/batch-level
state. These are explicitly best-epoch continuation branches, not exact
interrupted-batch resumes or independent seed replications. Early-stop patience
continues from the checkpoint's saved state (zero at a best-F1 save), not from
the later cancelled job's log tail.

## Queue and verification

- Short test: Star job2862, one GPU,8 CPUs,64G,1h, background QOS. It tests all
  three arms sequentially on real TRAIN/VAL batches with restored optimizer
  and AMP state,20 cached steps plus cap1800 stress. Updates are discarded.
- Continuation: Star array2863_[0-2%1], one GPU per task,8 CPUs,64G,72h per
  task. `afterok:2862` blocks all training unless all three smoke tests pass.
  Invalid dependencies cancel rather than remain pending indefinitely.
- Array throttle1 and the shared `/data1/PLC/core_training.lock` enforce one
  PLC GPU task at a time. Slurm device mapping is retained; no GPU is selected
  outside an allocation. `Requeue=0`; no automatic restart.
- At submission, smoke was PENDING with `AssocGrpGRES`, and training was
  PENDING with `Dependency`. Two unrelated jobs occupied Star's two GPUs.
  Neither was stopped or preempted; actual GPU smoke/training is not yet
  verified. The script's critical-smoke comment describes an optional mode;
  this submission uses its default background account/QOS.
- Local CPU regression tests:30 passed, including optimizer/scheduler next-step
  parity and resumed VAL/TB indexing. Shell syntax passed. Star core inventory
  preparation and the actual frozen trainer/helper imports passed on CPU.
  This is not a claim that the pending GPU smoke has passed.

Entrypoints:
`scripts/train/train_chain_boundary_resume_star.py` and
`scripts/slurm/run_chain_boundary_resume_star.slurm`.
Prepared configs, immutable input checkpoints, smoke reports, new outputs and
Slurm logs are isolated under `prepared/`, `initial/`, `smoke/`, `outputs/`,
and `logs/` in the continuation root. Hashes, candidate checkpoints and queue
metadata are recorded in
[`results/star_chain_boundary_resume_launch_20261106.json`](results/star_chain_boundary_resume_launch_20261106.json).
