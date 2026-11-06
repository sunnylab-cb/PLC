# Chain-boundary controls: Star resume feasibility, 2026-11-06

Subsequent user-authorized implementation and queued Star continuation are
recorded in [the launch report](star_chain_boundary_resume_launch_20261106.md).
That report uses the final post-cancellation checkpoint snapshot; the table
below remains the earlier compatibility preflight.

## Later update: HPC jobs cancelled

The user subsequently explicitly requested cancellation. Only jobs13573439_0,
13573439_1, and13573439_2 were cancelled. Slurm confirmed all three as
`CANCELLED by 204805`, with end time2026-11-06T13:28:02 as reported by HPC,
and no remaining entries for this array in `squeue`. The old allow-finish
exception is revoked. No files were deleted and no Star job was submitted
during cancellation. The checkpoint table below describes the earlier copied
snapshot, not necessarily the last checkpoint written before cancellation;
reconcile final source artifacts before any future resume.

## Earlier preflight

The user requested checking migration/resume of the existing HPC controls.
This pass copied artifacts and ran CPU compatibility checks on Star only.
No HPC job was stopped/requeued/restarted, and no Star GPU job was submitted.
This does not revoke the earlier user-directed stop of Star baseline2843.

## Verified checkpoints

| Variant | Saved epoch (1-based) | First resumed epoch | Saved best VAL F1_c0 |
| --- | ---: | ---: | ---: |
| GVP boundary | 14 | 15 | 0.587773 |
| Co-attention positions | 16 | 17 | 0.586809 |
| Both fixes | 15 | 16 | 0.586951 |

All three contain model, AdamW, scheduler, AMP scaler, F1-based early-stopping,
VAL metrics/gate, config, and epoch state. There is no saved RNG state and no
latest/per-batch checkpoint. Resume means branching from the saved best epoch,
not continuing the current HPC batch exactly.

Safe `torch.load(weights_only=True, map_location="cpu")` succeeded on Star
Torch2.9.1+cu126. The archived fixed model source strictly loads all three
state dicts (23,650,742 parameters). AdamW state loads with matching momentum
tensor shapes and finite values; scheduler state loads with exactly matching
LRs and epoch indices. The training recipe equals the existing Star core
config; model-config differences are only the two intended fix flags.

No GPU forward/backward, scaler update, real-data resumed step, or full
training launch was tested. Cross-host kernels and unsaved RNG/sampler state
prevent claims of bitwise continuation or identical optimization trajectories.

## Artifacts

Direct HPC-to-Star rsync copied854,024,243 bytes of three best-F1 checkpoints
and configs to:
`/data1/PLC/archive/hpc_resume_preflight_20261106/experiments/chain_boundary_controls_20261104/outputs`.
No source deletion or transfer Slurm job was used. Each safely loaded file's
SHA256 and restored-state details are in
[`results/star_chain_boundary_resume_preflight_20261106.json`](results/star_chain_boundary_resume_preflight_20261106.json).

The matching isolated fixed source is already archived under
`/data1/PLC/archive/hpc_acceptance_20261106/experiments/chain_boundary_controls_20261104/model_code`.
The core feature caches, metadata, and runtime remain under
`/data1/PLC/experiments/chainwise_esm_core_20261102`. No ESM weights or feature
re-extraction are necessary for this migration.

## Requirements before launch

1. Obtain the user's decision on stopping the still-running HPC controls and
   moving them to Star serial execution. Preserve the existing snapshots.
2. Prepare a separate Star run root, with HPC data/split/ESM/metadata paths
   relocated to the already-verified Star core bundle. Keep all modeling and
   training hyperparameters unchanged.
3. Add an explicit resume entrypoint: current scratch wrappers reject
   `--resume`, although the underlying trainer restores optimizer/scheduler,
   scaler, epoch and early-stop state. Fail if the resume file is absent;
   never silently fall back to scratch.
4. Copy the resume checkpoint into the new run directory before invoking the
   trainer. Its setup function uses the checkpoint parent as the output
   directory; do not run directly in the immutable archival directory.
5. Offset the chain-stratified VAL counter/TB steps to the restored epoch.
   The scratch wrapper currently initializes its `val_pass` counter at zero.
6. Run a bounded real-data GPU resume smoke inside a Star Slurm allocation;
   discard smoke updates and reload the original state before continuation.
   Retain F1 selection/early stopping and best-loss checkpoint policy.
7. Use one project GPU task at a time, global lock, and `--no-requeue`.
   No automatic replay of old HPC launchers or restarted stopped Star baseline.

Star has an unrelated `openfold3-long-batch` job running under the same user;
do not cancel it or infer that an empty PLC queue grants exclusive GPU access.
Migration is feasible but serial Star training is not a speedup over three
parallel HPC GPUs. Final GPU compatibility and explicit cutover are pending.
