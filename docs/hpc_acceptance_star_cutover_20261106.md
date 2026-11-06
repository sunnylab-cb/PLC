# HPC acceptance and Star-only cutover, 2026-11-06

## Update: Star training stopped by user

Later on November 6 the user requested stopping the Star training after the
observed plateau. Only PLC job2843 was cancelled with `scancel 2843`; no host
shutdown, other-job cancellation, HPC operation, or replacement run occurred.
Slurm confirmed `CANCELLED by 1001` after69h38m41s and an empty queue entry.
The last logged progress was epoch30 batch4300/8698. Epoch29 is the last
completed validation, so epoch30 is not reported as a completed result.

| Selection/observation | Epoch | F1_c0 | Pooled AUPRC | Pooled MCC at0.5 |
| --- | ---: | ---: | ---: | ---: |
| Best-F1 checkpoint retained | 14 | 0.591924 | 0.6222 | 0.5946 |
| Highest observed AUPRC, not a separately selected checkpoint | 24 | 0.590321 | 0.6375 | 0.5955 |
| Last complete validation | 29 | 0.588521 | 0.6271 | 0.5935 |

F1 had not improved for15 validations; the last5 F1 mean was0.584521 versus
0.588318 for the preceding5. This is a user-directed plateau stop, not automatic
patience30 early stopping, wall-time termination, or proof of full convergence.
Do not resume/relaunch automatically. The following sections retain the earlier
acceptance snapshot and are not a live queue view.

The original checkpoint/config/logs/TB were retained without deletion. Best
checkpoint on Star:
`/data1/PLC/experiments/chainwise_esm_core_20261102/outputs/joint_si30_G0_chainwise_esm_fullcomplex_seed44_chainesm_20261103_035403/best_f1_c0_model.pt`.
Post-stop SHA256:
`a96fcf2bcf3e3ac232119ba2fc9cb0d2a339487824626a0a79e98e426b70a2af`.

## Execution policy

All future experiments run on Star. Do not submit/requeue/restart training,
evaluation, extraction, or analysis on HPC. The user explicitly permits only
the three existing controls `13573439_0`, `_1`, `_2` to finish naturally.
Read and migrate their final artifacts afterward; do not replace failed jobs
on HPC. No jobs were cancelled or submitted during this acceptance pass.

Star retains the one-project-task/one-GPU limit. File transfers are direct
SSH/rsync/scp, never Slurm jobs. These rules are persisted in `AGENTS.md`.

## Training snapshot

Metrics were collected on November 6. F1_c0 is the existing trainer's selection
metric, NOT pooled residue F1 at threshold 0.5. AUPRC and MCC below are pooled
residue metrics at that run's best-F1 epoch, rounded from training stdout.
Detailed source/chain strata retain full precision in the accompanying JSON.

| Run / job | State | Completed epochs | Best F1_c0 | Best epoch | AUPRC at best | MCC at 0.5 at best |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HPC ESM-only / 13468092 | failed during epoch55 | 54 | 0.604437 | 42 | 0.6435 | 0.6048 |
| GVP boundary / 13573439_0 | running epoch12 | 11 | 0.584838 | 9 | 0.6137 | 0.5885 |
| Co-attention positions / 13573439_1 | running epoch12 | 11 | 0.581308 | 9 | 0.6100 | 0.5852 |
| Both fixes / 13573439_2 | running epoch13 | 12 | 0.581216 | 12 | 0.6234 | 0.5913 |
| Star ESM-only / 2843 | running epoch26 | 25 | 0.591924 | 14 | 0.6222 | 0.5946 |

The old HPC ESM-only run stopped with `DataLoader worker ... Bus error` during
epoch55. The error suggests shared-memory pressure, but does not establish the
exact resource cause. It is not a recorded CUDA OOM or F1 early stop. The best
checkpoint remains available and has been copied to Star. Do not restart it
on HPC.

The new controls passed their real-data GPU smoke and cap1800 stress checks.
They are early runs, not completed experiments. Compare equal training budgets:

| Run | Best F1_c0 in first 11 epochs | Multichain PDBbind AUPRC at epoch11 |
| --- | ---: | ---: |
| HPC ESM-only | 0.580020 | 0.587123 |
| GVP boundary | 0.584838 | 0.596720 |
| Co-attention positions | 0.581308 | 0.595925 |
| Both fixes | 0.579225 | 0.603397 |

Early ranking and selection F1 are not identical: both fixes have the highest
multichain AUPRC at epoch11, without the highest selection F1. None establishes
a final gain. Same-seed runs on different hosts are not independent seed
replicates. Do not compare the controls' epoch12 directly with the reference's
epoch42 and call them failures.

At the HPC ESM-only best epoch42, PDBbind single-chain/multichain Top20 sample
F1 is 0.682255/0.571015 and pooled AUPRC is 0.790597/0.615853. The remaining
context gap is substantial within strict VAL itself. This does not identify
its cause or establish external benchmark improvements.

## Completed frozen VAL audit

Job `13573440` completed successfully. Its protocol evaluates the OLD G0
epoch52 weights, with chainwise ESM and full-complex input in every mode.
It does not evaluate the newly trained fixes. Population: all950 multichain
strict-VAL pairs plus100 single-chain controls, 50 from each source. No TEST
or external benchmark inference was used.

Multichain results: Top20 means exactly20 residues per complex, not20 per chain
and not the top20 percent. F1/MCC below are sample means, AUPRC is pooled.

| Frozen intervention | Top20 F1 | F1 at 0.5 | MCC at 0.5 | AUPRC | Zero-contact-chain residue FPR at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neither fix | 0.563121 | 0.585009 | 0.574649 | 0.605310 | 0.005563 |
| GVP boundary | 0.564845 | 0.585528 | 0.575173 | 0.606271 | 0.005442 |
| Co-attention positions | 0.561529 | 0.577527 | 0.569768 | 0.588355 | 0.007672 |
| Both | 0.561801 | 0.578005 | 0.570184 | 0.588849 | 0.007543 |

GVP repair gives a small frozen gain: Top20 F1 +0.001724 and AUPRC +0.000960.
Position repair immediately lowers AUPRC by0.016955 in these old weights.
That is a frozen distribution-change result, not proof that training with
correct positions is ineffective. Neither frozen intervention solves the
context gap. No significance claim is made.

The graph audit counts3540 directed artificial chain-boundary backbone
bridges;3470 (98.02%) exceed the15 A spatial cutoff. This supports the existence
of the graph defect, but its removal has only a small immediate accuracy effect.

Chain-order reversal is a stronger diagnostic lead. All residue-aligned inputs
are permuted together and predictions are inverse-permuted before comparison.
Geometry, ligand, labels, and prediction budget remain unchanged. In baseline,
mean Top20 Jaccard is0.142699 for multichain rows versus0.995238 for single-chain
identity controls; mean absolute probability changes are0.053807 and0.000232.
Neither fix restores invariance (multichain Jaccard0.138-0.143).

These are sample MEANS, not medians. BF16 identity controls are close, not
exactly identical. Chain-gap RoPE is not mathematically permutation-invariant.
Before attributing the HOLO4K gap to chain order, independently verify FP32
identity/permutation alignment and canonical-order controls on Star. This is
a future diagnostic, not an experiment launched in this acceptance pass.

## Migration receipt and remaining work

Star archive: `/data1/PLC/archive/hpc_acceptance_20261106`.

- Copied five best-F1 checkpoints: legacy G0, failed HPC ESM-only, and the
  current best snapshot of each of the three running controls.
- Copied isolated fixed-model source, orchestration, prepared configurations,
  provenance, GPU smoke results, and all1050 frozen-VAL prediction files.
- The first1,465,695,968-byte artifact set passed a content-checking
  `rsync -anc --itemize-changes` dry run: exit0, no differing-file records.
- Copied an additional753,595,833 bytes of training logs/configs/TensorBoard.
  These are live snapshots; logs from running jobs will continue to grow.
- Local raw text receipts and their SHA256 fields were checked locally and
  copied to `local_receipts/` on Star. No source artifacts were deleted.

Existing Star core data/runtime remain under
`/data1/PLC/experiments/chainwise_esm_core_20261102`. No ESM extraction weights
are needed for cached-feature core training. Archive configs still contain HPC
paths: they are provenance artifacts, NOT portable launch configs. Any future
Star experiment must use a separate prepared root with relocated paths and
verified existing Star feature caches, without editing job2843's frozen code.

Star2843 was still running after60h43m with a72h allocation, leaving about11h17m
at the final queue check. It may reach wall time before convergence. No second
Star task was submitted and no runtime limit or training behavior was changed.
Star had about483GB free at the disk check; no full data mirror was duplicated.

Final acceptance is still pending for the three allowed HPC controls. Once
they finish, collect final best-F1 checkpoints/configs/TB/logs, verify them on
Star, and close HPC use. If a permitted job fails, preserve the failure and
available checkpoints; any recovery or follow-up runs must be on Star.

Machine-readable snapshot:
[`results/hpc_acceptance_star_cutover_20261106.json`](results/hpc_acceptance_star_cutover_20261106.json).
This report does not claim the running training jobs are complete.
