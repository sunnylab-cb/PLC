# Strict parallel site-head trials: implementation and submission

Date: 2026-09-26. User authorized implementation/submission after the parallel
experiment proposal. This is an execution ledger, not a performance report.

Follow-up checked 2026-10-04: diagnostic, CPU audit and all-branch smoke completed.
Full export failed on the 5ndf reference probability check; the full head array
was cancelled without running. See `strict_site_parallel_results_20261004.md`.
The submission-time states below are retained as historical observations.

## Job ledger

| Job | Role | Verified state at submission follow-up |
| --- | --- | --- |
| 12868960 | Repeated BF16/FP32 forward and intervention diagnosis | PENDING, Priority; -ue one GPU, reduced to 1 CPU/16G/5 minutes for backfill |
| 12869006 | TRAIN-only multisite union audit | RUNNING on cpu1-104, emergency_cpu, 4 CPUs/32G; 9000/10317 QC rows observed |
| 12869013 | Small real-data cache and ALL head/control smoke | PENDING, afterok:12868960; -ue one GPU/4 CPUs/64G/30 minutes |
| 12869014 | Full shared TRAIN/VAL token export | PENDING, afterok:12869013; -ue one GPU/4 CPUs/64G/12-hour cap |
| 12869016_[0-3%3] | Four head/control groups | PENDING, afterok:12869014; one GPU EACH, concurrency cap 3, 4-hour cap/task |

All dependent jobs have `KillOInInvalidDependent=Yes`, verified via scontrol.
Scheduling a dependency-protected task is NOT evidence that its training began.
Latest observed diagnostic start estimate: 2026-09-27 06:01 scheduler time;
earlier estimates were later. Estimates are not reservations or guarantees.

Star's occupancy check failed with `Failed to initialize NVML: Unknown Error`.
No task or data transfer to Star was attempted after that failed check. The
local environment also had no `nvidia-smi` binary. No unverified GPU was used.

The older 12868601 smoke failure and 12868608 cancellation remain preserved.
Its root cause is not yet established; the new GPU diagnostic has not run.

## Implemented trial groups

| Array index | Branch | Sequential controls |
| --- | --- | --- |
| 0 | Original candidate quality pilot | stats_only, protein_site, ligand_site |
| 1 | Original-relative gain | signed gain vs capacity-matched paired-feature absolute quality |
| 2 | Token late interaction | token matching vs matched mean matching vs ligand-blind matching |
| 3 | Boundary residual | ligand-aware refiner vs parameter-matched protein-only refiner |

All use the audited strict merged TRAIN 10317 / VAL 647, frozen base44, new heads
from scratch, seed44, existing normalization and truncation budgets. No resume,
new backbone training, test/COACH/HOLO use, native cross-molecule distances or
torch.compile. Val selection is protein-balanced F1 at a fixed .02 predicted
quality/gain replacement margin, with the original output always an option.
The original contact gate is retained (read from the exported checkpoint).

The gain branch uses shared candidate/original encoders plus their differences,
signed-gain SmoothL1 and a small sign-ranking term excluding near ties <=0.02.
The paired absolute-quality control has the same parameter count and inputs.

The token branch uses 64D normalized atom/residue projections and bidirectional
length-normalized soft similarity. Algebraic membership aggregation avoids a
candidate-by-residue-by-atom tensor. All three token controls have equal parameter
counts. Tokens are saved once per pair; candidate masks are packed indices.

The boundary branch has signed, zero-initialized, bounded +/-0.5 contact-log-odds
residuals, restricted to observed candidate residues and a protein-CA 4 A shell.
Original candidate zero is never changed. In a 20-epoch pilot, the refiner trains
for 10 fixed TRAIN epochs, then freezes; a separate quality head trains for 10
epochs on the actual refined-mask TRAIN F1. Refined VAL GT is never used to
choose a mask or a refiner checkpoint. Uniform TRAIN subsampling of up to 16
non-original candidates limits work; inference processes all candidates in chunks.
Refined mask indices are saved for reproducibility and future analysis.

Every new branch saves real/permuted/zero added-query VAL diagnostics. For the
boundary branch this recomputes BOTH refinement and selection. The frozen base
and original proposals remain real-pair outputs: these are head-only controls,
NOT full-model ligand-dependency results.

## Independence check and safety

The previous assertion compared two BF16 forwards to absolute 1e-5 without an
identical-input noise reference. The new exporter measures BF16 repeat and
intervention errors separately. It requires FP32 identical-input repeatability
AND both independent-state interventions to pass the SAME absolute 1e-5 bound,
plus finite responsive changes and no mutation of input tensors. It rejects
any module left in training mode and writes an audit JSON even when a numerical
independence check fails. We have NOT simply raised a BF16 tolerance.

This is a changed audit protocol, not a demonstrated diagnosis. If strict FP32
fails, the smoke stops and no full training runs. The separate diagnostic also
captures GVP/protein-context/optional ligand-self-attention outputs to localize
repeat differences. Model architecture and checkpoint weights are unchanged.

Full cache additionally verifies split/checkpoint/overlay hashes, reproduces the
audited VAL baseline within existing thresholds, checks finite tensors and full
populations, and hashes every row. Only a completed manifest is accepted by the
training jobs. Original source labels/features and the old result archives are
not modified. Retained quality-head checkpoints are best-VAL only.

## CPU audit

Complete-link geometry clusters require every aligned pair to pass max CA error
<=0.05 A; this avoids transitive geometry drift. Heuristic site groups also use
complete-link checks (Jaccard >=0.5 and both-frame centroid separation <8 A).
Unknown chemical identity remains unknown. The output includes known-positive
union indices and coverage, not certified negatives or an approved new dataset.
Site count is heuristic, not ground truth for biological distinctness.

## Deployment and verification

Remote root:
`/hpc2hdd/home/shuang886/PLC/experiments/strict_site_parallel_20260926`.

Standalone diagnosis/CPU audit use `code/`. GPU smoke, export and heads use the
separate frozen `code_pilots_v1/` snapshot. Shared full cache, if successful:
`run_12869014_none/cache`. Head outputs use
`run_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/{baseline,gain,tokens,boundary}`.
Logs are in `logs/`. No files go to jhspoolers; no git push was needed.

62 local tests passed, covering all three end-to-end synthetic training paths,
matched parameter counts, token permutation invariance, blind controls, finite
gradients, boundary support/amplitude, original fallback, target-free inference,
strict-state audit rejection, complete-link union behavior, and earlier label/
benchmark integrity checks. Python compilation and Slurm shell syntax passed.
Optional local PyG extension warnings remain; actual HPC CUDA smoke is pending.

Deployed file hashes matched local versions; the optional-module guard in the
standalone diagnostic was additionally synced and verified while still pending.
The `code_pilots_v1` training snapshot has not been edited since submission.

| File | SHA256 |
| --- | --- |
| export_site_quality_cache.py | 4adf95603dfaafe0befda5d815ce5d309fda751184f5874f6276c75228c3b76a |
| site_quality_features.py | f6fbf25bcdce4016203d5f48052afa8bec9e75975716bb0aaa89c46e4069f399 |
| train_site_quality_probe.py | 74c73ab4e92151d207ce19025d611c672b5c65908625c35fe74e4427209ebc18 |
| train_site_parallel.py | 434b7d22ac0e4ce54e3f6997f3f34d76426b245c5fc6bdc1ac56f410f5da650b |
| site_parallel_heads.py | 551dafc1db83d4c5d51504a6b26c120b4e14a9627fd08a4209a58ca6607300c9 |
| audit_train_multisite_union.py | d335b656c521e95cc701297b39a241a7713db2961c23042264fe31aa1658a371 |
| run_strict_site_parallel.slurm | da8855e54aa89d5cba3c6d957ca53cdd7005f48f7f1026628df88eda4e254882 |
| diagnose_site_state_independence.py (standalone) | 965f9abc71c4b104a5fe4f172d46807bcea05a38ab49e25142c1fc81a47e3f19 |

## Outstanding work

Updated 2026-10-04: investigate the 5ndf full-export probability mismatch before
restarting. Labels, coordinates and binary outputs agree; the maximum probability
error is 0.05126017 versus the existing 0.05 bound. The numerical cause on this
long protein remains unconfirmed. Do not silently relax the check or use the
partial cache without a complete manifest. The full head array did not run.
Review the complete TRAIN/VAL results before seed expansion or external testing.
