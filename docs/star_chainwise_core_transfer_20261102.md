# Star Core Training Bundle, No ESM Weights

## Scope

User requested filling Star's missing current training data, then explicitly
excluded ESM weights. This migration copies cached features only, not ESM2/3/C
model weights, trained PLC checkpoints, raw PDBs, old outputs or extra NMA confs.
It neither cancels HPC training nor starts duplicate training on Star.

Destination: `/data1/PLC/experiments/chainwise_esm_core_20261102`.
The old `/data1/PLC/data` and its runs remain untouched. Core input is the full
complex, with independently encoded chains reassembled into its residue axis.

## Inputs

- Same strict joint cohort: TRAIN139,158 / VAL17,395 / TEST17,395.
- Source normalized conf0 and ligandV5 from
  `experiments/joint_global_si30_gfgf_20261017/dataset` on HPC.
- ESM2 from `experiments/chainwise_esm_fullcomplex_20261101/esm2_latents`.
  The10,090 changed multichain caches must finish before the bundle is ready.
- Original5A labels, TRAIN-only stats, split membership, SI audit and source
  provenance. No normalization is recomputed and no old Star stats are reused.
- Frozen original G0 model/trainer snapshot, current thin training wrapper,
  FLA0.4.1/einops dependency snapshot. Source recipe and RNG seed unchanged.

## Two CPU Transfer Stages

1. Base: copy code/metadata, conf0, ligandV5, and163,858 unchanged ESM caches.
2. Final: depends on base and successful ESM extraction13468090. Copy all ESM
   caches incrementally, then perform sequential rsync checksum dry runs for
   all three feature families. Require no feature-file differences. Verify
   exact sample coverage, sizes and local regular-file paths on Star before
   writing the ready marker.

Transfers run directly HPC -> Star, one rsync connection at a time. `-L` resolves
HPC symlinks; `-H` preserves shared target inodes where applicable. No `--delete`.
No broad source-root copy and no external-baseline/checkpoint paths are included.
The original extraction manifest retains historical weight hashes as provenance,
but the portable training verifier never opens those weights or HPC-only paths.

Core verifier checks frozen labels/stats/config/code and all521,844 feature-file
entries. ESM feature availability and extraction completion are separate from
ESM weight availability. Partial transfers cannot pass the verifier.

## Runtime

Observed Star environment: `gpsite_full`, PyTorch2.6.0+cu124, PyG2.8.0,
Triton3.2.0, PyYAML6.0.3, TensorBoard2.21.0. The FLA runtime still needs a real
single-GPU smoke test before claiming GPU compatibility; no GPU is used for sync.

After data readiness, `scripts/run/run_chainwise_core_star.sh GPU_ID` performs
the required100-step smoke and then scratch training. It accepts exactly one
GPU index, holds a training lock, and enables offline mode. It is NOT launched
by either transfer job. Do not start it while an existing Star task occupies
the permitted training slot, or create a duplicate of HPC without deciding the
placement first.

## Submission

Submitted on2026-11-02, using immutable transfer code commit`278a67a`:

| Stage | Job | Partition | Last observed status |
| --- | --- | --- | --- |
| Base inventory and available cached inputs |13477555|i64m512u|RUNNING oncpu1-54; protein inventory173,948 verified |
| Final ESM fill and checksum/coverage verification |13477556|i64m512u|PENDING, afterok:13477555:13468090 |

Both jobs request one CPU node,2CPUs and16GiB, no GPU. The initial status above
is not a transfer-complete or training-ready claim. ESM extraction13468090
is still PENDING(Priority); original HPC frozenVAL/scratch training jobs
13468091/13468092 remain PENDING(Dependency), unchanged.

HPC transfer ledger, plans, phase status and logs:
`/hpc2hdd/home/shuang886/PLC/experiments/chainwise_core_star_export_20261102`.
The deployed archive SHA256 is
`a0026f8853621662c61c84f3b256b5e5f31d016739c9b96bc761d57001afd1fe`.
Core-bundle and existing chain/context tests:44passed.

## Completion and Star GPU Launch (2026-11-03)

- Base13477555 completed2026-11-02 19:55:52; final13477556 completed
  2026-11-03 02:09:07, both exit0. Star receipt has checksum_verified=true,
  and provenance has complete=true for173,948 samples. No weights transferred.
- Future transfers must NOT use CPU/Slurm jobs; use direct rsync/scp with
  logged background execution, as recorded in AGENTS.md. The two old transfer
  jobs were allowed to finish, not replaced.
- Star now enforces GPU access through Slurm. The manual GPU-index launcher
  above is not the current launch path; use
  `scripts/slurm/run_chainwise_core_star.slurm`, preserving Slurm device mapping.
- User authorized critical smoke followed by core training. Smoke2839 uses
  account labcritical, QoS critical, one GPU,8CPUs,64GiB,1hour. CUDA initialization
  succeeded on an otherwise idle A100-SXM4-80GB; smoke result still pending at
  this record's creation.
- Train2840 uses account lab, QoS background, one GPU,8CPUs,64GiB,72hours,
  afterok:2839 and kill-on-invalid-dep. No automatic requeue or resume; this is
  an independent seed44 scratch run with the original full-complex G0 recipe.
  HPC's existing training is not cancelled. Treat this as same-seed cross-host
  replication, not another independent seed for statistics.
- Smoke checks100 TRAIN steps, gradients via CheckedAdamW, cap1800 stress,
  finite VAL F1/AUPRC/MCC and strict state-dict reload. TEST is not forwarded.
  Smoke parameters are discarded before the independent training process.
- Logs are `logs/star_smoke_2839.{out,err}` and
  `logs/star_train_2840.{out,err}` under the Star bundle root. The launch script
  is deployed outside the hash-frozen code trees, under `launch/`.
- A72hour time limit is not a convergence guarantee. If training reaches this
  limit or is preempted, inspect saved results before deciding how to continue;
  no automatic scratch restart is enabled.

### Runtime Retry

Smoke2839 failed in FLA0.4.1 `prepare_wy_repr_bwd_kernel` with Triton3.2.0:
`arith.mulf op requires the same encoding for all operands and results`, then
`PassManager::run failed`. This was a compiler failure, not OOM; train2840 was
automatically cancelled without running. HPC uses torch2.9.1/Triton3.5.1.

Installed Triton3.5.0 with `pip --no-deps --target` into bundle
`runtime/triton350`, without changing shared gpsite_full or frozen model/deps.
The launcher prepends this isolated runtime and checks/prints its version.
Torch remains2.6.0+cu124; this is not an exact HPC environment clone and must
pass the full GPU smoke before use. Retry smoke2841 uses critical and one GPU.
Upstream FLA's current CUDA requirements also call for newer Triton:
https://github.com/fla-org/flash-linear-attention/blob/main/pyproject.toml
That moving upstream requirement is context, not proof of this pinned runtime's
compatibility; the actual smoke result is decisive.

Retry2841 also failed before completing a step: torch2.6's Inductor import
expects `AttrsDescriptor`, removed in Triton3.5. The overlay is therefore NOT a
working runtime and is no longer on the launcher PYTHONPATH. No train retry was
submitted for this failed smoke.

Preparing `runtime/torch291` as a separate venv with shared non-torch Python
packages and its own official torch2.9.1+cu126/Triton3.5.1 pair. The launcher
checks exact versions and removes inherited PYTHONPATH to prevent the failed
overlay leaking in. The frozen model/data/source verification remains unchanged.

Installed matched runtime successfully: torch2.9.1+cu126, Triton3.5.1,
pyg_lib0.6.0+pt29cu126, torch_scatter2.1.2+pt29cu126,
torch_cluster1.6.3+pt29cu126; torch-geometric2.8.0 remains inherited.
torchvision0.24.1+cu126 and torchaudio2.9.1+cu126 were also installed in the
venv to remove inherited torch2.6 pins. Core CPU imports pass; smoke2842
successfully initializes CUDA on the A100. No public environment was changed.
`pip check` now only reports inherited hatch-cython build-tool dependencies
(cython/hatch/hatchling), not training-runtime conflicts; these are not used by
the already-built wheel training path and were left untouched.

### Passed Smoke and Training

- Smoke2842 COMPLETED, exit0, elapsed6m24s. Report:
  `smoke/G0_chainwise_esm_complex.json` in the Star bundle, passed=true.
- All100 TRAIN updates passed; ordinary batch shape16x420, profile141.66s
  including first-use overhead, peak allocated3.85GiB. Cached-batch throughput
  is11.29pairs/s and is NOT a full-epoch throughput or ETA.
- Cap1800 stress batch passed, peak allocated31.57GiB. VAL finite F1/AUPRC/MCC
  checks and strict CPU state-dict reload passed; TEST not used.
- Formal scratch training2843 submitted with afterok:2842, background QoS,
  account lab, one A100 GPU,8CPUs,64GiB,72hour limit, no requeue/resume.
  Log paths: `logs/star_train_2843.out` and `logs/star_train_2843.err`.
- Successful runtime is the isolated torch291 venv, NOT gpsite_full directly
  or the failed triton350 overlay. Smoke weights are discarded; model code,
  data, normalization, split and seed44 training recipe remain unchanged.
