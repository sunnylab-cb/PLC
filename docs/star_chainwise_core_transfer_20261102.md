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
Core-bundle and existing chain/context tests:44passed. GPU smoke has NOT run.
