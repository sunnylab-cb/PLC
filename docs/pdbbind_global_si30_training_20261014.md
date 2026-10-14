# PDBbind-only global-SI split and scratch GFGF training

## Approved scope

The user approved the protocol change after the combined global/local edge
rule collapsed the QC pool into one connected component. The new guarantee
is **global chain SI <0.30 across each of TRAIN/VAL, TRAIN/TEST and VAL/TEST**.
The affine-gap BLOSUM62 alignment policy, original-letter identity, terminal
gap penalty and gap-inclusive denominator are unchanged. Local SI is still
computed and reported, but does not create graph edges or fail this audit.
This does NOT claim local/domain independence or pretraining decontamination.

Keep all 19,018 QC-passing PDBbind complexes, including every observed input
chain. Keep same-PDB/exact-sequence membership and entire connected components
together, targeting 80/10/10 with split seed42. No COACH420/HOLO4K exclusion.
The joint branch is not edited, cancelled, resubmitted or advanced today.

Labels remain native heavy-atom center distance **d_min <5.0 A**, with at least
three contact residues, also within the existing 1,800-residue training cap.
No vdW rule, 5.5 A threshold or contact-rule change is introduced.

## Fail-closed pipeline

1. Validate the completed 44,476,596-pair cache: pool, scoring/version contract,
   original source hash, all 2,224 chunk hashes/counts and ordered edge indices.
2. Filter global-threshold edges from that combined-rule superset, without
   overwriting any source artifact; construct connected components anew.
3. Recompute ALL cross-set chain pairs independently. Save global SI maxima
   per set pair, full pair counts and covered-local-only diagnostic counts.
4. After zero global violations, fit new TRAIN-only ligand/spatial statistics
   and materialize this dataset independently of the joint/old datasets.
5. Check exact production-loader coverage, label protocol and residue-axis
   loading. Freeze config/source/asset hashes before the GPU stage.
6. Run FLA/reference kernel checks and the production BF16 training loss and
   backward on real TRAIN rows; check FP16 inference on real VAL rows. Discard
   smoke weights. TEST is not forwarded or used for model/gate selection.
7. Launch seed44 training only after successful preparation and matching smoke.

Slurm chain: CPU `i64m512ue` (64 CPUs/160G), then single-GPU smoke and training
on `i64m1tga800ue` (8 CPUs/100G, one GPU per job). Both dependency links use
`afterok` with invalid dependencies killed, not ignored. No GPU allocation
is used while CPU preprocessing is running. Queue submission is idempotent.

Root: `/hpc2hdd/home/shuang886/PLC/experiments/pdbbind_global_si30_gfgf_20261014`.
Source cache: the `pdbbind_only/` subtree of
`internal_si30_dual_fixed5_20261013`. The latter remains read-only.
Deployed 512 tracked source/config/test files from commit `f72ed65`, verified
individually against SHA256 on HPC. Existing FLA0.4.1/einops dependencies are
reused from the original GFGF experiment. No dirty production-checkout pull.

Submitted October 14 at 08:06:38 UTC (16:06:38 UTC+8):

| Stage | Job | Observed state | Partition |
| --- | --- | --- | --- |
| SI audit + preprocessing + config | 13036932 | RUNNING, cpu1-26 | i64m512ue |
| GPU smoke | 13036933 | PENDING, afterok:13036932 | i64m1tga800ue |
| Scratch seed44 training | 13036934 | PENDING, afterok:13036933 | i64m1tga800ue |

The source-cache integrity checks and connected-component assignment completed.
The new candidate split has **15,214 TRAIN / 1,902 VAL / 1,902 TEST** samples,
all PDBbind. Fresh cross-set alignment is running, so this is not yet a
completed SI certificate. The initial log has no error; no training epoch or
new F1 has been produced. `submission.json`, `deployment.json`, logs, source
hashes and `split/all_pairs_reuse.json` are retained in the experiment root.

## Exact training recipe

Use the verified **V11m13 group_rescompat + GFGF** reference configuration,
SHA256 `c58d7a8d4dd1a1840232e5cda9b648b62085a93da7dc14c00753e3ac5944fa69`.
Only data root, split file, label file, experiment name and output root change.
All model and training settings are compared for equality.

- Seed44, scratch, fully unfrozen; no resume, initialization checkpoint or compile.
- Co-attention and local54 group/residue-chem compatibility, protein grad scale .6.
- GFGF bidirectional chain-reset GDN/full-attention protein layers, FLA backend.
- Batch16/BF16, AdamW 1e-4, unchanged loss/CNC, up to120 epochs, patience30.
- Existing max_atoms256 with alternating-ends crop, max_seq_len1800.
- Existing best-F1 and AUPRC/AUROC checkpoints and VAL-only gate metrics retained.

The strongest old-split test result motivates the recipe, not a comparable new
score: these holdouts and normalization statistics differ. No new F1 exists yet.

## Verification

41 focused local tests passed: global-only threshold/short-chain behavior,
superset cache reuse and corruption rejection, fresh cross-set diagnostics,
whole-component splitting, fixed5 labels, TRAIN-only normalization, tiny-cohort
production-loader end-to-end, and exact scratch-config safeguards.
Both new Slurm scripts pass `bash -n`. GPU validation remains a remote job gate.
An additional 29 regression tests passed for the existing trainer/probe,
versioned-label evaluation compatibility and validation probability metrics
(70 total). Local optional CUDA-extension warnings do not certify GPU support;
the actual HPC GPU smoke is required before training.

Entry points:
`run_internal_pairwise_split.py --edge-rule global_only --reuse-all-pairs ...`,
`prepare_internal_gfgf.py`, `smoke_internal_gfgf.py`, `submit_internal_gfgf.py`,
`run_pdbbind_global_si30_prepare.slurm`, `run_pdbbind_global_si30_gfgf.slurm`.
