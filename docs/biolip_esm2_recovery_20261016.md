# BioLiP ESM2 Recovery, October 16

## Scope and Evidence

The user requested broader ESM2 coverage. The first expansion produced
105,684/111,061 chain outputs (95.2%). Its5,377 failures comprise:

| Failure | Chain keys | Interpretation |
| --- | ---: | --- |
| Missing V4 file | 5,082 | Mostly a preparation dependency, not absent sequences |
| Chain absent from V4 | 250 | Requires checking the actual CIF chain identifiers |
| Duplicate residue keys | 45 | Ambiguous axes must not be deduplicated arbitrarily |

The83 missing V4 proteins exceeded the legacy temporary-PDB62-chain limit.
Read-only CIF inspection confirms that5y6p chain14 still has227 valid amino-acid
residues with unique keys despite its862-chain assembly;7va9 has64 chains and
a valid54-residue chainA. Neither requires a PDB conversion to run ESM2.

Other failures are not simple compute retries. For example,6j3z chain19 is
absent from both current fixed and original receptor CIFs. The fixed8i0r chainA
contains3,498 residues and repeated residue numbers with different identities
and spatial coordinates. Selecting the first duplicate would silently change
the protein. The alternate shared_receptors2 directory inspected here contains
pair subdirectories, not the expected full receptor CIF at its root.

## Recovery Contract

1. Reuse the sealed expansion candidate/chain manifests and failed-chain
   summaries; do not broaden chemistry eligibility or touch old split labels.
2. Parse the first CIF model directly. Prefer fixed receptors, then the existing
   original receptor if a chain cannot be resolved. Retain exact chain IDs,
   residue numbers, insertion codes, amino-acid identities and CA coordinates.
   Use the same canonical-residue policy as V4, not a broader modified-AA map.
3. Reject missing chains, duplicate chain identifiers, duplicate residue keys,
   nonfinite coordinates and chains without V4-compatible amino acids.
4. If V4 exists, check ordered keys, identities and CA coordinates explicitly.
   Otherwise mark V4 alignment pending. Recovery does not generate V4 features.
5. Use the same trusted local ESM2-650M checkpoint, final layer33 and existing
   overlapping-window inference. Reuse existing exact-sequence latents read-only;
   new outputs have checked dimensions, finite values and checkpoint provenance.
6. Store all outputs in an isolated recovery namespace. No automatic overlay,
   zero fill, original-cache overwrite, normalization refit or training admission.

This separates **sequence-cache coverage** from **usable training-pair coverage**.
Even successfully recovered ESM files cannot admit the affected pairs until V4
extraction/axis repair, native5A contact QC, deduplication and new joint SI pass.
No recovery success count is claimed before the submitted jobs produce it.

## Entry Points

- `scripts/diagnostics/biolip_esm2_recovery.py`: CPU audit (`--stage plan`) and
  GPU inference (`--stage extract`), with hash-sealed manifests and resumability.
- `scripts/slurm/run_biolip_esm2_recovery_cpu.slurm`:4 CPUs/128G on `i64m512ue`.
- `scripts/slurm/run_biolip_esm2_recovery_gpu.slurm`:one GPU/8 CPUs/80G on
  `i64m1tga800ue`; `LIMIT=3` for smoke, unset for full extraction.
- Run order: CPU audit -> GPU smoke -> full GPU recovery, all `afterok` gated.

Both Slurm scripts pass `bash -n`. The recovery, expansion, chain-alias,
pairwise-split and geometry/loader tests pass **58 tests** locally. Warnings are
from optional local hbond/PyG extensions and upstream deprecations; no failure.
Existing ligand extraction13077843 and final QC13077847 are not modified.

## Deployment

Source revision **48e23bd** is pushed to `plc/master`. A hash-checked immutable
snapshot, receipt, manifests, logs and new features reside under:

`/hpc2hdd/home/shuang886/PLC/experiments/biolip_feature_expansion_20261015/esm2_recovery_20261016/`

| Stage | Job | Status checked October 16, 17:32 UTC+8 |
| --- | --- | --- |
| Full failed-chain CIF-axis audit | 13124029 | PENDING, Priority |
| Three-chain GPU smoke | 13124030 | PENDING, afterok13124029 |
| All sequence-ready failed chains | 13124031 | PENDING, afterok13124030 |

The source archive SHA256 is
`f5aaa406317e9ee3e43cabc82ea2ac4f17098bc6f2aecdf3613199517a98b004`.
The remote runner and both Slurm scripts were independently hash-matched before
submission. A subsequent `squeue --start` check shows the audit waiting for
Resources, with a provisional start at **October17 00:57:01 UTC+8**; this is a
scheduler estimate, not a guaranteed launch time. No GPU inference or successful
recovery is claimed yet. Existing expansion/QC dependencies are unchanged.

## Completion Verified October 17

The preceding queue snapshot is historical. Accounting and output summaries
now confirm all three jobs COMPLETED with exit0:0:

| Stage | Job | Actual start/end on October16, UTC+8 | Elapsed |
| --- | --- | --- | --- |
| CIF-axis audit | 13124029 | 17:42:48 / 17:43:54 | 1m06s |
| GPU smoke | 13124030 | 17:43:54 / 17:44:17 | 23s |
| Full recovery | 13124031 | 17:44:18 / 17:45:26 | 1m08s |

The audit resolved **5,079/5,377** requested chain axes, all from fixed CIFs;
**298 remain unresolved**. Resolved axes correspond to14,143 candidate pair
rows, unresolved axes to440. The GPU stage succeeded on **5,079/5,079**, with
3 outputs resumed from smoke,4,963 exact-sequence cache hits and113 newly
computed unique sequences. This explains the short GPU runtime without any
zero filling or sequence truncation.

Together with105,684 existing outputs, scheduled-chain cache coverage is
**110,763/111,061 (99.73%)**. The sealed axis manifest SHA256 is
`af223d2ae1822a0d682a795a29d8a81eb97685be0a497e506956e4d92ec073ad`.

These are **sequence-cache successes, not training admission**. Recovery
outputs remain in the separate `esm2_features/` directory under the recovery
root; no overlay was promoted. V4 feature/axis recovery and geometry/chemical
QC are still required for the affected pairs. The original ligand extraction
and QC jobs are unchanged.
