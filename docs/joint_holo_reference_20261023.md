# Latest Joint Epoch25: HOLO4K Reference Evaluation

## External Isolation: Neither Benchmark Was Excluded

The expanded joint PDBbind + BioLiP cohort guarantees its own three-way
TRAIN/VAL/TEST **global chain SI<0.30**, with whole connected components and
a fresh cross-set audit. It does **not** guarantee isolation from COACH420 or
HOLO4K. Neither external dataset participated in this split. This was the
requested internal-only expansion protocol, not a claim that COACH remained
excluded while HOLO was included.

Evidence: [frozen construction protocol](joint_global_si30_training_20261017.md),
the corresponding preparation launcher, and the exact split manifest locked
by the internal epoch25 evaluation. No old external-clean certificate can be
inherited by this independently rebuilt cohort.

The new accession audit finds direct PDB reuse:

| External set | Split compared | Shared-PDB protein keys / all keys | Associated ligand rows / all rows |
| --- | --- | ---: | ---: |
| COACH420 | TRAIN |122 /299|164 /494|
| COACH420 | VAL |42 /299|70 /494|
| COACH420 | TEST |29 /299|44 /494|
| HOLO4K | TRAIN |1665 /3430|3458 /7245|
| HOLO4K | VAL |231 /3430|565 /7245|
| HOLO4K | TEST |272 /3430|657 /7245|

Counts use the full requested benchmark populations before the seven HOLO
LiBRe chemical-input exclusions. A shared PDB accession does not prove an
identical chain, ligand, pocket or full training pair. Conversely, absence
of a shared PDB does not certify low SI. These are accession-overlap counts,
not a complete sequence-identity audit. TEST overlap is not training exposure;
TRAIN overlap and VAL-based checkpoint selection require separate reporting.
The unaltered [audit JSON](results/joint_holo_external_overlap_20261023.json)
contains exact definitions and example protein keys.

Consequently this run is an **external-overlap-uncertified reference
comparison**, not a clean out-of-distribution/SI30 generalization claim.
It must not be used to select checkpoints, thresholds or model changes.

## Fixed Evaluation

| Item | Setting |
| --- | --- |
| Checkpoint | Joint V11m13 group_rescompat +GFGF, scratch seed44, VAL-best epoch25 |
| Checkpoint SHA256 |234ba442325a6fb512ba54670ca233006e1004999f95322ae24c5a61f65dad05|
| PLC gate | Existing VAL-selected0.5535550117492676 |
| PLC inference | BF16,batch2, full protein-key axis capped1800, uncapped ligand atoms(-1) |
| Labels | Native heavy-atom d_min<5 A, per ligand; no relabeling |
| Normalization | Joint TRAIN-only train_only_stats.json, no fitting on benchmarks |
| Stats SHA256 |bc18fbaf2617efbec485fc71f88aa1364c6ec043acd4c433047bfff82902aeb3|
| LiBRe | Reuse frozen benchmark12960610/holo4k predictions and official p>0.6 |
| Coverage | PLC requests7245 rows; matched comparison uses7238 prior LiBRe successes |
| Primary ranking | Prespecified Top20, not benchmark-selected K |
| Other controls | Top10/40/80,ceil(L/20),ceil(L/10),ceil(L/5),common p>=0.5 |
| Metrics | Pooled/sample-macro F1,MCC,P,R,AP/AUROC; protein-balanced and protein-best F1 |

The latest model's normalization differs from the old benchmark model.
An optional evaluator path reads raw89D spatial features and applies the
frozen joint TRAIN mean/std, with std<1e-8 replaced by1, then the original
clamp on dimensions62,76,78,79 to[-5,5]. Selected109D ligand dimensions use
the joint TRAIN z-score recipe. Existing ESM2 and physchem representations
are unchanged. Cached old spatial z-scores are not normalized a second time.
Missing raw features and nonfinite values fail closed, except for the
previously audited, explicit SASA-only ligand repair policy.

Defaults for legacy benchmark callers are unchanged. No model/backbone
source is modified. Deployment verified68 model/data source hashes against
the successful internal reference run. CPU QC precedes GPU inference; matched
comparison rechecks per-row residue keys, coordinates, labels, identities and
native prediction/CSV agreement against the cached LiBRe inputs.

## Submission

Submitted **2026-10-23 11:27 UTC+8**:

| Stage | Job | Partition | Resources | Initial checked state |
| --- | --- | --- | --- | --- |
| CPU preflight |13279553|i64m512ue|8 CPUs,100G,4h|RUNNING,cpu1-35|
| GPU inference/comparison |13279554|i64m1tga800ue|1 GPU,8 CPUs,100G,12h|PENDING(Dependency)|

The GPU job uses afterok:13279553 and kill-on-invalid-dep=yes. Time limits
are scheduler limits, not measured ETAs. No other run was cancelled.

Remote root:

```text
/hpc2hdd/home/shuang886/PLC/experiments/joint_holo_reference_20261023
```

Immutable snapshot code_v1, commit8119966; source archive SHA256:
80baefafb45f7a520642372946dbbbfa997cb0636a15dfde97bd8667766499c4.
Runner:scripts/benchmark/eval_joint_holo_reference.py.
Launcher:scripts/slurm/run_joint_holo_reference.slurm.
50 local targeted tests passed; Python and shell syntax checks passed.

Retain protocol.json,external_overlap.json,prepared.json,preflight/,
inference/,matched_prediction_audit.json,matched_summary.json and logs/.
At submission there is no new HOLO4K model score. The earlier0.3224 PLC
Top20 sample F1 is the old October10 checkpoint, not this expanded joint model.

First live check: CPU preparation is RUNNING and has reached472/3623
preflight batches (about13%,batch2). Logs confirm56 ligand-normalized
dimensions from the joint TRAIN statistics and the expected spatial clamp.
The GPU job remains PENDING(Dependency); preflight has not yet passed and
there are no new inference results.
