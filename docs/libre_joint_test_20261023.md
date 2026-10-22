# Released LiBRe Versus Latest PLC Joint TEST

## User-Approved Scope

On 2026-10-23 the user explicitly clarified: evaluate the released LiBRe
checkpoint on our latest locked TEST, regardless of historical supervised
training overlap. The earlier feasibility note is not a reason to block this
comparison. No overlap-based exclusion or retraining is requested.

The separate question about LiBRe's own split is answered by its paper:
training-set CD-HIT90 redundancy reduction, exclusion of shared train/test
PDB IDs, and a Foldseek structural-similarity filter >0.4. These do not
constitute a reported train/test pairwise SI<0.30 guarantee. See the
[source audit](libre_joint_test_feasibility_20261023.md).

This experiment compares released-model inference, not equal-training-data
architecture generalization. That qualification is metadata, not a blocking
gate. TEST remains reference-only and cannot drive subsequent PLC iteration.

## Fixed Comparison Protocol

| Item | Locked setting |
| --- | --- |
| Population | Latest joint TEST:17,395 rows,1,649 PDBbind +15,746 BioLiP |
| PLC | V11m13 group_rescompat +GFGF, scratch seed44, VAL-best F1 epoch25 |
| PLC inference | Reuse saved real predictions from13244903; no rerun |
| PLC threshold | Existing VAL gate0.5535550117492676 |
| LiBRe | Official revision e9151a85bab2d05b67b953bc55a74c780aa2bf24 |
| LiBRe weights | Released model.pt; SHA256 ba40bdf78ba4e596e7429e56d2c955c42b6debc67fb159a7438dfe6b9993c871 |
| LiBRe threshold | Official strict FP32 p>FP32(0.6), not fit on our TEST |
| Equal budgets | Top10/20/40/80 and ceil(L/20),ceil(L/10),ceil(L/5); Top20 prespecified primary |
| Additional control | p>=0.5 for both models |
| Targets | Existing native heavy-atom d_min<5 A contact labels |
| Metric axis | Exact original PLC residue axis, capped at1800 |
| Metrics | Pooled/sample-macro F1,MCC,P,R; pooled AP/AUROC; overall and each source |
| Overlap policy | Accepted; no SI/PDB/example-overlap filtering |
| Selection | No checkpoint/K/threshold search or model updates |

LiBRe inputs use raw FP32 ESM2 L33, residue-only tokens, official U/Z/O/B toX
replacement, and zero padding to1500. The existing sequence-hash cache is
reused; missing raw embeddings are extracted on the allocated GPU. PLC's
normalized features are not substituted. Each represented chain is predicted
for the ligand, without GT binding-chain selection. Chains above1500 use
250-residue-overlap windows; probabilities are averaged on shared positions.
Whole observed chain context may extend beyond the metric axis.

Ligands use the official40D SMILES graph without an atom cap. Canonical
isomeric SMILES come from fully sanitized original SDFs. Their heavy-atom
coordinates/order must match the native ligand cache used for label creation;
coordinates themselves are never passed to LiBRe. No heavy-atom deletion, guessed
bond repair or sanitization bypass is used to force input acceptance.

The1,649 required PDBbind SDFs were staged from local New/ only: a1.8MB archive,
not the full dataset. BioLiP SDFs remain at their existing HPC2 paths.
Protein sequence/residue keys, original label hashes and the saved PLC targets
are checked per row. Public assets, input manifests and evaluation source
digests are locked before GPU prediction.

Stable original residue order resolves Top-K ties; analytic uniform-tie
expected metrics are also saved. Sample-macro weights ligand rows, not proteins.
This run does not report external COACH420/HOLO4K results or protein-balanced
metrics. AP is sklearn average precision, not trapezoidal PR integration.

## Coverage and Failure Semantics

All17,395 rows are requested. CPU input errors are logged per sample in
prepared/failures.json; accepted inputs proceed without performance filtering.
If any input fails, both models are evaluated on exactly the same successful
subset. Coverage and source counts stay explicit; the output is provisional
partial coverage, not a claim of full TEST coverage.

CPU preparation exits successfully only when every requested row is accounted
for and at least one is accepted. GPU inference records complete traversal
and matched metrics, but exits nonzero if any requested input lacked a
prediction. Numerical/GPU failures are not automatically repaired by changing
precision, caps or inference policy. Atomic, fingerprinted row predictions
support an explicitly inspected restart without reusing changed inputs.

No original split, label, PLC result or upstream model file is modified.

## Initial Deployment and Submission

Submitted **2026-10-23 03:25:43 UTC+8**:

| Stage | Job | Partition | Allocation | Initial state |
| --- | --- | --- | --- | --- |
| Input preparation |13272415|i64m512ue|8 CPUs,48G,4h|PENDING(Priority)|
| LiBRe inference/comparison |13272416|i64m1tga800ue|1 GPU,8 CPUs,48G,12h|PENDING(Dependency)|

Inference uses afterok:13272415 and kill-on-invalid-dep=yes. The12h limit is
not a measured ETA. No final comparison scores are available at submission.

Remote root:

```text
/hpc2hdd/home/shuang886/PLC/external_baselines/libre/joint_test_20261023
```

Runner: scripts/benchmark/run_libre_joint_test.py.
Launcher: scripts/slurm/run_libre_joint_test.slurm.
Initial immutable code snapshot: code_v2, commit **880f888**.
Source archive SHA256:
f478f0b43764c6df6aa7797951a2373c0bc73055b384b1eb2a651ad96aff01a4.
SDF archive SHA256:
235eb1df61805bd73b9dcc1474eae64428571846df3771379d55e6cb849d4838.

The initial08fa897 snapshot and input smoke are preserved separately.
The final version avoids retaining per-row multiprocessing tensor file
descriptors; workers return plain NumPy arrays, converted to tensors by the
parent. No prediction recipe changed in this resource-safety correction.

29 local targeted tests passed; the7 new tests were rerun after the correction.
Shell syntax and remote Python compilation passed. Real input smoke passed
for BioLiP1a47_A_BS05_ADH (683 residues,11 ligand atoms) and PDBbind1a0t
(1,239 residues across3 chains,23 ligand atoms). This smoke validated
inputs/labels/graphs only, not model performance.

Retain submission.json, deployment_v2.json, logs/, prepared/protocol.json,
prepared/complete.json, prepared/failures.json and results/summary.json.
Raw embeddings and per-row prediction caches remain on HPC2, not in git.

## Input-Compatibility Correction and Replacement Jobs

The first CPU job13272415 completed:16,162 accepted,1,233 rejected.
909 failed full native-SDF sanitization. The other324 hit an overly strict
adapter check that equated graph nodes with heavy atoms. Inspection found
stereochemical explicit H retained by RDKit and the official SMILES graph,
for example1ddo_A_BS02_ITR:15 heavy atoms but16 valid graph nodes.
Those are not missing heavy atoms and must not be excluded for this reason.

Commit **7f074c9** checks heavy-atom element/isotope conservation through
SMILES round-trip, then validates graph shape/edge bounds against the exact
official parsed molecule, including retained H. Native geometry, full
sanitization, upstream graph code and evaluation policy remain unchanged.
The30 targeted tests pass, including the stereo-H regression.

Replacement jobs submitted **2026-10-23 03:35 UTC+8**:

| Stage | Job | Partition | Resources |
| --- | --- | --- | --- |
| Corrected preparation |13272844|i64m512ue|8 CPUs,48G,4h|
| Corrected inference |13272845|i64m1tga800ue|1 GPU,8 CPUs,48G,12h|

Inference depends on afterok:13272844. The superseded13272416 was held
before it started and cancelled only after replacement submission succeeded.
The initial preparation artifacts are retained, not overwritten; its coverage
numbers above are diagnostic, not the final matched cohort.

Active code:code_v3, commit7f074c9. Source archive SHA256:
037f430eca6da3eed5ba1135169ee0efc7a9eb50b5b28ab5363d4b0dc718ebb7.
Active work root is the v3/ subdirectory of the remote root above. It uses
a symlink to the unchanged staged SDFs; submission.json records replacement
job IDs and the correction rationale. Final coverage and inference metrics
must be read from v3/prepared/ and v3/results/, respectively.
