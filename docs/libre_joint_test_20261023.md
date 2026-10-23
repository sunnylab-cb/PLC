# Released LiBRe Versus Latest PLC Joint TEST

## Completed Matched Comparison: 2026-10-23

GPU job **13272845** finished all **16,486 accepted rows** at04:35:25 UTC+8,
after12m08s wall time. Slurm reports FAILED(1:0) because the runner deliberately
exits nonzero for incomplete original-cohort coverage:909 BioLiP inputs failed
native-SDF sanitization before inference. This is **not an OOM or an interrupted
prediction run**. summary.json has traversal_complete=true, complete=false;
there are16,486 unique per-row prediction files, with no additional inference
failures. CUDA peak allocated/reserved memory was2.915/3.809 GiB (not total GPU
process memory). No replacement job is needed.

The cohort is1,649 PDBbind +14,837 BioLiP rows,6,181,200 residues and248,186
positive residues. Both models use precisely these rows, labels and residue
axes. PDBbind coverage is complete; BioLiP coverage is partial. The released
checkpoint's possible training overlap was accepted, not used for filtering.

### Main Results

F1 and MCC below are pooled-residue metrics, not legacy training F1_c0 or
sample-macro averages. Native thresholds remain fixed: PLC's existing VAL
gate0.5535550117492676 and LiBRe's official p>0.6. Top20 was the prespecified
equal-budget primary comparison, not selected after observing TEST scores.

| Source | Model | Native F1 | Native MCC | Top20 F1 | Top20 MCC | AUPRC (AP) | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All matched | PLC |0.594912|0.577970|0.562365|0.547368|0.614588|0.949701|
| All matched | LiBRe |0.423227|0.438794|0.469208|0.448652|0.457398|0.923496|
| PDBbind | PLC |0.621448|0.608982|0.603521|0.590569|0.660390|0.961091|
| PDBbind | LiBRe |0.375604|0.435311|0.484445|0.467338|0.385830|0.958787|
| BioLiP | PLC |0.591380|0.573631|0.557294|0.541997|0.607775|0.947569|
| BioLiP | LiBRe |0.432666|0.441731|0.467331|0.446230|0.484298|0.918228|

All-matched PLC minus LiBRe: native F1+0.171685, native MCC+0.139175,
Top20 F1+0.093157, Top20 MCC+0.098715, AP+0.157190 and AUROC+0.026204.
Sample-macro native F1 is0.551122 versus0.411326; sample-macro Top20 F1 is
0.534880 versus0.447704. The direction is not an artifact of choosing pooled
rather than sample-macro aggregation. Samples are ligand rows, not equally
weighted proteins or homology clusters.

### All Prespecified Ranking Budgets

| Rule | PLC pooled F1 | LiBRe pooled F1 | PLC minus LiBRe |
| --- | ---: | ---: | ---: |
| Top10 |0.489282|0.394987|+0.094295|
| Top20 |0.562365|0.469208|+0.093157|
| Top40 |0.449213|0.411665|+0.037548|
| Top80 |0.288022|0.276857|+0.011164|
| ceil(L/20) |0.516640|0.429416|+0.087224|
| ceil(L/10) |0.435181|0.393859|+0.041321|
| ceil(L/5) |0.291625|0.279019|+0.012605|

The common p>=0.5 control gives pooled F1/MCC0.592686/0.576820 for PLC and
0.406346/0.428093 for LiBRe. All budgets, sample-macro values and tie-aware
controls are retained in the [unaltered result JSON](results/libre_joint_test_20261023.json).
At Top20, uniform tie expectation changes LiBRe overall F1 from0.469208 to
0.469101, too small to account for the observed gap.

Do not claim a win at every budget in every subset. On PDBbind, LiBRe has
slightly higher pooled F1 at Top80 (0.341029 vs0.333019), ceil(L/10)
(0.377517 vs0.376810), and ceil(L/5) (0.253039 vs0.241305). Those were
predeclared controls and are not a reason to select a different K on TEST.

### Interpretation and Claim Boundary

At the native thresholds, LiBRe predicts38.79 residues per sample versus
PLC16.85, with precision/recall0.29375/0.75681 versus0.56323/0.63037.
The broad prediction pattern is especially pronounced in PDBbind:70.60
predictions per sample and recall0.90680, versus PLC19.03 and0.63138.
This supports a precision/coverage tradeoff in this evaluation, not a
conclusion that LiBRe cannot identify binding regions.

Native-threshold differences do not fully explain the result: the fixed
Top20 comparison, average precision and sample-macro metrics also favor
PLC. The defensible claim is that **the current frozen PLC model has higher
observed residue-prediction performance than the released LiBRe checkpoint
on this matched internal TEST cohort under the stated protocol**.

This is not an equal-training-data architecture comparison or a universal
SOTA claim. Training corpora, preprocessing and target definitions differ;
LiBRe receives chainwise sequence context, whereas PLC also uses structure.
Multi-chain evaluation can predict several equivalent sites for LiBRe;
this experiment does not identify which cause explains its PDBbind behavior.
There is one evaluated PLC checkpoint and one released LiBRe checkpoint,
no seed-level superiority test or cluster-bootstrap confidence interval.
The result does not overturn earlier external COACH420/HOLO4K findings.
No model, threshold, budget or new training experiment is selected using
these TEST outcomes.

Verification: prepared-data, failure-manifest, protocol and evaluation-code
hashes match their locked records; prediction counts and both models'
sample/residue/positive/Top20-prediction denominators agree by source.
Local downloaded summary matches remote SHA256:
7e1b9e465fd6f095fb0cd3b6d2c95279f3740647eacea6d2feb20f7a90a64b0c.

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

## Corrected Preflight Completed

Job13272844 finished input preparation successfully. Every requested row is
accounted for; all324 stereo-H graph rejections were recovered without
changing the official graph or repairing molecule chemistry.

| Source | Requested | Accepted | Rejected |
| --- | ---: | ---: | ---: |
| PDBbind |1,649|1,649|0|
| BioLiP |15,746|14,837|909|
| Total |17,395|16,486|909|

The remaining909 native SDFs do not pass full RDKit sanitization. The matched
comparison therefore covers94.77% of the requested cohort, with complete
PDBbind coverage. Both models will be scored on the same16,486 accepted rows;
do not compare LiBRe subset metrics directly to the older full-cohort PLC
summary. No overlap-based filtering occurred.

Preparation fingerprints:

- ready.pt:7664d92b9da94c6de85b0e1f286cd2f3634187589e0169ac33c8a6a5f7c16087
- failures.json:da8a59037be852fbae63543d8bb88f207640cb21eabaa402fdcc4db21d15239c
- protocol.json:8c67f93003e6356022b66910ce1d1ba17fb9156561b4c87457e3c809442209ec

At the post-preparation check,13272845 is PENDING(Priority), not blocked on
dependency. No LiBRe TEST scores exist yet.
