# Joint HOLO4K: Nonshared-PDB Reference Comparison

## Scope

The latest joint cohort was not filtered against either HOLO4K or COACH420.
This follow-up removes direct PDB accession reuse from the **same matched
HOLO4K rows for both PLC and released LiBRe**. It uses existing predictions,
not another inference or training run.

Primary cohort: exclude any benchmark PDB occurring in PLC **TRAIN or VAL**.
VAL is included because it selected the checkpoint and inference gate.
Internal TEST is not training exposure; excluding all three internal splits
is separately reported as a stricter accession-only sensitivity control.

| Cohort | Excluded internal accessions | Pairs | Protein keys | Role |
| --- | --- | ---: | ---: | --- |
| all_matched | None |7238|3424| Preserve the original matched comparison |
| no_train_pdb | TRAIN |3780|1759| Training-accession exclusion control |
| no_train_val_pdb | TRAIN + VAL |3215|1528| Primary nonshared-PDB comparison |
| no_internal_pdb | TRAIN + VAL + TEST |2558|1256| Additional accession-only control |

All rows for a protein remain together. Membership uses only the frozen
split manifest and successful LiBRe sample IDs, never F1, probabilities,
labels, contact counts, or model-specific successes on selected subgroups.
The full cohort is the prior7238 jointly evaluated rows; seven native LiBRe
chemical-input failures remain excluded for both models.

**This is accession-disjoint, not protein-sequence-disjoint or SI30-clean.**
Different PDB entries can contain identical or homologous proteins. The
exclusion is relative to the PLC split, not a certification against LiBRe's
unavailable complete training manifest. Full-cohort results were already
known when the user requested this analysis; it is explicitly post-hoc,
metadata-defined analysis, not prospectively isolated external testing.
Do not use any subgroup to tune the model, threshold, checkpoint or K.

## Fixed Scoring And Verification

Reuse joint seed44 epoch25 and its original VAL gate0.5535550117492676;
LiBRe retains released FP32 p>0.6. Primary equal budget is fixed Top20.
Also retain Top10/40/80, ceil(L/20), ceil(L/10), ceil(L/5), and p>=0.5.

Report sample-mean and pooled-residue F1/MCC/P/R, protein-mean-of-ligand-mean,
protein-mean-of-ligand-max F1, AP and AUROC. Protein-best is an optimistic
diagnostic, not the per-ligand headline. Native thresholds yield different
prediction counts, so native F1 and equal-budget ranking must both be shown.

Before scoring, verify parent protocol/split/CSV/code hashes and every saved
PLC/LiBRe prediction SHA256 against the completed parent audit. Recheck
residue keys, coordinates, labels, protein/ligand identity and native
prediction counts/F1. Recomputed full-cohort metrics must reproduce the
parent summary to absolute tolerance1e-12. Fail on changed coverage or data.

## Completed Results

CPU13280188 completed with exit0:0 in4m05s, MaxRSS866808K. All14476
prediction files passed the frozen SHA256/axis/label checks. Full-cohort
recomputation reproduced the parent metrics to absolute tolerance1e-12.

All following F1/MCC values are **sample means over individual ligand rows**,
not pooled-residue or optimistic protein-best scores.

| Cohort | PLC Top20 F1 | LiBRe Top20 F1 | LiBRe minus PLC | PLC Top20 MCC | LiBRe Top20 MCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_matched |0.354668|0.404402|0.049734|0.336147|0.390166|
| no_train_pdb |0.306655|0.379610|0.072956|0.285543|0.364792|
| no_train_val_pdb (primary) |0.306322|0.382451|0.076129|0.284673|0.367583|
| no_internal_pdb |0.303579|0.383738|0.080159|0.281014|0.368446|

Additional primary-cohort metrics, same3215 pairs /1528 proteins:

| Metric | PLC | LiBRe |
| --- | ---: | ---: |
| Top20 protein-mean-of-ligand-mean F1 |0.406499|0.498795|
| Top20 protein-mean-of-ligand-mean MCC |0.381119|0.482945|
| Native-threshold sample F1 |0.316193|0.385165|
| Native-threshold sample MCC |0.292346|0.420615|
| Native-threshold protein-mean-of-ligand-mean F1 |0.419499|0.488185|
| Native-threshold pooled-residue F1 |0.312849|0.341931|
| Native mean predicted contacts per pair |20.846345|88.762364|
| Pooled average precision |0.210492|0.336906|
| Pooled AUROC |0.815991|0.942947|

Removing TRAIN+VAL shared accessions lowers PLC Top20 sample F1 by0.048346
and LiBRe by0.021951 relative to the full cohort, widening the observed gap
from0.049734 to0.076129. Protein-balanced Top20 also favors LiBRe, so this
is not solely a few high-ligand-count proteins dominating the sample mean.
The primary-cohort native-threshold result agrees with the fixed-budget
direction; the residual difference is not removed by equalizing Top20.

This **does not identify a causal leakage effect**: the excluded and retained
proteins differ in composition/difficulty, homologous sequences remain
uncertified, and LiBRe's training overlap remains unknown. It also does not
establish that a particular encoder or architecture is responsible. These
are descriptive results for this frozen checkpoint and these subsets, not
significance estimates or permission to iterate on benchmark outcomes.

The result is less flattering than the full matched cohort, but more directly
addresses performance without exact PLC TRAIN/VAL PDB reuse. Retain both
reports, never relabel accession-disjoint as sequence-SI30-clean.

Unaltered [summary JSON](results/joint_holo_nonshared_20261023.json), SHA256:
3bae54ae85588b598dac9164d6d2cc8c94bcf8a70dcc747793b16ec8c4de29f3.
It includes all four cohorts and every fixed budget, not only the rows above.

## Execution

CPU job13280188, i64m512ue,4 CPUs,32G,2h time limit. No GPU or model edits.
Ran on cpu1-57; all four membership lists were frozen before reading NPZs.
Code commit2108263; source archive SHA256:
c5705087930ad192684bc05a0744cd38ed958b88220bb12449b0de8419232d8d.
61 targeted local tests passed, including legacy benchmark input integrity;
shell syntax checked. Existing optional torch-scatter/torch-sparse import
warnings in the local BIO environment do not affect this CPU metric path.

```text
/hpc2hdd/home/shuang886/PLC/experiments/joint_holo_nonshared_20261023/
  code_v1/
  source.tar.gz
  job_id.txt
  logs/13280188.{out,err}
  results/protocol.json
  results/cohorts.json
  results/summary.json
```

The independent snapshot does not alter the completed parent inference:
[full HOLO4K results](joint_holo_reference_20261023.md#completed-results-2026-10-23).
