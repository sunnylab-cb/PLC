# LiBRe / PLC matched ranking and candidate diagnostics

**Completed:** initial diagnostic job 12963165 and tie-sensitivity job 12963186
both exited `COMPLETED 0:0`, covering the same 7729 approved pairs. No GPU,
training, benchmark gate selection or changes to original predictions were used.

## Scope and execution

CPU-only follow-up to the accepted matched benchmark in
`libre_benchmark_20261010.md`. No training, checkpoint changes, new inference,
chemistry repair or benchmark threshold fitting. Original results stay frozen.

- Job: `12963165`, `emergency_cpu`, four CPUs, 16 GB host memory, no GPU.
- One task processes both COACH420 (491 pairs) and HOLO4K (7238 pairs).
- Remote result root:
  `/hpc2hdd/home/shuang886/PLC/external_baselines/libre/results/ranking_12963165/`.
- Logs: `external_baselines/libre/logs/ranking_12963165.{out,err}` under the same
  remote PLC root.
- Isolated snapshot: `external_baselines/libre/runners/ranking_v1/`.
- Runner: `scripts/benchmark/diagnose_libre_ranking.py`.
- Slurm: `scripts/slurm/run_libre_ranking_diagnostics.slurm`.
- Nine local tests passed; a two-protein/two-pair remote smoke passed before
  submission. Full results must not be inferred from that smoke.

The run verifies original summary/CSV/protocol hashes against the accepted
report, requires the exact approved sample population, checks identical residue
keys, coordinates and binary contact labels for every matched pair, and checks
cached native predictions against each model's original CSV F1/contact counts.
Each NPZ digest is saved in `prediction_audit.csv`.

## Prespecified diagnostics

1. Fixed Top-10/20/40/80 contact-probability residues; Top-20 is the primary
   equal-budget diagnostic. Counts are capped by the common residue-axis length.
2. Top-5%/10%/20% of the common residue axis, using ceiling and minimum one.
3. Reciprocal budgets: both models predict exactly as many residues as PLC's
   native mask, then exactly as many as LiBRe's native mask. No GT count is used.
4. Spatial components at 8 A, minimum three residues, on native / Top-20 /
   Top-40 / Top-10% masks. Rank each component by its top-20%-mean probability.
5. A shared, geometry-only pool: up to 32 farthest-point CA seeds, 8 A spheres,
   minimum three residues. The two models rank exactly the same candidates by
   top-20%-mean probability, with no target-dependent proposal construction.

All settings are reported, not selected by best benchmark score. Ties in
residue ranking use stable shared residue order; ties in candidate ranking use
the residue-index tuple. Cutoff tie frequency is reported. No ROC/PR threshold
search or test-label-derived contact budget is performed.

Report sample means and protein-balanced means, plus 2000 paired protein
bootstrap resamples (seed 20261010) for native, Top-20, Top-10%, reciprocal
budgets and shared-proposal selected F1. These intervals describe this benchmark
population, not seed variance, multiple-testing correction or training overlap.

## Candidate interpretation limits

GT contact labels enter only metrics, oracle ceilings and subgroup definitions.
The oracle F1 selects the best candidate using GT and is NOT deployable accuracy.
Candidate DCC uses distance to the current GT contact-residue centroid, not the
native ligand center. Report both DCC<4 A and residue IoU>=0.25 hits for selected,
top-three and any candidate. These diagnostics are not standard SOTA metrics or
a direct replacement for biological ligand assignment accuracy.

Shared geometry candidates are a fixed coverage control, not a learned pocket
detector. If their oracle ceiling is low, a low top-1 score cannot establish a
pure assignment failure. Model-specific components can merge nearby sites via
transitive connections; their number/coverage differs between models.

Strata use original ligand-row multiplicity (one, two-three, four-plus), and
distinct-pocket targets (Jaccard<=0.2 AND centroid separation>=8 A). An additional
different-chemistry flag uses unequal cached canonical SMILES, not a fresh
chemical-equivalence audit. These are descriptive, label-defined strata, not
signals used to construct or rank proposals.

## Completed first pass

Job **12963165 COMPLETED 0:0**, wall time 1m24s, MaxRSS 1,072,068 KiB.
All 7729 approved pairs passed exact label/key alignment and coordinate/count/F1
checks. Small artifacts are in `tmp/libre_ranking_12963165/`; all transferred
digests match. Full per-row CSVs remain on HPC2.

### Equal-cardinality residue ranking

F1 below uses stable shared residue order for tied probabilities. It is a
prespecified diagnostic, not a newly selected deployment policy.

| Dataset | Rule | LiBRe F1 | PLC F1 | Difference |
|---|---|---:|---:|---:|
| COACH | Original decisions | 0.4494 | 0.4439 | +0.0055 |
| COACH | Top-20 | 0.4733 | 0.4456 | +0.0277 |
| COACH | Top-10% | 0.4571 | 0.4248 | +0.0324 |
| COACH | Both use PLC's native count | 0.4708 | 0.4439 | +0.0269 |
| COACH | Both use LiBRe's native count | 0.4494 | 0.4231 | +0.0263 |
| HOLO | Original decisions | 0.3863 | 0.3300 | +0.0563 |
| HOLO | Top-20 | 0.4044 | 0.3224 | +0.0820 |
| HOLO | Top-10% | 0.4082 | 0.2833 | +0.1249 |
| HOLO | Both use PLC's native count | 0.4209 | 0.3300 | +0.0909 |
| HOLO | Both use LiBRe's native count | 0.3863 | 0.2716 | +0.1147 |

Top-20 paired protein-bootstrap 95% intervals for LiBRe minus PLC are
[+0.0100, +0.0455] on COACH and [+0.0778, +0.0863] on HOLO. The latter's
protein-balanced difference is +0.0903, interval [+0.0853, +0.0953].
Identical residue budgets do not remove LiBRe's ranking advantage. Broadening
PLC to LiBRe's native count instead reduces PLC's HOLO F1 to 0.2716.

This is not solely a multi-ligand assignment finding. On one-ligand proteins,
Top-20 F1 is 0.6885 vs 0.6138 (COACH), and 0.6965 vs 0.6050 (HOLO). On COACH's
different-chemistry/distinct-pocket subset, however, Top-20 is approximately tied
(0.3182 vs 0.3190). Do not claim superiority in every stratum.

### Multi-region candidate diagnostics

HOLO, native masks, 8 A components with minimum three residues:

| Model | Mean candidates | Selected F1 | Oracle F1 | Top-1 DCC<4 | Top-3 DCC<4 | Any DCC<4 | Any IoU>=0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LiBRe | 2.3846 | 0.3585 | 0.5480 | 0.4043 | 0.5876 | 0.6268 | 0.7820 |
| PLC | 1.2537 | 0.3077 | 0.3213 | 0.2079 | 0.2153 | 0.2153 | 0.4637 |

DCC/IoU success denominators exclude the five zero-contact HOLO rows (7233
eligible pairs); F1 retains all 7238. Candidate counts differ, so unrestricted
any-candidate success alone is not a fair fixed-budget ranking comparison.
Nevertheless, these results show why the old single-centroid LAA ordering is
insufficient: LiBRe often has a spatial component near the current target even
when averaging its full prediction places the centroid elsewhere.

The important corrective finding for PLC is that re-ranking its native-mask
components alone has little headroom: oracle F1 is only 0.3213 under this fixed
proposal rule. This does not prove every possible PLC-derived proposal fails,
but contradicts treating the problem as *only* choosing among already-good
native candidates. Missing coverage / residue ranking also needs attention.

### Shared geometry control

The common FPS32 / radius8 pool has identical oracle F1 for both models:
0.3857 COACH and 0.2925 HOLO. Its any-candidate DCC<4 success is only 0.1358 /
0.0355; any-candidate IoU>=0.25 is 0.4671 / 0.2404. Therefore this geometry-only
pool itself is too weak to establish a clean assignment-head ceiling.
Selected F1 is 0.2114 / 0.2098 (LiBRe / PLC) on COACH and 0.1578 / 0.1287 on HOLO.
Do not interpret these poor sphere-proposal scores as the original models'
benchmark performance.

## Tie-sensitivity follow-up

The first pass revealed exact cutoff ties in LiBRe on 6.81% of HOLO Top-20 rows,
21.17% of Top-10% rows, and 24.47% when using PLC's native count. PLC has no such
cutoff ties in these rules. Stable residue order is reproducible but may favor
particular chain copies, so a follow-up explicitly measures tie sensitivity.

Job **12963186**, same CPU allocation and entire approved cohort, uses isolated
snapshot `runners/ranking_v2/`. It preserves the original protocols and results,
adding exact expected F1 under uniform selection at tied cutoffs, pessimistic/
optimistic tie bounds, and mean outcomes across tied top-scoring candidates.
The expectation uses GT to evaluate a fixed random-tie policy, never to select
the correct tied residues. Bounds are not model predictions. Eleven local tests
pass, including agreement with exhaustive tie enumeration; a remote two-pair
smoke passed. The full-population results are recorded below.

### Completed tie audit

Job **12963186 COMPLETED 0:0**, wall time 1m31s, MaxRSS 1,036,404 KiB. All 7729
pairs completed. Every original v1 aggregate metric, sample/protein count and
input NPZ digest remained unchanged. V2 adds diagnostics only. Verified small
artifacts are in `tmp/libre_ranking_12963186/`; full row CSVs remain at the matching
remote `results/ranking_12963186/` path.

| HOLO rule | LiBRe stable-order F1 | LiBRe random-tie expected F1 | LiBRe tie bounds | PLC F1 |
|---|---:|---:|---|---:|
| Top-20 | 0.404402 | 0.404411 | [0.403165, 0.405608] | 0.322387 |
| Top-10% | 0.408178 | 0.408191 | [0.407448, 0.408964] | 0.283325 |
| Both use PLC native count | 0.420888 | 0.421002 | [0.417372, 0.424436] | 0.329977 |

Even the pessimistic tie bounds retain the equal-budget ranking advantage.
The random-tie Top-20 difference is +0.08202, paired protein-bootstrap 95% interval
[+0.07785, +0.08631]. At PLC's native count it is +0.09102, interval
[+0.08652, +0.09551]. On COACH the corresponding scores are unchanged because
there are no cutoff ties in these two rules. No evaluated cached probability was
exactly 1.0; these ties are not saturation at that endpoint. Their biological or
numerical cause has not been established by this audit.

LiBRe native-component selected F1 changes only from 0.358534 to 0.358495 under
uniform top-score ties; selected DCC<4 changes from 0.404258 to 0.404730. This
does not change the candidate coverage-versus-selection interpretation. The
candidate tie average is conditional on the existing component pool, not an
average over all possible tied-residue masks for Top-k component construction.

## Conclusion and next direction

1. **Prediction extent alone does not explain the gap.** LiBRe ranks true
   contact residues better at the fixed budgets tested, including one-ligand
   proteins. Merely increasing PLC's contact budget is not an adequate remedy.
2. **Assignment alone is too narrow a diagnosis for PLC.** Under the tested
   native-mask component generator, even oracle selection has low F1. Improve
   residue ranking and true-pocket candidate coverage before relying on a
   stronger reranker over the same inadequate candidate set.
3. **Single-centroid LAA does not establish LiBRe's inferiority at localization.**
   Multi-region diagnostics reveal much more target coverage in LiBRe. They do
   not prove correct biological ligand assignment or replace a matched,
   fixed-budget object-level detection protocol.
4. **Geometry-only FPS spheres are not a sufficient candidate generator.** Their
   poor oracle coverage prevents a clean attribution of the whole deficit to
   candidate scoring. Do not deploy this weak control as the next architecture.

Next controlled work should use strict VAL: compare ranking-oriented checkpoint
selection (AUPRC versus F1, where checkpoints exist) and a broader protein-side
candidate source that is not restricted to the final high-threshold ligand mask.
Track candidate recall and best-candidate F1 separately from selection accuracy,
then freeze the policy before any further benchmark run. Do not promote the
highest scoring Top-k setting here to a test-selected inference rule.

This comparison is specific to the released LiBRe checkpoint and PLC GFGF run.
Checkpoint training overlap and chemistry quality limitations from the parent
report remain; these diagnostics are not evidence that one architectural module
causes the gap or that LiBRe is a leakage-controlled SOTA baseline.
