# Full strict VAL precision audit: result and single count-gate failure

Subsequent update: the user approved a separate v2 acceptance policy and new
job chain. See `strict_site_precision_v2_recovery_20261005.md`. The v1 results
and status recorded below remain unchanged.

Checked 2026-10-05 using Slurm accounting, logs, all 647 comparison rows and
prediction NPZs. Local evidence: `tmp/site_precision_val_12919239/`.
Remote evidence: `strict_site_precision_20261005/audit_12919239/` under the HDD
PLC experiments directory. Archived 2k00 prediction was fetched separately to
verify the exact differing residue and probabilities.

## Execution status

Job 12919239 processed every requested VAL sample and produced a complete report.
It intentionally exited **2:0** because one numerical acceptance gate failed:
`2k00: binary disagreement`. Slurm state is FAILED; elapsed 00:08:11, from
02:32:17 to 02:40:28 scheduler time. There was no traceback, OOM or timeout.

Jobs 12919240 (smoke), 12919243 (new cache), and 12919245_[0-3%3] (heads) were
automatically CANCELLED without running. The queue is empty at this check. No
full head results, new TEST results or external benchmark results exist.

## Full-population measurements

Same checkpoint, strict VAL, source-aligned targets, original contact gate
0.391169935464859 and unchanged atom/residue budgets throughout.

| Metric | Archived BF16 | Fresh legacy BF16 | BF16 + FP32 GVP reduction |
| --- | ---: | ---: | ---: |
| Sample / exact-protein mean F1 | 0.521867761 | 0.521922443 | 0.521790290 |
| Micro F1 | 0.539017566 | 0.539059464 | 0.538862117 |
| Micro precision | 0.507463779 | 0.507538056 | 0.507317430 |
| Micro recall | 0.574755511 | 0.574755511 | 0.574589756 |
| Mean predicted contact count | 21.122102 | 21.119011 | 21.122102 |
| Predicted count p50 / p90 | 20 / 30 | 20 / 30 | 20 / 30 |

Mixed-protocol sample F1 differs by -0.0000774704 from the archive and by
-0.0001321529 from the fresh legacy pass. This is not an efficacy gain, but does
not indicate meaningful aggregate degradation or contact-count collapse here.
The sample and protein means coincide because the audited keys are unique in
this VAL population; this is not a sequence-cluster-balanced analysis.

Two mixed-protocol forwards were **bitwise equal in their exported probabilities
on all 647 samples**: maximum probability difference 0, binary flips 0.
This strengthens the earlier two-sample repeatability result, but is not a
guarantee across GPU types, software versions or unlimited repetitions.

Versus the archived BF16 reference:

- Maximum per-residue probability difference: 0.04833409, below 0.05 everywhere.
- Mean per-sample binary disagreement fraction: 0.000102916, below 0.001.
- 29 samples have binary changes, totaling 30 residue flips.
- F1 improved on 11 samples, worsened on 14 and was unchanged on 622.
- Absolute mean F1 change: 0.0000774704, below 0.001.
- The only failing criterion is per-row binary disagreement >0.01 on 2k00.

The fresh unmodified legacy pass happened to have no probability/count outlier
in this run. That does not negate the previous ten-repeat probe in which legacy
4k7o exceeded 0.05; a single stochastic pass does not establish repeatability.

## Exact 2k00 event

| Property | Value |
| --- | --- |
| Full / observed residues | 92 / 92 |
| Differing residue | A354, zero-based array index 45 |
| Archived and fresh legacy probability | 0.3910095989704132 |
| Mixed-protocol probability | 0.39469435811042786 |
| Fixed classification gate | 0.391169935464859 |
| Ground truth | Contact |
| Binary disagreement | 1/92 = 0.0108695652 (1.0869565%) |
| Predicted contacts, old -> mixed | 19 -> 20 |
| True positives, old -> mixed | 12 -> 13 |
| Sample F1, old -> mixed | 0.6666667 -> 0.7027027 |

The maximum probability difference anywhere in this sample is only 0.00397506.
This is a near-threshold crossing, not a large probability shift. Because the
protein is shorter than 100 residues, the original 1% rule effectively permits
zero binary changes. Improvement in this specific sample is NOT a legitimate
reason to exempt it selectively or change the classification threshold.

## Decision and next protocol proposal

The v1 acceptance result stays **FAILED**; no report, threshold, model or output
has been rewritten to claim a pass. The diagnostic nevertheless supports
preserved aggregate accuracy and markedly improved repeatability on this VAL.

For a prospective, separately versioned approval policy, consider an integer-aware
per-row budget: `flips <= max(1, floor(0.01 * n_full))`, while retaining the
maximum probability difference, aggregate disagreement/F1 checks, fixed contact
gate, source/axis checks and repeatability requirements. This applies to every
sample, does not depend on its label or whether F1 improves, and changes the
short-protein count tolerance explicitly rather than hiding an ID exception.
Under this proposed rule the present measurements would satisfy the gates, but
that post-result policy choice must remain disclosed and is not a pre-registered
pass or an independent efficacy result.

No such policy revision or new submission was implemented in this status check.
If adopted, preserve the v1 failure and create a new approval artifact, then
rebuild both TRAIN/VAL with the same mixed numerical protocol and compare every
head against its matched newly exported base output. Do not mix old/new tokens
or replace historical paper metrics without a matched evaluation.
