# Practical recovery: full results checked 2026-10-08

## 2026-10-08 production update

Live queue empty at the initial check. Cache 12930151 completed in 01:26:44.
Heads 12930152_0/1/3 completed in 00:09:34 / 01:10:00 / 02:22:54.
12930152_2 timed out at 04:00:26, not OOM. Tokens and mean controls each
finished 20 epochs; blind control stopped after epoch 10, so the tokens summary
correctly says `complete=false`. Array element concrete job IDs 12933329,
12933356, 12933375 explain the different output folder names; these are not
additional successful reruns.

Full cache: TRAIN 10317 / VAL 647, protocol `bf16_gvp_fp32_v1`, practical v3,
gate 0.391169935464859. Results synced into `tmp/strict_site_practical_20261008`.
The following are model-selected strict VAL metrics, NOT TEST/COACH/HOLO.

| Method | Best epoch | Sample F1 | Delta vs frozen baseline |
| --- | ---: | ---: | ---: |
| Frozen base44 | n/a | 0.521790 | 0 |
| stats_only | 1 | 0.520408 | -0.001382 |
| protein_site | 14 | 0.522232 | +0.000442 |
| ligand_site | 15 | 0.521952 | +0.000162 |
| gain | 20 | 0.522317 | +0.000527 |
| paired_absolute | 14 | 0.521335 | -0.000455 |
| tokens | 17 | 0.521794 | +0.000004 |
| mean token control | 15 | 0.521581 | -0.000209 |
| ligand_refiner | 10 | 0.522200 | +0.000410 |
| protein_refiner | 6 | 0.521912 | +0.000122 |

All 647 VAL protein keys are unique, so sample/protein means coincide here.
Paired bootstrap over those proteins (5000 replicates, RNG44), conditional on
the VAL-selected model, gives gain delta CI95 [-0.0003945, 0.0015224]. Every
completed method's delta interval crosses zero. These intervals do not account
for checkpoint selection or multiple model comparisons and are not confirmatory.

The ranking candidate-pool oracle is 0.644847, leaving 0.122530 regret even for
gain. This is a GT-selected upper bound over available masks, not deployable
performance or proof that the head can recover that gain. Gain changes only
20 rows (11 improve, 7 worsen); its unguarded F1 is 0.522048. Stats-only best is
epoch1 and falls to 0.517748 by epoch20. Other quality/token heads mostly stay
near baseline rather than gaining steadily. Boundary refinement changes 351
rows but improves the mean only 0.000410; predicted count p50 moves 20 -> 19,
precision rises while recall falls. No major assignment improvement is supported.

Gain real/permuted/zero: 0.522317 / 0.521568 / 0.521583.
Ligand-site real/permuted/zero: 0.521952 / 0.519406 / 0.521389.
Tokens real/permuted/zero: 0.521794 / 0.521502 / 0.521685.
These perturb only the NEW HEAD'S query, holding real-ligand backbone outputs
and proposals fixed. They are NOT full-model ligand-dependency evaluations.

Do not promote these heads or expand their seeds on this evidence. Finishing
the blind token control remains optional diagnostic work, not a prerequisite
for recognizing the negligible completed token-head gain. The separately
approved GDN architecture probe still needs its GPU smoke repaired; it has no
accuracy result yet. Historical submission-time observations follow unchanged.

## Historical 2026-10-06 check

Checked 2026-10-06 at approximately 19:41 scheduler time using squeue, sacct,
scontrol, audit summary and all smoke branch summaries. Synced lightweight
results: `tmp/strict_site_practical_20261006/` (no large caches or weights).

| Job | Status | Runtime / next condition |
| --- | --- | --- |
| 12930149 full VAL approval | COMPLETED 0:0 | 00:07:53; all 647 rows |
| 12930150 all-control smoke | COMPLETED 0:0 | 00:02:33; TRAIN 68 / VAL 38 |
| 12930151 full cache | PENDING Priority | Dependencies fulfilled; not yet running |
| 12930152_[0-3%3] full head groups | PENDING Dependency | afterok:12930151 |

The cache's provisional start estimate was 2026-10-07 13:18:00 scheduler time.
This can change and is not a promise. All jobs remain in i64m1tga800ue, one GPU
per task, head concurrency <=3. No jobs were changed or resubmitted in this check.

## Audit result

`complete=true`, `passed=true`, policy `prediction_stable_v3`, no failures.
Mixed-protocol full VAL sample F1 is **0.5217902904**, compared with archived
BF16 **0.5218677608** (difference -0.0000774704). Micro F1 is 0.5388621172;
predicted count p50/p90 remains 20/30.

This particular audit had no tolerated repeat-drift rows; its recorded v2
comparison also passed. This does not erase the earlier 6fac repeat failure or
prove general determinism. The practical policy remains explicit and unchanged.

## Smoke results, not production efficacy

| Head / control | Small-VAL sample F1 | Replaced predictions |
| --- | ---: | ---: |
| Unchanged frozen baseline | 0.499399 | 0 |
| stats_only | 0.298548 | 17, all worsened |
| protein_site / ligand_site | 0.499399 | 0 |
| gain / paired_absolute | 0.499399 | 0 |
| tokens / mean / blind | 0.499399 | 0 |
| ligand_refiner / protein_refiner | 0.499399 | 0 |

The first three groups ran only two epochs; the boundary group used one refiner
epoch and one quality epoch. Losses/gradients are finite and every code path
completed. Most heads fall back to the original prediction, so equality is not
evidence that new representations improve assignment. The stats-only collapse
is a smoke observation, not grounds to reject the fully trained control.

These are 38-row VAL numbers, not the 647-row full VAL result and not TEST.
No full four-group training result or new COACH420/HOLO4K result exists yet.
