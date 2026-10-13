# Assignment experiment tracker

Updated 2026-10-09 from Slurm accounting, checkpoint metadata and TB inspection.
See `../docs/strict_site_parallel_results_20261004.md` for results and failure
analysis; the dated 20260926 execution tracker preserves submission-time states.
The targeted numerical diagnosis is in
`../docs/site_probability_repeat_results_20261005.md`.
The approved recovery chain is in `../docs/strict_site_precision_recovery_20261005.md`.
Its completed audit and one-row acceptance failure are in
`../docs/site_precision_val_results_12919239.md`.
Latest: full cache and head groups 0/1/3 completed; group2 timed out during its
blind control after completing tokens/mean. Best full-VAL gain-head F1 0.522317
vs baseline 0.521790, bootstrap delta CI crosses zero. No promotion warranted.
See `../docs/strict_site_practical_status_20261006.md` for all 647-row results.
Hybrid retry smoke and both scratch jobs now COMPLETED. At epoch35 both hit
loss-based early stopping. Best VAL F1 FFFF0.529855/e33 vs GFGF0.535996/e32;
tail-five0.521567 vs0.520920, so weak positive maximum but not stable superiority.
All recorded gradients finite. User approved best-F1 TEST without any training
change: `12954704_[0-1%2]` both COMPLETED 0:0 on -ue (00:03:25/00:09:58).
All647 TEST rows audited under real/shuffle/zero, argmax and fixed VAL gates.
Global real F1 FFFF/GFGF: argmax0.527015/0.531211; fixed-gate0.527644/0.536686.
Real-shuffle gaps shrink: argmax0.053363/0.041746; fixed-gate0.051199/0.043894.
Sample-mean real gains' paired bootstrap CIs cross zero. Small positive pilot,
not stronger ligand assignment or confirmed superiority; no new jobs submitted.
Subsequently approved COACH/HOLO: **12956268_[0-3%2]** all COMPLETED0:0 on
emergency_gpu;494/7245 rows per arm audited. Fixed-VAL-gate sample F1 FFFF/GFGF:
COACH0.449124/0.443525; HOLO0.327646/0.330157. HOLO LAA0.137698/0.146059,
PDR0.418496/0.443616; protein-balanced F1 0.439984/0.447219.
HOLO F1 gain is driven by single-ligand proteins; multi-ligand F1 remains weak.
COACH LAA declines. Keep as candidate, not a general assignment improvement.
See `../docs/qwen_hybrid_attention_feasibility_20261006.md` for the full result
and early-stopping mismatch. The earlier missing-einops failure is historical.

| Run / job | Stage | Purpose | Status | Next gate |
| --- | --- | --- | --- | --- |
| 12868601 sitequality_smoke | M0 | Frozen pre-cross export and small heads | FAILED 1:0, 00:01:41 | Diagnose independence check; not OOM |
| 12868608 sitequality_full | M1 | Full cache and stats/protein/ligand controls | CANCELLED, never ran | A new immutable retry must depend on a passing smoke |
| 12868960 state diagnostic | M0 | Repeat/intervention FP32 and BF16 | COMPLETED, 00:01:39 | FP32 stable on first sample; BF16 repeat noise measured |
| 12869013 all-branch smoke | M0 | Token export and every training control | COMPLETED, 00:02:31 | TRAIN 68 / VAL 38; execution only, not efficacy |
| 12869014 shared cache | M1 | Full TRAIN/VAL pooled and token cache | FAILED 1:0, 02:32:31 | 5ndf max probability mismatch; no completed manifest |
| 12869016_0 baseline controls | M2 | stats/protein/ligand quality heads | CANCELLED dependency, never ran | Repair and audit export before retry |
| 12869016_1 site_gain_rank | M2 / P0 | Candidate vs original predicted gain | CANCELLED dependency, never ran | Paired-feature absolute control still required |
| 12869016_2 site_lateinteraction | M2 / P0 | Atom-residue token matching | CANCELLED dependency, never ran | Matched mean-pooling and ligand-blind controls still required |
| 12869016_3 site_boundary_residual | M2 / P1 | Signed local boundary correction | CANCELLED dependency, never ran | Protein-only control and TRAIN-fitted scorer still required |
| 12869006 train_multisite_union_audit | M0-M2 / P1 | Protein-centric label coverage | COMPLETED, 00:09:28; 10317/10317 | 633 heuristic multisite groups / 1421 rows; not certified negatives |
| 12918127 probability repeat diagnosis | M0 repair | 5ndf/4k7o, BF16 vs FP32 vs FP32 GVP scatter accumulation | COMPLETED 0:0, 00:01:23, 2026-10-05 | BF16 varies ~0.039; FP32 scatter removes repeat drift on both probes; full VAL audit still required |
| 12919239 full precision audit | M0 recovery | 647 VAL rows; legacy vs mixed, mixed repeat, all outliers | FAILED 2:0 after complete audit, 00:08:11 | Sole failure: 2k00 1/92 binary changes; all mixed repeats identical |
| 12919240 new-protocol smoke | M0 recovery | All head/control paths on small new cache | CANCELLED dependency; never ran | New documented acceptance decision required |
| 12919243 unified cache | M1 recovery | Fresh TRAIN and VAL under one numerical protocol | CANCELLED dependency; never ran | Audit-bound inputs/predictions and complete manifest |
| 12919245_[0-3%3] matched heads | M2 recovery | Original quality, gain, tokens, boundary groups | CANCELLED dependency; never ran | No promotion until full VAL comparisons exist |
| 12925293 integer-aware v2 audit | M0 recovery v2 | Fresh 647-row audit, explicit v1/v2 verdicts | FAILED 2:0, 00:10:00, gpu1-41 | 6fac repeat max probability difference 0.00481236; zero binary flips |
| 12925308 v2 smoke | M0 recovery v2 | All head/control paths on fresh cache | CANCELLED dependency; never ran | Localize residual repeat variation before resubmission |
| 12925310 v2 shared cache | M1 recovery v2 | Re-export TRAIN/VAL with version-bound approval | CANCELLED dependency; never ran | Complete manifest required; no old cache reuse |
| 12925312_[0-3%3] matched heads | M2 recovery v2 | Original quality, gain, tokens, boundary groups | CANCELLED dependency; never ran | One GPU/task, maximum three concurrent when retried |
| 12930149 practical v3 audit | M0 recovery v3 | Existing full-VAL approval with practical repeat tolerance | COMPLETED 0:0, 00:07:53 | 647 rows, VAL F1 0.521790, passed=true |
| 12930150 practical smoke | M0 recovery v3 | Existing small all-control execution check | COMPLETED 0:0, 00:02:33 | TRAIN 68 / VAL 38, all paths completed, not efficacy |
| 12930151 practical shared cache | M1 recovery v3 | Full TRAIN/VAL export using same v3 tolerance | COMPLETED 01:26:44 | Complete manifest; TRAIN10317/VAL647 |
| 12930152_[0,1,3] matched heads | M2 recovery v3 | Quality, gain, boundary | COMPLETED | Negligible VAL gains; no promotion |
| 12930152_2 token controls | M2 recovery v3 | Tokens, mean, blind | TIMEOUT 04:00:26 | Tokens/mean 20 epochs completed; blind only 10 epochs |
| 12931196 GDN smoke | Hybrid preflight | FLA numerical/full-model tests | FAILED missing einops, 00:00:34 | Import repaired; no GPU correctness verdict |
| 12931199_[0-1] hybrid scratch | Hybrid trial | FFFF vs GFGF | CANCELLED dependency, never ran | Replaced by 12947232 |
| 12947231 GDN smoke retry | Hybrid preflight | Same smoke with isolated einops0.8.1 | COMPLETED 00:06:17 | GPU numerical and real-batch checks passed |
| 12947232_0 FFFF scratch | Hybrid trial | Matched seed44 control | COMPLETED 04:05:23, early stop e35 | Best VAL F1 0.529855/e33 |
| 12947232_1 GFGF scratch | Hybrid trial | Matched seed44 hybrid | COMPLETED 04:25:08, early stop e35 | Best VAL F1 0.535996/e32; no stable tail gain |
| 12954704_[0-1%2] best-F1 TEST | Hybrid held-out eval | FFFF/GFGF real, shuffle, zero; argmax and fixed VAL gate | COMPLETED 0:0, both647 rows audited | Small real gain, smaller dependency gaps; retain candidate only |
| 12956268_[0-3%2] best-F1 external | Hybrid external eval | FFFF/GFGF x COACH420/HOLO4K per-ligand | COMPLETED0:0, all4 tasks | Small HOLO gain, no COACH gain; multi-ligand F1 not solved |
| successful head seeds43/45 | M3 | Confirm seed sensitivity | CONDITIONAL | Strict VAL gain and count/precision/recall guardrails |
| strict TEST + COACH/HOLO + ligdep | M4 | Confirm frozen protocol | CONDITIONAL | Freeze all selection, masking and calibration rules first |

The four-element GPU array uses one GPU per task and at most three concurrent
tasks. All depend on a passing smoke and complete export, with invalid-dependency
cancellation. 62 local tests previously passed. Only tiny smoke metrics exist;
no full head performance result exists. Queue was empty at the first 2026-10-04
check; the user then approved targeted diagnosis, submitted as 12918127.
Mainline model and tolerances unchanged. Static audit: 66 reference-side model,
data, utility and inference Python files match the newer frozen snapshot exactly.
At the first 2026-10-05 follow-up, the queue was empty. The user then approved
continuation and the 12919239 -> 12919240 -> 12919243 -> 12919245 chain was
submitted. The audit completed all 647 rows on gpu1-25 but exited 2 on the
short-protein binary-count gate. Downstream jobs were cancelled; the queue is
empty again. Mixed VAL F1 0.521790 vs archived 0.521868, repeat probability
differences zero on all 647 rows. No full training success is claimed.
Full FP32 changes predictions materially on the diagnostic samples; do not
substitute it for the BF16 baseline without a matched evaluation. Recovery keeps
BF16 with FP32 GVP reduction and explicit cache protocol labels. The v1 gates
above remain unchanged in the historical record. Subsequently the user approved
v2's uniform `flips <= max(1, n_full // 100)` count budget; other gates, model
weights and classification threshold stay unchanged. Offline re-evaluation of
the 647 saved rows passes v2 and still fails v1. 79 local tests passed. The fresh
GPU audit under `strict_site_precision_v2_20261005` subsequently failed on 6fac
repeatability, not the integer count rule. Its first probability arrays match
the previous audit exactly on all 647 rows; only one repeated forward differs,
without a binary flip. All dependent jobs were cancelled. No tolerance change,
unchanged retry, new submission or full training occurred in that status check.
Subsequently the user approved pragmatic tolerance. v3 explicitly raises only
repeat probability tolerance to 0.01 and keeps binary/aggregate/data guardrails.
80 tests passed. At the subsequent check 12930149/50 completed successfully;
12930151 remains Priority pending and 12930152 awaits it in -ue. Old failed
reports remain untouched. No new model efficacy result is claimed. Qwen-style
hybrid attention is documented separately in
`../docs/qwen_hybrid_attention_feasibility_20261006.md`.

After user approval, a separate protein-context experiment was implemented and
submitted: GPU smoke `12931196` (Priority pending), then scratch seed44 FFFF
control / GFGF hybrid `12931199_[0-1%2]` (afterok dependency). All are single-GPU
`-ue` jobs under `experiments/protein_hybrid_gdn_20261006`, with isolated FLA
dependencies and source snapshot. The assignment-head chain above is untouched.
Eight local tests passed; no GPU smoke or training outcome is available yet.
