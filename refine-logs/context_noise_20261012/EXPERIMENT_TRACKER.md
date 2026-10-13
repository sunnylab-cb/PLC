# Context-noise experiment tracker

Updated: 2026-10-13. Original-model noise WITHOUT self-distillation.
42 local tests passed. Smoke 12990623 and all six training tasks of 12990624
completed, exit 0. All arms stopped at e35 by unchanged VAL-loss patience.
See [results](../../docs/context_noise_results_20261013.md) and the
[execution ledger](../../docs/context_noise_execution_20261012.md).

| ID | Stage | Variant / purpose | Split | Primary measure | Status |
| --- | --- | --- | --- | --- | --- |
| N0 | Reference | Existing GFGF seed44, no KD or added context noise | Strict VAL | Best F1 .535996 | Existing result |
| M0 | Sanity | Protein-only mask QC and augmentation smoke | TRAIN only | Axis/no-op/RNG/finite checks | 12990623 completed; 128-row QC passed |
| M1 | Diagnostic | Existing-checkpoint background versus random sensitivity | Fixed VAL subset | Unperturbed pocket-output change | Proposed |
| R05 | Pilot | Random amplitudes, sigma .05, rho .5 | Strict TRAIN/VAL | Clean best F1 and AUPRC | 12990624_0 completed; F1 .537633/e32, AP .550313 |
| B05 | Pilot | Background amplitudes, sigma .05, rho .5 | Strict TRAIN/VAL | Same-strength delta vs R05 | 12990624_1 completed; F1 .531978/e33, delta -.005654 |
| R10 | Pilot | Random amplitudes, sigma .10, rho .5 | Strict TRAIN/VAL | Clean best F1 and AUPRC | 12990624_2 completed; F1 .531351/e32, AP .538171 |
| B10 | Pilot | Background amplitudes, sigma .10, rho .5 | Strict TRAIN/VAL | Same-strength delta vs R10 | 12990624_3 completed; F1 .529397/e32, delta -.001955 |
| R20 | Pilot | Random amplitudes, sigma .20, rho .5 | Strict TRAIN/VAL | Clean best F1 and AUPRC | 12990624_4 completed; F1 .531116/e32, AP .539036 |
| B20 | Pilot | Background amplitudes, sigma .20, rho .5 | Strict TRAIN/VAL | Same-strength delta vs R20 | 12990624_5 completed; F1 .532950/e32, delta +.001834 |
| M3 | Confirmation | rho/patch controls; surviving triplet seeds43/45 | Strict TRAIN/VAL | Mean +/- sample std | Conditional, not submitted |
| M4 | Final | Fixed selected checkpoint and VAL gate | TEST + COACH/HOLO | Real/shuffle/zero; F1/LAA/PDR/BP | Conditional, not submitted |

No arm passes the planned +.003 best-F1 screen. Random .05 shows a small
single-seed lead, but global F1 at .5 is nearly unchanged. There is no consistent
targeted-background advantage and no noise-model TEST evidence. This does not
establish reward hacking or causal background-input reliance. No follow-up
jobs submitted. See [plan](EXPERIMENT_PLAN.md).
