# Context-noise strength and identity controls

Date: 2026-10-12. Subsequently approved and implemented on the ORIGINAL model,
without KD. Local 42 tests passed; smoke 12990623 and conditional training array
12990624_[0-5%2] submitted in -ue. Update 2026-10-13: all completed successfully,
stopping at e35. No arm passes the +.003 primary screen; no expansion submitted.
See [completed results](../../docs/context_noise_results_20261013.md).
The proposal below is retained; implementation uses on-demand protein-only
V4 geometry masks instead of an exported candidate cache. M1 sensitivity is
deferred rather than blocking the approved controlled training screen.
See [execution ledger](../../docs/context_noise_execution_20261012.md).
The historical assignment and LiBRe-transfer plans in the parent are preserved.

## Claim map

| Hypothesis, not result | Required evidence | Counter-explanation |
| --- | --- | --- |
| Conservatively perturbing non-pocket context improves clean residue prediction | Gain over no added noise AND equal-budget random noise on strict VAL, then independent seeds and fixed TEST | Generic noise regularization, seed variation, threshold changes |
| Improvement does not depend on identifying the corruption pattern | Clean inference gains; matched-noise controls; stable results under weaker mask/noise contrast | Noise encodes the training label or an oracle foreground mask |

Main-paper evidence would be clean held-out F1/ranking and ligand dependency.
Mechanism diagnostics and strength curves belong in the appendix. Cut new
encoders, KD weight sweeps, coordinate corruption and new feature extraction
from this experiment. This is not a claim of a novel foundation model.

## Frozen protocol

- GFGF V11m13 group_rescompat, strict merged TRAIN/VAL/TEST 10317/647/647.
- Start all new arms from scratch, seed44 pilot, no resume or frozen trunk.
- Preserve feature normalization, labels, ligand inputs, all existing losses,
  source dropout, batch size, scheduler horizon and early stopping. No KD.
- Existing reference best VAL F1 .535996, e32; stopped e35 by loss patience.
  Do not turn this into a new 35-epoch schedule or alter best-F1 selection.
- Add noise to one protein hidden path before the protein context encoder.
  Do not modify saved features, coordinates, GVP edges, padding, chain IDs,
  residue axes or the late physchem classifier path.
- No torch.compile. One GPU per run; on Star at most one job/one GPU. The
  approved deployment uses HPC -ue, with at most two single-GPU training tasks.

## Block 1: mask and intervention sanity

Use a frozen protein-only geometric/candidate mask with soft boundaries,
representing the union of plausible pockets plus a buffer. The same structure
must have the same mask for every ligand row. Neither contact labels nor the
current ligand may determine this mask. Learned proposal models must not see
VAL/TEST during training; document their TRAIN memorization risk.

First audit mask provenance, train-only tuning, residue alignment and candidate
coverage. The current ligand-conditioned model's predictions are NOT a valid
protein-only cache by default. Implementation uses V4 CA-only channels 46:49,
eight NMS centers and a soft Gaussian union, calculated on demand. It is a
geometric heuristic, not a GT substitute; queued smoke reports TRAIN coverage.

For valid hidden token h_i, define r_i = stopgrad(RMS(h_i)), soft background
score b_i in [0,1], epsilon_i ~ N(0,I), and noise floor rho=.5:

```text
amplitude_i = rho + (1-rho) * b_i
h'_i = h_i + sigma_sample * r_i * amplitude_i * epsilon_i
sigma_sample ~ Uniform(0.5*sigma, 1.5*sigma)
```

Noise is enabled on 25% of TRAIN batches, disabled on the remaining 75% and
on all ordinary evaluation. These are starting engineering settings, not
values established by a paper. Keep the original hard-label objective, without
adding another teacher or consistency loss in this first screen.

For each paired random control, permute the SAME amplitudes over valid tokens
within each protein; padding receives no noise. This matches the amplitude
histogram and expected relative noise energy, not exact realized absolute
energy. Record both. An additional spatial-patch-matched control is necessary
before attributing a gain specifically to background location rather than
the spatial smoothness of the mask.

Use a separate augmentation RNG and preserve initialization/data/dropout
streams as far as supported. Zero strength must be an exact no-op with no
extra RNG draw. Eval must be a no-op even if a cached mask is supplied.
Checkpoint recomputation must reuse noise, not draw a second realization.

## Block 2: compact strength screen

| ID | Selection of noise amplitudes | sigma | rho | Purpose |
| --- | --- | ---: | ---: | --- |
| N0 | None; existing reference | 0 | n/a | No-added-noise baseline |
| R05 | Spatially shuffled mask | .05 | .5 | Low-strength regularization control |
| B05 | Protein-only soft background mask | .05 | .5 | Low-strength targeted perturbation |
| R10 | Spatially shuffled mask | .10 | .5 | Medium-strength regularization control |
| B10 | Protein-only soft background mask | .10 | .5 | Medium-strength targeted perturbation |
| R20 | Spatially shuffled mask | .20 | .5 | High-strength regularization control |
| B20 | Protein-only soft background mask | .20 | .5 | High-strength targeted perturbation |

Primary: clean strict VAL best class0 F1, unchanged aggregation. Secondary:
AUPRC/AUROC, P/R, F1/MCC at .5 and own VAL gate, Top-L budgets, predicted
contact-count distributions. Never compare different metrics as one F1.

A +.003 absolute F1 screen versus N0 is a pragmatic pilot gate, not statistical
significance. Also require advantage over the strength-matched random control
without a material ranking decline before claiming region selection helps.
Repeat a surviving targeted/random/no-noise triplet on seeds43/45; report
mean and sample standard deviation, not only the maximum seed.

## Block 3: separate noise identity from useful suppression

Only after a useful strength emerges, hold its expected relative noise energy
fixed and compare rho=.25, .5 (already screened), and 1.0. Renormalize the
amplitude vector by its valid-token RMS when changing rho, so this measures
contrast rather than silently reducing total augmentation energy. The rho=1
case is uniform token noise. Use these comparisons diagnostically, not to
expand indefinitely on the same VAL set.

Rho=0 is an optional failure-mode control, not the recommended method. Nonzero
floor and random strengths reduce a clean/noisy identity cue but cannot prove
its absence. Never select on GT-masked TEST predictions.

Diagnostics: clean/noisy feature RMS and norm distributions; clean versus
augmented P/R; sensitivity of unperturbed pocket outputs to remote-context
noise; and gradient norms for protein context, co-attention and local delta,
measured after AMP unscale and before clipping. On a few diagnostic batches,
compare clean/noisy gradient cosine, holding other randomness fixed. Do not
enable expensive second-order gradient penalties in this screen.

If only augmented-view F1 improves, reject. If all noise arms improve similarly,
retain the simplest random/uniform regularizer instead of claiming pocket
focus. A strong-noise decline is compatible with destroying useful structural
context, not evidence that more targeted pressure is required.

## Run order and budget

| Stage | Work | Decision gate | Cost basis |
| --- | --- | --- | --- |
| M0 | Mask provenance + CPU tests + real-data GPU smoke | Clean no-op, axes, finite gradients, no recomputation resampling | Profile first; no validated ETA |
| M1 | Existing-checkpoint sensitivity audit | Report pocket-output changes separately from corrupted-position errors | Small fixed VAL subset, no checkpoint selection |
| M2 | Six seed44 training arms | Clean best-F1 and matched random comparisons | About 25 GPU-hours if the historical 4.1h/run duration repeats |
| M3 | rho/patch controls and seeds43/45, only if justified | Gain not explained by generic noise or one seed | Additional budget after M2 |
| M4 | Fixed strict TEST real/shuffle/zero, then external benchmarks | Real F1 retained; protein-balanced assignment/PDR/BP reported | Evaluate selected checkpoints only |

The 25 GPU-hour number extrapolates the early-stopped KD runs, not a measured
runtime of this unimplemented feature. Noise may change stopping time; a full
120-epoch run would cost much more. Candidate preparation, smoke, confirmation
and evaluation are excluded. There are no new foundation-model downloads.

## Risks and final checklist

- [ ] A valid protein-only, ligand-invariant protection mask is available.
- [ ] Noise ablation cannot leak through GT-derived evaluation inputs.
- [ ] Other plausible sites and structural boundaries remain protected.
- [ ] Matched controls, independent RNG and checkpoint replay are tested.
- [ ] Actual and relative perturbation energy and class denominators are logged.
- [ ] Clean strict-VAL gain exists before TEST/benchmark expansion.
- [ ] Ligand dependency is preserved without targeting a maximal zero drop.

Literature, current numeric evidence and code audit:
[background-noise review](../../docs/background_noise_audit_20261012.md).
