# Original GFGF context-noise trial

Date: 2026-10-12. User approved trials on the original model, explicitly
excluding self-distillation. Implementation and submission ledger, not results.

Update 2026-10-13: smoke and all six training tasks completed successfully.
All six stopped at epoch35 under the unchanged VAL-loss patience. Random .05
has best clean VAL F1 .537633 vs reference .535996; no background arm beats
the reference and no arm passes the planned +.003 screen. See the
[completed results and diagnostics](context_noise_results_20261013.md).
Queue descriptions below record the original 2026-10-12 submission check.

## Submitted matrix

| Array index | Run | Noise placement | Relative sigma |
| --- | --- | --- | ---: |
| 0 | gfgf_random_noise05 | Permuted amplitudes | .05 |
| 1 | gfgf_background_noise05 | Soft geometric background | .05 |
| 2 | gfgf_random_noise10 | Permuted amplitudes | .10 |
| 3 | gfgf_background_noise10 | Soft geometric background | .10 |
| 4 | gfgf_random_noise20 | Permuted amplitudes | .20 |
| 5 | gfgf_background_noise20 | Soft geometric background | .20 |

Reference: existing original GFGF seed44, strict VAL best F1 .535996. There is
no new zero-noise training arm; zero-noise behavior is tested as an exact no-op.
All six runs are SCRATCH seed44. They are not initialized from a teacher
or from any checkpoint, and do not load teacher caches or add KL/consistency.

The original Focal/Tversky/CNC/distance losses, source dropout, feature inputs,
normalization, strict split, B16, BF16, LR1e-4, 120-epoch scheduler horizon,
loss-based patience30 and best-F1 checkpoint selection are unchanged. Existing
caps remain 1800 residues / 256 ligand atoms; this is not an atom-budget trial.
No new parameters or changes to the inference architecture/state-dict format.

## Augmentation definition

V4 spatial channels 46/47/48 are the CA-derived CX, packing and shell-density
features. The score is `.5*x46 + .25*x47 + .25*x48`, as in the pre-existing
V11m16 geometric proposer, without attaching its assignment head. Greedy
spatial NMS selects up to eight centers separated by 10 Angstrom. Protection
is the maximum Gaussian membership around these centers, width 8 Angstrom.

This is a **heuristic geometric candidate union**, not a known true pocket.
No target labels, ligand features, native P-L distances or teacher predictions
are used to construct it. Coordinates and feature tensors are not modified.
It is computed on demand from the aligned protein inputs; no new cache/export
is required. Identical protein inputs produce identical protection regardless
of the paired ligand. Candidate quality still needs the queued TRAIN QC.

On 25% of TRAIN batches, add Gaussian hidden-state noise with per-token scale:

```text
amplitude_i = .5 + .5 * (1 - protection_i)
sample_sigma ~ Uniform(.5*sigma, 1.5*sigma)
delta_i = epsilon_i * stopgrad(RMS(hidden_i)) * sample_sigma * amplitude_i
```

Padding receives no noise. Random controls permute the same amplitude values
within valid residues of each protein, preserving the multiset and expected
relative energy. Actual relative/absolute noise energy is logged, not assumed
identical. Both arms draw the same number of private RNG samples; global model
dropout/data RNG is not consumed by augmentation. The other 75% of batches and
all normal VAL/TEST inference receive no added context noise.

The only shared-model edit is a default-off five-line hook just before the
protein context encoder and OUTSIDE its checkpoint call. The existing clean
`b0` and GVP/physchem skip paths remain unchanged. ESM and GVP have already
mixed information, so this tests one context path, not removal of all protein
background evidence. Training labels are used only for post-forward diagnostic
class means, never as augmentation inputs.

## Checks and diagnostics

Local BIO: **42 tests passed** across new context-noise tests, existing transfer
training tests and GDN tests. Python and Slurm syntax checks passed. Checks cover
zero-strength identity and RNG, matched amplitude histograms, linear noise
strength, padding/non-finite safety, rigid geometry invariance, gradient flow,
checkpoint replay without resampling, strict old-class loading and exact CPU
eval equivalence. AST comparison verifies the train loop only adds logging;
all original loss arithmetic stays identical.

An existing full-model CoAttention + checkpoint-replay error was exposed by an
extra test and reproduced with zero noise. This trial keeps the original
`use_gradient_checkpointing=False` and rejects enabling it, rather than
silently changing co-attention. Pure cross-attention checkpoint replay and the
standalone noise-before-checkpoint path pass. No torch.compile is enabled.

GPU smoke is still pending. It will perform a 128-row TRAIN-only mask audit,
three real B16 production-loss optimizer steps per arm with forced augmentation,
finite-gradient checks, zero-strength identity and strict original-class load.
Its clean GPU probability comparison allows the existing .01 repeat budget
for scatter numerical variability; this is not bitwise equivalence evidence.
The 128-row mask gate checks completeness/non-degeneracy, not a claimed pocket
recall level. It records contact/near/far protection means and coverage for
interpretation. No VAL or TEST labels are used for this audit.

New TB tags: `ContextNoise/*` for application frequency, feature/noise RMS,
protection and class-conditional means; and
`Gradients/context_noise_{protein_context_encoder,cross_attention_layers,local_delta_head}_{norm,rms}`.
Parameter gradients are logged after AMP unscale and before clipping, not
interpreted as causal attribution. No extra self-distillation objective or
second-order gradient penalty is present.

## Deployment and queue

Root: `/hpc2hdd/home/shuang886/PLC/experiments/context_noise_gfgf_20261012`.
Snapshot source: `experiments/libre_transfer_20261010/code_v1`, whose default
configuration is the same original GFGF model; NONE of its four probe settings
is enabled by the copied base config. FLA dependencies are reused read-only.

- Smoke job **12990623**, `i64m1tga800ue`: last verified **Pending (Priority)**.
- Training **12990624_[0-5%2]**, same partition: **Pending (Dependency)**.
- One A800, eight CPUs and 100G RAM per task; at most two concurrent training
  tasks. Smoke has a 2h scheduler cap, training 36h; these are not runtime ETAs.
- Array uses `afterok:12990623` and `--kill-on-invalid-dep=yes`; each task also
  verifies a successful matching smoke and frozen config/source hashes.

Deployment checked original model/trainer hashes, then verified removing just
the new hook reproduces the original model-source SHA256. The frozen config,
strict split and aligned labels retain their prior hashes. All eight uploaded
file hashes and exact submission commands are recorded remotely in
`deployment.json`, `prepared/provenance.json` and `{smoke,train}_submission.json`.

Unsolicited remote shell `proxy` messages broke rsync and SFTP. A small tar
archive was streamed over SSH stdin instead, with post-transfer hash checking.
No remote shell configuration, existing experiment or dirty git state was
changed. No dataset/checkpoint downloads or copying to jhspoolers occurred.

## Next interpretation

First inspect TRAIN mask QC, smoke memory and the actual augmentation rate.
Then compare clean strict VAL best F1/AUPRC and P/R at matched epochs. A gain
shared by random controls supports generic regularization, not foreground
selection. Only augmented-view gains do not count. This trial cannot yet claim
reward hacking, background reliance or any F1 improvement.

Fixed-checkpoint sensitivity evaluation, rho contrast controls, spatial-patch
controls, additional seeds and held-out ligand-dependency remain conditional
follow-ups. No noise-model TEST or external benchmark has been run.
