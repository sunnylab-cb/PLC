# Self-distillation, background shortcuts and context noise

Date: 2026-10-12. Read-only model/loss audit and literature review, with a
proposed noise-strength experiment. No new model code, training job or
student TEST evaluation was produced by this analysis.

Subsequent user-approved implementation is recorded separately in
[the original-model noise execution ledger](context_noise_execution_20261012.md).
It does not use self-distillation; the review below describes the motivating
evidence available before that experiment.

## What the current experiment establishes

All three results are seed44 strict VAL, selected by the existing best-F1
rule. They are not HOLO4K results and are not independent-seed estimates.

| Run | Best contact F1 | AUPRC at that checkpoint | VAL-fitted global F1 |
| --- | ---: | ---: | ---: |
| GFGF, no KD | .535996 | .546287 | .539223 |
| KD T2, lambda .05 | .530556 | .543971 | .539569 |
| KD T2, lambda .10 | .529393 | .538321 | .532087 |

The .05 arm's raw T-squared KL falls from .92270/e6 to .12315/e35, without
a best-F1 improvement. Its precision rises .01193 and recall falls .02175.
Its threshold-fitted difference of +.000346 is approximately a tie, not
held-out evidence of improvement. Larger KD also increases pre-clip gradient
norms, but recorded scalars remain finite. See the
[completed result report](self_distillation_results_20261012.md).

The frozen reference's matched eval-mode TRAIN/VAL contact F1 is
.919765/.535538. This is compatible with copying a teacher that already fits
TRAIN very well; it does not identify background reliance as the cause.
See [the unified evaluation](self_distillation_execution_20261012.md).

### Reward hacking is not established

Reward hacking concerns optimizing a proxy at the expense of the intended
objective. That is a useful analogy, but a failed auxiliary-loss experiment
does not establish a particular exploit. The formal RL treatment does not
diagnose this supervised model. [Skalse et al.](https://arxiv.org/abs/2209.13085)

The inspected `src/losses/self_distillation.py` uses a fixed detached teacher,
TRAIN-only aligned cache, and original hard loss plus KL. Eligibility depends
on sampled ligand-source dropout, not student predictions. Contact labels are
not fed to the model forward. No prediction-controlled route to switch off
the penalty was found. This is a scoped audit, not a proof of every invariant.

The supported description is objective mismatch / unsuccessful transfer;
background shortcut learning remains a hypothesis. Student eval-mode TRAIN,
TEST ligand-dependency and background interventions have not been measured.
All runs stopped at e35 under VAL-loss patience, not proven F1 convergence.

### Background output loss is not background input reliance

From e35 sampled TB batches:

| True class | Eligible residue count share | KL share, .05 | KL share, .10 |
| --- | ---: | ---: | ---: |
| Contact | 3.57% | 11.19% | 12.78% |
| Near | 5.11% | 17.54% | 18.98% |
| Far/background | 91.32% | 71.27% | 68.24% |

Counts are raw eligible residues, but KL terms have inverse-row-length
weighting. These are descriptive shares with different weighting, not exact
class-conditional KL means. Do not divide them and claim a per-residue ratio.
Most KL appearing at background output positions cannot establish that the
model uses background input features to locate a pocket.

Local-delta attention in `src/models/pocket_gvp_fusion_v11m10.py` normalizes
over ligand atoms separately for each residue. Increasing attention to atoms
is not equivalent to selecting the correct residue pocket. Also, V11m4's
`base_logits` already includes upstream co-attention: it is not a measured
ligand-independent baseline.

## Literature and transfer limits

The exact Fei-Fei Li paper described in the conversation was not identified.
The primary sources below support related mechanisms, but should not be
attributed to her or described as one identical ViT method.

| Work | Verified mechanism | Relevance and limit |
| --- | --- | --- |
| [Noise or Signal, ICLR 2021](https://arxiv.org/abs/2006.09994) | Foreground/background manipulations expose background dependence in image classifiers. | Supports an intervention-based diagnostic, not a diagnosis from attention maps or KL mass. |
| [Background Augmentations, 2021 technical report](https://arxiv.org/abs/2103.12719) | Background removal/replacement and paired views discourage spurious context in visual self-supervision. | Closest intervention precedent; image-level foreground masks are not dense residue labels. |
| [Focal and Global Distillation, CVPR 2022](https://arxiv.org/abs/2111.11837) | Separates foreground/background feature distillation and preserves global relationships. | Motivates region-aware KD as a separate arm; not a prescription to discard all background. |
| [Vision Transformers Need Registers, ICLR 2024](https://arxiv.org/abs/2309.16588) | Some high-norm background tokens act as internal computation storage; extra tokens reduce artifacts. | A background-position hidden state can carry useful global information. This is not background noise augmentation. |
| [Right for the Right Reasons, IJCAI 2017](https://www.ijcai.org/proceedings/2017/371) | Input-gradient constraints guide models away from annotated irrelevant features. | Possible later sensitivity regularizer; double-backward support and cost must first be checked for the GDN/FLA implementation. |

Visual results do not predict a numerical gain in protein localization.
Proteins have structural coupling; another pocket or an allosteric residue is
not automatically irrelevant background. Background augmentation can improve
general pocket recognition without improving ligand-to-pocket assignment.

## Noise amplitude and noise identity are separate variables

Lowering background noise does not by itself remove a label shortcut. If true
contacts are always clean and other residues are always corrupted, the
corruption signature can reveal the answer. Training-only GT augmentation is
not automatically TEST leakage, but can create a label-conditioned shortcut
and train/inference mismatch. Using GT to prepare TEST inputs would leak.

Proposed first screen: strengths .05/.10/.20 relative to valid hidden-feature
RMS; each gets a matched spatially shuffled-amplitude control. Use a nonzero
foreground noise floor, initially half the background maximum. Randomize
per-example strength, retain mostly clean training examples, and evaluate
only clean full proteins for model selection. These reduce, not prove absence
of, augmentation shortcuts.

Prefer a soft, frozen protein-only candidate-union protection mask, identical
for different ligands of the same protein. Protect all plausible sites and
their neighborhoods rather than only the current ligand's GT contacts.
Such a mask must be audited and cached before implementation; no valid
protein-only mask is claimed to have been produced here. A ligand-conditioned
teacher output or post-co-attention tensor is not automatically protein-only.

The first intervention should be at the protein context encoder input,
leaving ligand chemistry, coordinates, graph edges and residue indexing intact.
This tests one feature path, not deletion of all background information:
ESM/GVP features already mix context and classifier skip paths remain.
Do not begin with coordinate replacement, atom perturbation or global masking.

## What would discriminate the explanations

1. At fixed checkpoints, compare clean, far-context perturbation and matched
   random perturbation. Measure changes at unperturbed pocket residues, not
   only full-protein F1 that directly includes corrupted negatives. A GT-based
   sensitivity mask may be used for this labeled diagnostic only, never as a
   deployable benchmark input or selection trick.
2. Compare targeted and random training noise at equal expected relative
   energy. Both winning suggests generic regularization; targeted alone winning
   motivates a context-specific hypothesis. Spatial clustering is another
   confound: confirm a promising result with matched random spatial patches.
3. Measure clean versus augmented performance, fixed-checkpoint TRAIN/VAL gaps,
   predicted contact counts, input sensitivity and pre-clip parameter gradients.
   Gradient magnitudes must be normalized by group/residue size as appropriate;
   summed background gradients and attention maps are not causal evidence.
4. If a branch wins clean strict VAL, freeze its best-F1 checkpoint and VAL gate,
   then measure TEST real/shuffle/zero and protein-balanced multi-site metrics.
   Do not maximize the gap by damaging shuffle scores while real F1 falls.

Full staged matrix and explicit unsubmitted status:
[experiment plan](../refine-logs/context_noise_20261012/EXPERIMENT_PLAN.md),
[tracker](../refine-logs/context_noise_20261012/EXPERIMENT_TRACKER.md).
