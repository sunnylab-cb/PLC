# PDBbind-only global-SI GFGF: completed training, VAL results

Checked October 15, 2026, approximately 13:07 UTC+8 using Slurm accounting,
training stdout/stderr, the complete TensorBoard scalar stream and checkpoint
metadata loaded on CPU. No new training, TEST inference or joint processing.

## Completion And Protocol

| Job | Stage | State | Elapsed |
| --- | --- | --- | --- |
| 13036932 | SI audit, preprocessing and loader QC | COMPLETED, 0:0 | 00:46:31 |
| 13036933 | GPU smoke | COMPLETED, 0:0 | 00:07:26 |
| 13036934 | Scratch GFGF/V11m13 seed44 | COMPLETED, 0:0 | 06:36:30 |

Training ran October 14, 17:00:35 to 23:37:05 UTC+8 on an A800. It stopped
normally after epoch36 because VAL loss failed to improve for 30 epochs;
there was no OOM, time limit or numerical-error termination. All 84,784,118
parameters were trainable, with random initialization and no torch.compile.

The split is certified under the approved **global** SI protocol:
15,214 TRAIN / 1,902 VAL / 1,902 TEST, 1,865 connected components,
194,724 global-threshold edges. Fresh cross-set recalculation covered all
18,801,387 pairs with zero global SI >=0.30 violations.

| Set pair | Recomputed pairs | Maximum global SI |
| --- | ---: | ---: |
| TRAIN / VAL | 8,650,260 | 0.2972972973 |
| TRAIN / TEST | 8,608,572 | 0.2992957746 |
| VAL / TEST | 1,542,555 | 0.2980132450 |

Local SI remains diagnostic: 92,432 covered-local-only matches and maximum
covered local SI=1.0. Thus this does not certify domain/local-SI independence.
Labels are native heavy-atom d_min <5A, with >=3 positives per pair. Separate
TRAIN-only feature statistics and exact production-loader coverage passed.
COACH420/HOLO4K are not excluded by this protocol.

## Checkpoint Results

All entries below are **VAL**, not TEST. Epochs are one-based; checkpoint
`epoch` and TensorBoard steps are zero-based. Legacy F1 is the existing
batch-aggregated argmax class0 metric; global contact F1 pools residue counts
and thresholds p(class0). They are distinct metrics, not calibration deltas.

| Selector | Epoch | Legacy F1_c0 | AUPRC | AUROC | Global F1@0.5 | Global F1@VAL gate | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best F1 | 36 | 0.566083 | 0.587411 | 0.956024 | 0.568376 | 0.569704 | 0.408461 |
| Best AUPRC | 27 | 0.565277 | 0.588037 | 0.957647 | 0.566615 | 0.567875 | 0.381138 |
| Best AUROC | 19 | 0.561493 | 0.582475 | 0.958934 | 0.565255 | 0.565432 | 0.472797 |
| Best loss | 6 | 0.471348 | 0.488462 | 0.938444 | 0.500189 | 0.504134 | 0.536235 |

At the best-F1 checkpoint, legacy precision/recall=0.564400/0.570179.
Global MCC@0.5=0.556585; MCC@VAL gate=0.557810. The actual gate gain in global
F1 is only **0.001328**, not the difference against legacy F1. Gate fitting
and reporting use the same VAL data; this is not held-out calibration evidence.

At epoch36, per-sample macro precision/recall among the top-ranked residues:

| Budget | Precision | Recall |
| --- | ---: | ---: |
| L/20 (5%) | 0.458509 | 0.642764 |
| L/10 (10%) | 0.308912 | 0.782813 |
| L/5 (20%) | 0.185665 | 0.882157 |

## Training Trend And Health

| Epoch | TRAIN loss | VAL loss | Legacy VAL F1 |
| --- | ---: | ---: | ---: |
| 1 | 0.636428 | 0.614743 | 0.371179 |
| 6 | 0.218131 | 0.559847 | 0.471348 |
| 12 | 0.133271 | 0.685102 | 0.543903 |
| 19 | 0.089330 | 0.814353 | 0.561493 |
| 27 | 0.063433 | 1.013264 | 0.565277 |
| 36 | 0.048889 | 1.055166 | 0.566083 |

F1 largely plateaus after epoch19, with late fluctuations around 0.55-0.566.
Tail5 mean F1=0.557720. Epoch36 improves on epoch27 by only 0.000807;
its AUPRC is slightly below epoch27. A last-epoch record is not evidence of
continued large gains, but loss-based stopping also does not establish F1
convergence. The LR is still about 8.33e-5 at stop under the 120-epoch cosine
schedule; no lower-LR tail was completed.

VAL focal loss rises from 0.564761 (epoch6) to 1.323865 (epoch36), while
VAL Tversky loss improves from 0.548382 to 0.428200. This is consistent with
increasing confidence in remaining errors despite better contact decisions,
not evidence of exploding optimization. Confirming calibration/overfitting
would require prediction-level analysis. TRAIN and VAL loss paths are not
identical (augmentation/CNC/auxiliaries), so the loss ratio is not a matched
TRAIN-vs-VAL F1 gap.

All 3,420 logged total-gradient samples are finite. Norms below are logged
before clipping; the configured clip is 1.0.

| Gradient path | Overall median | Last100 median | Maximum |
| --- | ---: | ---: | ---: |
| Total | 0.847700 | 0.585094 | 23.577728 |
| Protein encoder | 0.408122 | 0.286350 | 11.020245 |
| Ligand encoder | 0.155926 | 0.193890 | 3.688280 |
| Cross/co-attention | 0.080188 | 0.066708 | 1.699210 |
| Local delta head | 0.104804 | 0.061852 | 1.162785 |

There are finite spikes, not a sustained divergence in the logged tail. The
ligand encoder has nonzero gradients; this establishes trainability, not
correct ligand dependence. Last100 medians: local-delta absolute logits=6.4375,
compatibility alpha=0.030164, group-KV alpha=0.027665. Neither alpha saturates
its configured bound. Larger delta logits are not themselves a performance claim.

## Interpretation And Next Evaluation

This establishes a working, independently audited PDBbind-only scratch baseline.
Do not rank its 0.5661 against earlier 0.536/0.6158 figures: the holdouts,
normalization and/or metric protocols differ. There is only one seed and no
TEST ligand-dependency result for this run.

Recommended next action, **not submitted in this status check**: evaluate
`best_f1_c0_model.pt` (epoch36, primary) and `best_auprc_c0_model.pt` (epoch27,
secondary selector sensitivity) on this frozen TEST: real/shuffle/zero, each
with its own VAL-fixed gate. Do not tune the gate on TEST or re-tune it per
perturbation; do not select the headline checkpoint by TEST performance.
No loss/early-stop changes or resumption were made.

Run directory under
`/hpc2hdd/home/shuang886/PLC/experiments/pdbbind_global_si30_gfgf_20261014/outputs/`:
`pdbbind_global_si30_GFGF_scratch_seed44_v11m13_20261014_170110`.
TensorBoard is in `runs/`; job stdout is `logs/train_13036934.out` under the
experiment root. The explicit best checkpoints were checked against their
own `val_metrics`, not historical `best_val_*` fields copied from other epochs.
`checkpoint_latest.pt` is saved before the current selector counters update,
so those historical fields can lag one epoch; its current `val_metrics` agree
with epoch36. Prefer explicit named best checkpoints for the next evaluation.
