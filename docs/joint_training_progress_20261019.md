# Expanded Joint GFGF Training: Interim Check

Completed update, October21: job13187469 finished34 epochs on October20
15:31:33 (40h01m01s, exit0:0). Best F1=.588257 at epoch25, AUPRC=.625655,
MCC@0.5=.589897; last F1=.582968, tail5=.581594. See the
[completed VAL results](joint_global_si30_results_20261021.md) for checkpoint
metrics, interpretation and coverage caveats. Earlier snapshots below are
historical, not the current job status.

## Policy for Subsequent Launches

User instruction after this inspection: stop on VAL F1_c0, not validation loss.
New shared V11m3/V11m4/V11m13-family launches default to
`training.early_stopping_metric: f1_c0`; the internal GFGF preparation now writes
this explicitly. Patience remains30 unless configured otherwise. This is the
legacy class0 F1 used to select best_f1_c0_model.pt, not the gate-tuned F1.

Only two metric-best checkpoints are saved: `best_model.pt` for the lowest
finite validation loss and `best_f1_c0_model.pt` for the highest finite F1_c0.
No separate Dice/AUPRC/AUROC best weights are saved. AUPRC/AUROC/F1/MCC, gate
and Top-L metrics still remain in TB/checkpoint metadata. Latest recovery and
configured periodic snapshots are separate from best-checkpoint selection.
Matching early-stop state resumes exactly; legacy/loss-monitored state does
not donate its patience counter to F1. A NaN loss does not stop a run whose F1
continues to improve, nor overwrite the existing best-loss checkpoint.

The existing13187469 process and immutable source/config were NOT modified;
its loss-based early-stop ETA below still applies. Previously written weights
were not deleted. Subsequent runs have a different stopping policy and should
not be described as identical training schedules to the historical baseline.

## Latest Check: 2026-10-19 21:21 UTC+8

Training13187469 is RUNNING on gpu1-39;18 epochs validated, epoch19 running.
The reporting convention from this check onward includes both global MCC at0.5
and MCC at the VAL-F1-selected gate, alongside legacy F1_c0, global F1, AUPRC
and AUROC. The gate is selected for F1, not independently optimized for MCC.
These are VAL results, not TEST results.

| Checkpoint epoch | Legacy F1_c0 | AUPRC | AUROC | Global F1 at0.5 | MCC at0.5 | Global F1 at VAL gate | MCC at VAL gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | .5784 | .6132 | .9548 | .5990 | .5829 | .6002 | .5833 |
| 17 | .5842 | .6260 | .9538 | .6062 | .5903 | .6085 | .5920 |
| 18 | .5879 | .6263 | .9556 | .6047 | .5886 | .6052 | .5886 |

Best legacy-F1 and AUPRC checkpoints are now epoch18; best AUROC remains
epoch10 (.9566). Global F1/MCC peak at epoch17 in this snapshot. Do not mix
independently selected maxima into a single checkpoint result.

For context, the exact-recipe PDBbind-only run's best-F1 checkpoint (epoch36)
had legacy F1=.5661, AUPRC=.5874, global F1/MCC at0.5=.5684/.5566, and
global F1/MCC at its VAL gate=.5697/.5578. These are DIFFERENT holdouts;
the numerical increase does not establish a controlled BioLiP benefit.

### ETA and Existing Stop Rule

Recent five completed epoch intervals have median4168.81 seconds (69.48min),
including validation/checkpoint overhead. The configured120 epochs are a cap,
not the likely completion time: early stopping still monitors VAL LOSS with
patience30. Its minimum remains epoch4; after epoch18 the counter is14.
If no lower validation loss occurs, training stops after epoch34, around
2026-10-20 15:45-16:15 UTC+8 (about18.5h after this inspection).
If the counter resets and all120 epochs run, approximately118h remain,
finishing around October24 evening at the present speed.
The7-day Slurm walltime expires October25 23:30:32.

Epoch15/16 validation losses were NaN; epoch17/18 are finite. The user notes
this has occurred before and requests performance reporting to focus on
F1/AUPRC/MCC rather than treating this alone as model failure. This does NOT
change the code's loss-based early stopping rule. No stop rule was changed.
All15,725 logged total/ligand/cross-attention/local-delta gradient observations
were finite; the latest200 total-gradient maximum was2.7319 before clipping.

### GPU Memory and Batch Size

Read-only sampling used srun --jobid=13187469 --overlap --exact within the
existing allocation, not nvidia-smi on the login node. Twenty samples at3s
intervals,21:20:04-21:21:02, show:

- NVIDIA A800-SXM4-80GB, memory.total81920MiB.
- Device memory.used58500MiB (57.13GiB) throughout the observation.
- Device memory.free22655MiB (22.12GiB).
- Training process1650125 reports58478MiB (57.11GiB).
- GPU utilization48-94%; these are sampled readings, not an epoch average.

This is a maximum within a one-minute sampling window, NOT the historical
training peak or PyTorch max_memory_allocated. Slurm TRES has no gpumem
accounting, TensorBoard has no memory tags, and the training loop does not
record CUDA allocator peaks. Its smoke-test9.05GiB is not a production peak.
Allocator caching can make device occupancy constant despite varying active
tensor memory, so extrapolation from these numbers is not a safety proof.

Current batch16, max_seq_len1800, max_atoms256. A small batch18 probe is
reasonable; batch20 needs separate long-protein/large-ligand forward/backward
and optimizer-step stress testing, recording allocated/reserved peaks and
samples/sec. Do not assume batch24 fits. Increasing batch also changes padding
and updates per epoch. No training restart, new GPU job, batch-size edit,
optimizer/LR change, or allocator instrumentation was performed here.

## Earlier Snapshot: 2026-10-19 13:30 UTC+8

Verified on HPC at2026-10-19 13:30 UTC+8. QC13187467 and smoke13187468
COMPLETED. Training13187469 RUNNING on gpu1-39 after14h; epoch12 in progress,
11 validation epochs completed. No TEST result exists in this inspection.

| Epoch | Train loss | Val loss | Val F1_c0 | Global AUPRC | Global AUROC | Global F1 at0.5 | Global F1 at VAL gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | .5211 | .6439 | .4614 | .4543 | .9192 | .4842 | .4851 |
| 4 | .2720 | .5691 | .5497 | .5901 | .9488 | .5807 | .5818 |
| 5 | .2508 | .5730 | .5583 | .5975 | .9495 | .5858 | .5858 |
| 8 | .1826 | .5952 | .5597 | .6002 | .9520 | .5860 | .5899 |
| 9 | .1741 | .6496 | .5730 | .6030 | .9527 | .5919 | .5931 |
| 10 | .1657 | .6016 | .5713 | .6113 | .9566 | .5919 | .5940 |
| 11 | .1588 | .6562 | .5697 | .6051 | .9498 | .5959 | .5965 |

Best-F1 checkpoint:epoch9; best-AUPRC and best-AUROC:epoch10.
These are different metric definitions, not interchangeable F1 estimates.
Epoch11 global MCC at0.5=.5788; at VAL-selected gate=.5794. Gate=.5235;
global precision/recall at0.5=.5714/.6225, at gate=.5850/.6084.
Legacy class0 precision/recall=.5278/.6304. Calibration gain is small here.

Epoch11 macro top-L precision/recall:
- L/20:.5244/.5578.
- L/10:.3731/.7325.
- L/5:.2266/.8522.

TB snapshot through step98897:9,881 gradient samples, no non-finite values
in inspected gradient/health tags. Last200 logged observations:
- Total grad norm mean .9964, maximum12.9997 (spikes exist; no sustained divergence established).
- Ligand encoder grad norm mean .3757; cross-attention .1656; local delta head .1321.
- CNC clip fraction mean .01293. Its5-epoch warmup means the loss drop after
  epoch5 is not solely evidence of improved fitting under an unchanged loss.
- Local delta absolute logits mean6.1977; local54 compatibility alpha last.04752
  versus initialization.02. Active pathways do not prove ligand specificity.

Interpretation: strong early learning, followed by slower ranking/F1 improvements
and a widening train/validation loss gap. Val loss minimum occurs at epoch4,
while F1/ranking improve later. This suggests monitoring confidence and
generalization, not declaring failure or changing the running recipe now.
No claim that added BioLiP improves performance is supported yet: this is a new
mixed-source holdout, about90% BioLiP rows, not the old PDBbind-only holdout.
Final evaluation should report sources separately and real/shuffle/zero gaps.

Approximate epoch throughput is1.2h including validation;120 epochs could take
about6 days total if speed persists, but early stopping can finish sooner.
No new jobs, weights or hyperparameters were changed during this inspection.
