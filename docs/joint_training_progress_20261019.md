# Expanded Joint GFGF Training: Interim Check

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
