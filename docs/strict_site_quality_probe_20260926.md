# Strict TRAIN/VAL site-quality pilot

## Purpose and fixed design

Follow-up to `strict_assignment_results_20260920.md`, authorized 2026-09-26.
Previous hybrid-union oracle F1 was 0.644453 on strict VAL base44, whereas
probability-only selection either collapsed to small fragments or simply retained
the original prediction. The question is whether independent pre-cross features
provide usable selection information beyond the final residue probabilities.

This is a **new head trained from scratch on a frozen backbone**, not a new
full-model scratch run or a resumed optimizer. It does not repeat V11m16's
tiny centered scalar gate. It directly selects a residue mask, with an explicit
original-prediction option and a fixed predicted-quality replacement margin.

| Control | Inputs to the learned head | Question |
| --- | --- | --- |
| Unmodified V11m13 | Original saved contact gate | No-change reference |
| stats_only | 19 label-free candidate/probability/size statistics | Does learned quality fix heuristic size bias? |
| protein_site | Same statistics + pooled pre-cross protein state | Do residue representations add site-quality evidence? |
| ligand_site | Same + independent pre-cross ligand state and projected interactions | Does explicit ligand context add value beyond the other controls? |

`stats_only` and `protein_site` are **not ligand-independent complete systems**:
their candidate pool and probability statistics still come from real-pair V11
predictions. Only the added representation/query path is independent. The
controls also have different parameter counts, which are recorded; a positive
pilot would still need a capacity-matched control.

## Data and provenance

- Strict merged split: TRAIN 10317, VAL 647, TEST 647, SHA256
  `170a4e6364c19662ad922ace7957e1733409dfe146ebdb74b15c95130bb622bc`.
- Frozen base44 checkpoint: epoch 28 V11m13 from run
  `pocket_gvp_fusion_seed44_chain_si30_biolipmerge_scratch_seed44_12222454_1_v11m13_20260826_055836`.
  SHA256 `d8d54dd9d8183f8c8038a656dd0d3937adac325434af7d7ef32783afc3e6e848`.
- Reproduce prepared normalization, conf0, BF16, 256 atoms with alternating-ends
  cropping, 1800-residue model budget and saved contact gate 0.391169935464859.
- Proposal coordinates retain full protein CA axes. Missing model tails have
  zero probability and no hidden contribution; observed fraction is an explicit
  statistic. This is not uncapped model inference.
- Use the source-verified label-axis overlay for ten anomalous TRAIN/VAL rows.
  Source data remain unchanged. Only the head's supervision uses corrected
  contact labels for 2xct/4glx; the frozen checkpoint was trained on legacy labels.
- Check all VAL targets against the audited prior export and verify reproduction
  of its baseline F1. Reject substantial prediction drift, wrong split/cache
  hashes, silent sample filtering, nonfinite data and incomplete exports.
- Only TRAIN/VAL rows are exported. No TEST/COACH420/HOLO4K evaluation or tuning.
- All caches and checkpoints go under HDD `PLC/experiments`, not jhspoolers.

## Features, supervision and selection

The unchanged `hybrid_union` pool from the previous diagnostic averages about
170 masks per sample and contains the original prediction at index zero.
It combines component/NMS/FPS proposals and their intersections with original
predicted boundaries. No GT target is passed to proposal construction.

Each candidate pools `h_prot_global` over observed residues. The ligand query is
masked-mean `h_ligand_context` over atoms, excluding the global token. These are
before cross-attention. A runtime intervention test checks that zeroing ligand
features leaves the protein state unchanged and zeroing ESM leaves the ligand
state unchanged, while the changed modality's own state responds. This is a
dependency smoke test, not a comprehensive model explanation.

No protein-ligand shared coordinate frame, ligand pose, cross-distance, GT
centroid, label count, candidate target F1 or sample ID is an input to the head.
Candidate statistics include size, probability summaries, prediction overlap,
protein-only geometry and observed fraction. Target arrays are separate.

Targets are each mask's actual F1 against TRAIN contact labels. Loss is mean
squared quality error plus 0.1 times a soft-distribution ranking loss
(`softmax(target_F1 / 0.1)`), allowing equally good candidates to share mass.
Candidate losses are averaged within each sample; exact-axis groups receive
inverse-frequency weights. This is quality supervision under the available
annotation, **not proof that an unlabelled pocket is a biological nonbinder**.
No sparse same-protein hard-negative objective is added in this pilot.

Train-only statistics normalization; LayerNorm plus 64D projections for hidden
states; 128D MLP, dropout 0.1. AdamW LR 2e-4, weight decay 1e-3, batch 32,
gradient clipping 1, 20 epochs, seed44. No torch.compile. Full precision suffices
for the small heads; only frozen feature extraction uses BF16.

At inference, choose the highest predicted quality only when it exceeds the
original prediction's quality by **0.02**; otherwise retain the original. Ties
always prefer the original. This does not guarantee true F1 cannot decrease.
No margin/threshold sweep is performed. Select checkpoint by strict VAL
protein-balanced selected F1 at this fixed margin, retaining the earlier epoch
on exact ties. Also report unguarded argmax selection as a diagnostic.

## Evaluation and limitations

Report sample/protein-mean F1, pooled micro F1/P/R, oracle F1, quality MAE,
fraction within 0.02 of oracle, original retained/replaced, improved/worsened
rows, and predicted count p50/p90. Save per-sample chosen candidate indices;
cache stores packed residue indices to reconstruct the actual output masks.
TB records training loss/gradient norm and every validation metric per epoch.

For the ligand head, permute or zero **only its added query**, keeping real-pair
candidates/features fixed. These are head-only query controls, NOT full-model
real/shuffle/zero ligand-dependency measurements. Do not compare those gaps to
the historical ~0.08 full-model gap. Real full-model ligand dependency remains
a later requirement if the pilot warrants follow-up.

Proceed only if real ligand-site selected VAL F1 improves by roughly >=0.003,
beats stats/protein controls, and does not obtain gains via contact-count collapse.
This is a pilot go/no-go criterion, not a promised gain or significance threshold.
One seed and selected VAL bootstrap cannot establish a paper-ready claim.

The frozen backbone has seen the same TRAIN examples: this is in-sample stacking,
not out-of-fold training. TRAIN candidate quality can be more optimistic than
VAL quality. Independent exact-structure bootstrap is descriptive and is not a
sequence-cluster uncertainty estimate. The strict VAL is PDBbind-only, so a gain
does not by itself demonstrate multi-site HOLO4K assignment improvement.

## Deployment and tests

One allocated `i64m1tga800ue` GPU, 4 CPUs, 64G RAM. All three heads are sequential
inside one Slurm allocation. First run `SMOKE=1`: up to 64 TRAIN/32 VAL rows plus
all verified label-overlay rows, 3 epochs per head, actual GPU forward/export.
Full run must follow a successful smoke; failed smoke must not start training.

Remote root: `/hpc2hdd/home/shuang886/PLC/experiments/strict_site_quality_20260926`.
Use a separate snapshot copied from the previously verified diagnostic snapshot
with only the new files added. No git push or edits to the active remote mainline.

Local checks: 52 tests pass, including synthetic end-to-end three-head training,
GT-independent selection, geometry-frame invariance, original fallback, padding,
finite gradients, train-only normalization, cache hash/split checks and the
previous benchmark/label-axis tests. New scripts compile and Slurm syntax passes.

| Stage | Job | State |
| --- | --- | --- |
| Real-data GPU smoke | 12868601 | FAILED, exit 1:0, elapsed 00:01:41; pre-cross independence assertion |
| Full cache and three heads | 12868608 | CANCELLED, elapsed 00:00:00; failed afterok dependency prevented execution |

Both request exactly one GPU; dependency prevents simultaneous GPU usage by
these two jobs. Full work was intended to be 20 epochs per head. No trained-head
result is available, and the synthetic smoke is not scientific performance evidence.

### Runtime failure, checked 2026-09-26

The first frozen forward succeeded. Zeroing `ligand_x` in a subsequent BF16
forward changed `h_prot_global` by max absolute 0.020873308181762695 and changed
`h_ligand_context` by 4.428080081939697. The independence assertion requires
the former to be <=1e-5; it stopped before exporting the full cache. This is
not an OOM, and not evidence that the proposed head fails scientifically.

Static inspection of V11's forward shows protein self-attention produces
`h_prot_global` before the ligand encoder. No direct ligand input to that path
was found in the inspected code. This does NOT resolve runtime independence:
repeated identical inputs, module-boundary clones/hooks, input mutation checks,
and an FP32 comparison are needed to distinguish numerical nondeterminism from
aliasing or a hidden dependency. Do not relax the tolerance blindly or claim
ligand leakage from this one comparison. Preserve the failed snapshot/logs;
any retry must have its own recorded source hashes and output directory.

Parallel follow-up proposals are in `../refine-logs/EXPERIMENT_PLAN.md`.
They were subsequently approved and implemented. New submissions and the
revised precision-separated audit are recorded in
`strict_site_parallel_execution_20260926.md`; real GPU smoke is still pending.

Local/remote deployed source SHA256 matches:

| File | SHA256 |
| --- | --- |
| site_quality_features.py | 2e63aad6134339a1253d61ef1aad3b9c003e4f18ff1272726ebad4fa842d78c7 |
| export_site_quality_cache.py | 14dcdd803bdaa7a8c867c5d1746cf0b52939d427e670f5b020f91ae1fe8f9100 |
| train_site_quality_probe.py | 0f0068743aec07ac6b634b0156aeceacfd782ac5861fcc40e22fac7ae0cecdd3 |
| site_quality_head.py | 164acfc9e04ef0035a66635cfebd7b0c26c0400706e683de4ec201b300ee7c1e |
| run_strict_site_quality_probe.slurm | a16d28630cb95450be20f1b4689a90017f8a9e093622d52c68e5dcf37d39c4b5 |
