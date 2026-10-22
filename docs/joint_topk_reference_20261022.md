# Joint TEST Fixed Top-K Reference Diagnostics, 2026-10-22

## Scope and Execution

CPU-only recomputation of the saved epoch25 predictions from
[joint reference TEST](joint_reference_test_20261021.md).
No new inference, training, checkpoint comparison or threshold fitting.

- Job **13260590**, `i64m512ue`, four CPUs, 16G RAM, no GPU.
- COMPLETED 0:0, 2026-10-22 14:54:42 to 14:55:46 (UTC+8), 64 seconds.
- Analysis time 42.36 seconds; Slurm batch MaxRSS 609,932 KiB.
- Snapshot commit `5ea7839`; archive SHA256
  `6000231fb1f8d56742e6862fa805cec2375e61be659144afe25e621b82703a90`.
- Remote root:
  `/hpc2hdd/home/shuang886/PLC/experiments/joint_topk_reference_20261022`.
- Runner: `scripts/diagnostics/recompute_joint_topk_reference.py`.
- Launcher: `scripts/slurm/run_joint_topk_reference.slurm`.

**TEST remains reference-only. Do not choose K, gates, checkpoints or future
model/experiment directions from these results.** Top-20 was declared the
primary diagnostic before computing results; every other declared budget is
reported, not searched for a deployment winner.

## Frozen Protocol and Checks

The model is V11m13 group_rescompat + GFGF, scratch seed44, epoch25 selected
on VAL F1. Original fixed VAL gate: 0.5535550117492676. Original caps remain
max_seq_len=1800 and max_atoms=256 with alternating-ends ligand trimming.

All three modes cover the same 17,395 ligand rows and 6,539,679 residues,
including 262,533 class0 contact positives. PDBbind contributes 1,649 rows;
BioLiP contributes 15,746 rows. The binary target is class0 under the native
heavy-atom distance <5 A contact definition. Classes1/2 are noncontact.

Top-K means selecting the K highest p(class0) residues **within each saved
protein-ligand row**, not selecting pockets or ligands. K is capped by row
length; fractional budgets use ceil(L/divisor), minimum one. L is the saved,
possibly capped residue axis, not necessarily the full protein length.
No GT contact count is used to choose K.

Exact probability ties use stable saved residue order. Analytic uniform-tie
expected metrics are also archived. All Top-20 rows in all three modes have
zero cutoff ties, so the main result has no tie-breaking ambiguity.

Sample/source/donor identity, label axes and all five input hashes were
checked against the locked reference protocol. Both original controls
(p>=0.5 and the fixed VAL gate) reproduce all pooled and sample-macro
F1/MCC/P/R metrics within absolute 1e-12 in each mode and source. Input
hashes were unchanged after analysis. Stderr is empty.

Pooled metrics combine residue confusion counts. Sample-macro metrics
weight each protein-ligand row equally, **not each protein**. Protein-balanced
metrics cannot be reconstructed from these saved rows alone without a
separate protein-identity mapping. One seed and one shuffle realization are
evaluated; no significance or across-seed claim is made.

## All Declared Budgets

All metrics below are pooled across residues unless explicitly marked macro.

| Rule | Real F1 | Real MCC | Precision | Recall | Real sample-macro F1 | Shuffle F1 | Zero F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-10 | 0.489311 | 0.483169 | 0.613901 | 0.406760 | 0.469718 | 0.412548 | 0.414545 |
| Top-20 (primary) | 0.563148 | 0.548118 | 0.494056 | 0.654706 | 0.535587 | 0.478709 | 0.485921 |
| Top-40 | 0.450689 | 0.474981 | 0.310388 | 0.822449 | 0.432145 | 0.400660 | 0.411823 |
| Top-80 | 0.288853 | 0.347829 | 0.171772 | 0.907212 | 0.281401 | 0.269633 | 0.278298 |
| Top-L/20 (5%) | 0.518316 | 0.499674 | 0.462107 | 0.590093 | 0.502165 | 0.435791 | 0.441733 |
| Top-L/10 (10%) | 0.436328 | 0.452251 | 0.304698 | 0.768185 | 0.454604 | 0.382943 | 0.396125 |
| Top-L/5 (20%) | 0.292186 | 0.345302 | 0.175264 | 0.877768 | 0.330808 | 0.270335 | 0.281754 |
| p >= 0.5 | 0.593724 | 0.577887 | 0.542413 | 0.655758 | 0.553922 | 0.485050 | 0.397009 |
| Fixed VAL gate | 0.595887 | 0.578981 | 0.564499 | 0.630972 | 0.551983 | 0.484521 | 0.336419 |

### Primary Top-20 by Source

| Source | Rows | Real F1 | Real MCC | Real macro F1 | Shuffle F1 | Zero F1 | Real-shuffle | Real-zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 17395 | 0.563148 | 0.548118 | 0.535587 | 0.478709 | 0.485921 | 0.084438 | 0.077227 |
| pdbbind | 1649 | 0.603521 | 0.590569 | 0.595780 | 0.538462 | 0.513409 | 0.065060 | 0.090112 |
| biolip | 15746 | 0.558470 | 0.543152 | 0.529283 | 0.471786 | 0.482735 | 0.086684 | 0.075734 |

### Top-20 Versus Frozen Gate

| Mode | Gate mean predicted count | Gate F1 | Top-20 mean count | Top-20 F1 | Top-20 MCC | Top-20 sample-macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 16.869675 | 0.595887 | 20.000000 | 0.563148 | 0.548118 | 0.535587 |
| shuffle | 16.306007 | 0.484521 | 20.000000 | 0.478709 | 0.458664 | 0.456509 |
| zero | 4.808048 | 0.336419 | 20.000000 | 0.485921 | 0.466304 | 0.466166 |

## Descriptive Findings, Not Selection

1. At equal Top-20 budgets, real exceeds shuffle by 0.084438 pooled F1
   (sample-macro gap 0.079078). Thus the matched-ligand advantage is not
   explained solely by different prediction counts or a fixed threshold.
   It supports ligand-sensitive within-row residue ranking, not proof of
   biologically correct same-protein multi-pocket assignment.

2. The zero intervention yields only 4.81 predicted contacts per row at
   the locked gate, versus 16.87 for real. Fixing both to 20 raises zero
   F1 from 0.336419 to 0.485921 while real F1 changes from 0.595887 to
   0.563148. The real-zero gap consequently changes from 0.259468 to
   0.077227. The gated zero deficit therefore cannot be interpreted as
   pure loss of residue ranking: score level and prediction count matter.
   This is not an exact causal decomposition of calibration and information.

3. Increasing a fixed budget from 10 to 80 raises real recall from 0.406760
   to 0.907212 but lowers precision from 0.613901 to 0.171772. Both too
   few and too many selected residues can hurt F1. A length-based budget
   need not match pocket size; no length/contact scaling law is established.

4. The zero Top-20 score exceeds shuffle overall (0.485921 vs 0.478709),
   but not on the PDBbind subset (0.513409 vs 0.538462). Avoid a universal
   claim that missing ligand information is better than a wrong ligand.
   Zero retains atom counts and edge indices and is not a protein-only
   architecture; shuffled ligands are not certified biological nonbinders.

5. AUPRC/AUROC do not change because the underlying scores do not change.
   Real pooled AUPRC/AUROC remain 0.615422/0.950212. Top-K is a decision
   rule for diagnostic reporting, not a newly trained model or evidence
   of an accuracy improvement.

No training or deployment policy is changed from these TEST observations.

## LiBRe Comparison Boundary

This is the latest PLC **joint internal TEST**, not COACH420/HOLO4K, and
LiBRe has not been evaluated on this same frozen cohort here. Do not compare
its 0.563148 pooled Top-20 F1 against external sample-macro benchmark scores.

The prior [matched ranking report](libre_ranking_diagnostics_20261010.md)
already tested Top-20 on the approved common external cohort:

| Historical external cohort | LiBRe sample F1 | Old PLC sample F1 |
| --- | ---: | ---: |
| COACH420, 491 pairs | 0.4733 | 0.4456 |
| HOLO4K, 7,238 pairs | 0.4044 | 0.3224 |

Those are an older PLC checkpoint and separate datasets, not a measurement
of the current joint model's gap to LiBRe. The historical tie audit did not
remove LiBRe's ranking advantage; public-baseline training overlap and
protocol comparability remain separate limitations.

## Artifacts and Tests

The [full aggregate JSON](results/joint_topk_reference_20261022.json)
contains every budget, mode, source, MCC/P/R, tie expectation, original
ranking metric and input hash. Detailed row CSVs and immutable input
predictions remain on HPC2; they are not added to git.

Before submission, 21 targeted tests passed, including sklearn metric
parity, fixed-budget rounding, exhaustive tie expectation, no input mutation,
reference-control reproduction and overwrite refusal. Slurm shell syntax
passed. Full-population control and hash checks passed in the completed job.
