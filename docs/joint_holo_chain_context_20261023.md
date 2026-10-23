# HOLO4K chain-context diagnostic (2026-10-23)

## Question and Frozen Protocol

Does the latest joint model lose accuracy when whole multi-chain receptor
context differs from the chain-sliced BioLiP training inputs?

This is a reference-only input-context diagnostic. It does not train, select a
checkpoint, tune a threshold, or establish external SI30 separation. Keep joint
GFGF seed44 VAL-best epoch25, TRAIN-only normalization, native heavy-atom labels
at <5 A, VAL gate 0.5535550117492676, BF16, ligand max_atoms=-1, and the original
1800-residue cap. No new residues are added beyond the parent evaluation axis.

For every original protein-ligand row, infer **every** observed protein chain
with the same ligand, including chains with zero contact labels. Split by exact
residue chain keys, not ligand chain names or GT contact masks. Rebuild the
downstream GVP graph and attention context per chain. Existing ESM and receptor
V4 descriptors are sliced, not recomputed: receptor-derived SASA/neighborhood
features can still contain full-receptor context. This is not fully isolated
chain feature extraction.

## Comparisons

1. Primary: stitch chain scores onto the original residue axis; use global
   per-pair Top20 and the frozen VAL gate. Compare original full-context PLC,
   chain-context PLC, and existing chainwise released LiBRe predictions.
2. Conditional diagnostic: restrict to GT-contact chains and calculate Top20
   within each chain. Crop original PLC and LiBRe predictions onto those SAME
   axes. This uses oracle binding-chain information, so it is not comparable
   to blind whole-protein localization as a headline metric.
3. Report all zero-contact chains' native-threshold false-positive rate and
   predicted contact counts. They are included during inference and in the
   primary stitched evaluation, never discarded to inflate scores.
4. Check single-chain samples' probability differences as a numerical control;
   BF16 batching/padding can produce small changes even without context changes.

Preserve the parent's four fixed cohorts: all matched, no TRAIN shared PDB,
no TRAIN+VAL shared PDB (primary nonshared cohort), and no internal shared PDB.
The 7 previous LiBRe failures remain excluded from matched comparisons only;
PLC still predicts all 7,245 rows. Report sample-macro and protein-balanced
F1/MCC/P/R, pooled ranking metrics, and tie-aware fixed-budget controls.

Full matched parent Top20 sample F1: PLC 0.354668, LiBRe 0.404402.
No TRAIN+VAL shared-PDB parent Top20 sample F1: PLC 0.306322, LiBRe 0.382451.

## Execution

- Runner: `scripts/benchmark/eval_holo_chain_context.py`.
- Slurm: `scripts/slurm/run_holo_chain_context.slurm`.
- Preparation and summarization: CPU `i64m512ue`.
- Inference: one GPU on `i64m1tga800ue`, chain microbatch 2.
- Remote root: `/hpc2hdd/home/shuang886/PLC/experiments/joint_holo_chain_context_20261023`.
- Immutable `code_v1`, `results/protocol.json`, `chains.json`, `cohorts.json`,
  per-pair NPZ predictions, progress/VRAM statistics, and output SHA256 audit.
- Preparation freezes identifiers and chain axes without selecting by labels.
- Inference verifies labels/coordinates/residue keys against parent predictions.
- Summarization must exactly reproduce parent full-context metrics before
  accepting the new comparison.

Local verification: 67 tests passed, including residue/ligand-axis separation
when atom and residue counts coincide, zero-contact chain retention,
case-sensitive/noncontiguous chain slicing, exact stitching, global budget,
and matched-axis conditional metrics. Slurm syntax check passed.

## Submission

Immutable evaluation commit: `74c6489` (pushed to public `plc/master`).
Archive SHA256: `e7ef6892b08fd19ac8b8aa8d9fd601534292839b9df97b53a09c4c10cd448f99`.
Remote `af3` has no `pytest`; a CPU-only import/split/stitch/conditional-metric
smoke test passed there. The 67-test suite passed locally in `BIO`.

| Phase | Job | Partition | Initial State |
| --- | --- | --- | --- |
| Freeze chain manifest | 13280632 | i64m512ue | Pending/Priority |
| All-chain inference | 13280633 | i64m1tga800ue, one GPU | Pending/afterok:13280632 |
| Matched summary | 13280634 | i64m512ue | Pending/afterok:13280633 |

All phases use independent locks. Invalid dependencies cancel downstream
jobs; no expensive inference proceeds after failed preparation. There are
no new chain-context performance results at submission time.

## Completed Results

All three jobs completed with exit0:0 on 2026-10-23 (UTC+8). Preparation took
1m46s, GPU inference 19m05s, and CPU summarization 6m49s. The final summary was
written at14:44:49. All7,245 PLC pairs and17,604 chain-ligand inputs succeeded;
matched comparisons retain7,238 pairs. Eight very short chains (<3 residues)
were retained, not silently skipped. PyTorch peak allocated/reserved memory
was0.838/1.209 GiB (not total device occupancy).

Collected artifact: [complete summary](results/joint_holo_chain_context_20261023.json).
SHA256: `ff138d388f729c985e126ba90e0d93917be4324349cd32bf2a24ebbd9ee2a637`.
Protocol SHA256: `d20c13cf509003aa0f7a58df7292a2695d99fd6e22b234c9502a634c8e017ca5`.
The collector verified prediction hashes and identical axes, and exactly
reproduced the original full-context scores before accepting the comparison.

### Primary: Same Axis, Global Top20

All values below are sample-macro metrics on identical protein-ligand rows.
The prediction budget stays20 residues across ALL chains combined.

| Cohort | Pairs | Full PLC F1 / MCC | Chain PLC F1 / MCC | LiBRe F1 / MCC | PLC F1 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| All matched | 7238 | 0.354668 / 0.336147 | 0.377242 / 0.359943 | 0.404402 / 0.390166 | +0.022574 |
| No TRAIN shared PDB | 3780 | 0.306655 / 0.285543 | 0.325776 / 0.305584 | 0.379610 / 0.364792 | +0.019121 |
| No TRAIN+VAL shared PDB | 3215 | 0.306322 / 0.284673 | 0.323659 / 0.302795 | 0.382451 / 0.367583 | +0.017336 |
| No internal shared PDB | 2558 | 0.303579 / 0.281014 | 0.319480 / 0.297575 | 0.383738 / 0.368446 | +0.015901 |

Full-cohort relative Top20 F1 gain is6.36%; the noTRAIN+VAL gain is5.66%.
The observed gap to LiBRe narrows, but remains0.027160 and0.058793 respectively.
Protein-balanced Top20 F1 also improves: full cohort0.478915 ->0.491048
(LiBRe0.527826); noTRAIN+VAL0.406499 ->0.415672 (LiBRe0.498795).
Uniform tie expectations differ negligibly from the reported Top20 results.

### Native Gate, Ranking, And False Positives

PLC gate remains0.5535550117492676; LiBRe retains released FP32 p>0.6.
No benchmark calibration was performed. AP/AUROC below are residue-pooled,
while the first two metric columns are sample-macro.

| Cohort / model | Native F1 / MCC | Native precision / recall | Pooled AP / AUROC | Mean predicted residues |
| --- | ---: | ---: | ---: | ---: |
| All / full PLC | 0.368942 / 0.348798 | 0.367360 / 0.385596 | 0.248721 / 0.838507 | 20.89 |
| All / chain PLC | 0.413344 / 0.411317 | 0.360095 / 0.576847 | 0.251861 / 0.900400 | 41.32 |
| All / LiBRe | 0.386316 / 0.426072 | 0.272061 / 0.845644 | 0.334489 / 0.943814 | 87.94 |
| No TRAIN+VAL / full PLC | 0.316193 / 0.292346 | 0.322069 / 0.324039 | 0.210492 / 0.815991 | 20.85 |
| No TRAIN+VAL / chain PLC | 0.354292 / 0.343215 | 0.314854 / 0.483263 | 0.221954 / 0.872734 | 41.13 |
| No TRAIN+VAL / LiBRe | 0.385165 / 0.420615 | 0.274506 / 0.821591 | 0.336906 / 0.942947 | 88.76 |

The full-cohort native sample F1 now exceeds LiBRe, but this is not an
across-metrics win: Top20, MCC, AP, AUROC and the nonshared native scores still
favor LiBRe. Full-cohort residue-pooled native PLC F1 slightly falls
0.368300 ->0.365131, despite the sample-macro improvement. Do not interchange
pooled and sample-macro F1 in claims.

Zero-contact-chain residue false-positive rate rises from2.225% to6.471%
(LiBRe14.150%) across8,217 matched chain-ligand rows. Mean positive predictions
per such chain rise5.60 ->16.29. The same pattern holds without TRAIN/VAL
shared accessions:2.189% ->6.209%,6.01 ->17.04 predicted residues. These
chains stayed in inference and primary metrics; their increase is not hidden.

#### Fixed-Threshold MCC Supplement

At the user's request, future result summaries must explicitly pair F1 with
MCC at both0.5 and the previously frozen VAL gate, separating sample-macro
and residue-pooled values. Do not substitute Top20 MCC for threshold MCC.
PLC native threshold here is0.5535550117492676; LiBRe native isFP32 p>0.6.
Neither threshold was fitted on HOLO4K.

| Cohort / model | MCC@0.5 macro | MCC@0.5 pooled | Native MCC macro | Native MCC pooled |
| --- | ---: | ---: | ---: | ---: |
| All / full PLC | 0.349784 | 0.349958 | 0.348798 | 0.348596 |
| All / chain PLC | 0.412944 | 0.362647 | 0.411317 | 0.362769 |
| All / LiBRe | 0.418355 | 0.386683 | 0.426072 | 0.395072 |
| No TRAIN+VAL / full PLC | 0.295452 | 0.294444 | 0.292346 | 0.291452 |
| No TRAIN+VAL / chain PLC | 0.347172 | 0.306796 | 0.343215 | 0.304763 |
| No TRAIN+VAL / LiBRe | 0.414631 | 0.393894 | 0.420615 | 0.401678 |

### Conditional On Known Binding Chain

These are oracle-chain diagnostics, not blind localization. The original
full-context scores are cropped to exactly the same GT-positive chain axes.
Top20 is now per positive chain; sample units and budgets differ from primary.

| Cohort | Positive chain-ligand rows | Cropped full PLC Top20 F1 | Chain PLC Top20 F1 | LiBRe Top20 F1 |
| --- | ---: | ---: | ---: | ---: |
| All matched | 9379 | 0.411405 | 0.463550 | 0.508900 |
| No TRAIN+VAL shared PDB | 4077 | 0.354850 | 0.402815 | 0.488214 |

Within identical positive-chain axes, chain-context inference improves local
ranking, so the gain is not solely removal of irrelevant chains at scoring
time. However, LiBRe still leads even with the correct chain given. Remaining
error cannot be attributed only to choosing the wrong chain.

The2,349 matched single-chain control samples have mean absolute probability
change0.000246624. This is consistent with small BF16 batch/padding numerical
differences, rather than a substantially changed checkpoint/input recipe.

### Interpretation And Next Step

- Supported: inference context matters. Splitting all chains improves fixed
  budget ranking without retraining and without a larger global prediction
  budget. Chain-vs-complex context mismatch is a plausible partial contributor.
- Not supported: a pure assignment fix or a general SOTA claim. Higher recall
  is accompanied by more predicted residues and false positives; pooled AP
  gains are modest, and nonshared/known-chain LiBRe gaps remain.
- Not established: insufficient regularization or excessive model size as
  the root cause. This experiment changes context/graph structure, not model
  capacity, training loss or training regularization.
- Next permissible check is the SAME frozen full-vs-chain inference protocol
  on strict VAL, stratified by chain count/source and single-chain versus
  interface contacts. Any inference policy/calibration decision must use VAL,
  not these HOLO or internal TEST results. No further jobs launched here.

This is one frozen checkpoint, not a multi-seed or uncertainty-qualified
architecture comparison. Exact-accession filtering is not sequence isolation;
cached full-receptor descriptors remain in the chain inputs.
