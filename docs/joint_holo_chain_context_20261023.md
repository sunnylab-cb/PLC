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
