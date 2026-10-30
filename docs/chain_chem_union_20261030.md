# Chain And Exact-Chemical Union Pilots (2026-10-30)

## Task Contract

User-approved new task: receptor structure/chain + chemical ligand identity
maps to the union of its observed binding residues. A ligand occurrence is not
a distinct query if another indistinguishable chemical copy is on the same
receptor chain. Different chemical ligands and receptor chains remain separate.

- Identity: fully sanitized native SDF, canonical isomeric SMILES. Preserve
  stereochemistry, charge, tautomer and fragment distinctions. Do not group
  using a three-letter residue/ligand name alone.
- Group scope: existing split, source-specific preparation, PDB, receptor chain,
  exact residue axis and CA geometry. No cross-structure/coordinate-frame merge.
- Unknown chemistry: explicit unresolved singleton, retained and counted.
- Labels: minimum native heavy-atom distance over the observed copies; contact
  <5A, noncontact-near <8A, far otherwise. Distance auxiliary supervision uses
  that same minimum, not the original representative-copy distances.
- Ligand input: one deterministic representative, lexicographic sample ID,
  independent of labels or predictions. Do not concatenate atom copies.
- Extract each complete chain BEFORE applying the1800-residue per-chain cap.
  Training ligand cap256 remains unchanged; custom HOLO has max_atoms=-1.
- Keep the original strict chain-SI30 connected-component membership. No sample
  moves between TRAIN/VAL/TEST. Preserve upstream TRAIN-only normalization.
- Apply the former >=3-contact supervision eligibility AFTER union/capping.
  Retain all zero/low-contact chain queries in the audit manifest.
- No duplicated224GB feature tree: the dataset is a view over existing caches.
  Cached ESM/V4 descriptors still carry their original extraction context;
  this is not an isolated-chain feature re-extraction experiment.
- This is a versioned NEW task: chain_exactchem_union_fixed5_v1. Old labels,
  checkpoints, snapshots, and benchmark results are not overwritten.

Union covers available QC-approved observations only; it cannot invent absent
ligand copies/annotations. Cross-source preparations remain conservatively
separate. Report unresolved chemistry and nonmerged geometry cases explicitly.
A metric increase after relabelling is NOT a model improvement claim.

## Three Approved Directions

1. Frozen G0 TRAIN/VAL gap:4096 deterministic source-proportional TRAIN pairs
   and ALL17395 strict VAL pairs. Same eval-mode model, BF16, input caps and
   stored VAL gate; no dropout/NMA. Report F1/MCC at0.5 and fixed VAL gate,
   AUPRC/AUROC, Top20/Top-L, plain three-class NLL, source/PDB balancing and
   zero-contact-chain false positives. Use the ORIGINAL G0 source snapshot.
2. Full-versus-chain context: recover the completed OLD joint e25 VAL result
   and repeat on frozen G0 e52 inside the above diagnostic. Same original
   residue axis and global Top20 budget; all chains inferred and stitched.
3. Three matched seed44 G0 scratch runs on the NEW union target:

| Run | Weight Decay | Model Dropout | Other Changes Relative To New-Task Baseline |
| --- | ---: | ---: | --- |
| union_base | .05 | .15 | none |
| union_wd010 | .10 | .15 | none |
| union_drop020 | .05 | .20 | none |

All retain the G0 width256 architecture, local54/group-rescompat/coattention,
gradient protection and current losses. No resume/freeze/compile. Batch16,
gradient checkpointing, VAL F1 early stopping patience30. Save best-F1 and
best finite-loss checkpoints only. Compare against union_base, not an old
per-copy score. Additional seeds are a later confirmation, not yet submitted.

## Recovered Context Evidence

The completed old-checkpoint strict VAL diagnostic does NOT show a universal
benefit of dropping other chains: PDBbind Top20 sample F1 .602390 -> .486098,
while already-chain BioLiP .544362 -> .544342. Multichain PDBbind drops more.
See joint_val_chain_context_20261023.md for verified hashes and strata.

Therefore do not enforce unconditional full/chain consistency, equate contact
on multiple chains with a validated interface, or claim regularization fixes
the external generalization gap before new VAL evidence is available.

## Custom HOLO4K

Use scripts/benchmark/eval_chain_chem_union.py, not the legacy per-copy runner.
CPU preparation verifies key/geometry alignment, creates the same chain/chem
union task and retains ALL chain queries. Any failed parent is recorded and
blocks inference until explicitly reviewed; no silent exclusions.

Inference reports all-query and >=3-contact conditional strata separately,
plus zero-contact false-positive counts; F1/MCC at0.5 and saved VAL gate,
AUPRC/AUROC, fixed Top20/Top-L, sample and protein-balanced means, and contact
count distributions. Labels never choose inference chains or representative
ligand input. The old single-copy-center BP/LAA is NOT reused after multi-site
union. Their old definitions are not valid for this new target.

HOLO has no new SI30 isolation certificate. It remains reference-only, not a
source of checkpoint, threshold, regularization or architecture selection.
Keep published/legacy HOLO numbers separately labelled; the custom protocol
cannot be compared numerically as though it were the same benchmark.

## Execution Gates

1. Local union/identity/axis/metric regression tests.
2. Immutable code snapshot, CPU joint union preparation and parent-loader QC.
3. Each training arm must pass real-data production-loss backward, cap-sized
   stress, finite-VAL-logit checks and strict state-dict reload before training.
4. Independent frozen-G0 TRAIN/VAL diagnostic and custom HOLO CPU preflight.
5. Only later evaluate selected new-task checkpoints; do not use TEST to tune.

Slurm: CPU i64m512u; GPU i64m1tga800u, one GPU and8 CPUs per task. Existing jobs
are not cancelled or modified. Preparation errors block dependent training.

Job IDs and measured manifest coverage will be appended after submission.
