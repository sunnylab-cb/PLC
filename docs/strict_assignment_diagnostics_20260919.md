# Strict validation proposals and merged-train negative coverage

## Latest Collection

See `strict_assignment_results_20260920.md`: corrected retries 12804286 and
12804287 both completed successfully. All 10317 TRAIN rows passed QC and both
models exported all 647 strict VAL examples. Conservative valid-negative coverage
is still 179/10317 (1.74%). No single candidate family improves F1, but combining
families gives oracle headroom, not an achieved prediction gain. CPU follow-up
12814298 also completed: shape-preserving hybrid unions plus count-aware scoring
retain the original prediction in 98--99% of cases and do not meaningfully improve
F1; top20 scoring collapses to small fragments. No new training or test search
was launched. All three successful jobs and their detailed tables are recorded
in the results document.

The original failures (not OOM), source-verified label-axis overlays and unchanged
source caches are documented in the results file. All six val mismatches were
extra tails; two train samples had actual internal shifts. Corrections changed
zero VAL protein contact labels and did not alter conservative negative counts.

## Scope

Follow-up to `benchmark_integrity_results_20260919.md`. The old thresholded
contact-component candidate family had essentially no useful oracle gain on
HOLO4K. This does not justify training a reranker on that family. The current
step tests broader proposals on strict VAL and audits supervision on TRAIN.

- Current merged split: train 10317 (PDBbind 5160 + BioLiP 5157), val 647,
  test 647. Split SHA256:
  `170a4e6364c19662ad922ace7957e1733409dfe146ebdb74b15c95130bb622bc`.
- Baseline checkpoint, final merged root and ESM3 root split hashes agree.
- No test/COACH/HOLO labels enter candidate selection or parameter search.
- No new model training, optimizer changes, feature re-normalization or new
  gate fitting. These are frozen-checkpoint diagnostics, not a new model claim.
- Remote isolated root:
  `/hpc2hdd/home/shuang886/PLC/experiments/strict_assignment_20260919`.

Submitted jobs: `12804060` (CPU negative audit), `12804061` (single-GPU VAL).
Both initially pending with reason `Priority`. At the first health check,
CPU job is running on `cpu1-91`, processed 5250/10317 rows at elapsed 3m06s,
and stderr is empty. GPU job remains pending (`Priority`, estimated start N/A).
These are progress observations, not completed-result claims.
Diagnostic source SHA256 values match between local and remote.
Remote `export_strict_val_predictions.py --help` import smoke test passed.

## Strict VAL protocol

Export base44 and ESM3-global-prior44 sequentially, one GPU, BF16, batch size 1.
Use the checkpoint's training config and saved contact gate. Preserve
`max_atoms=256`, `alternating_ends`, sequence budget 1800, original prepared
feature normalization and spatial clamp. The old shared internal-eval builder
does not pass `atom_truncation_policy`; this separate exporter explicitly does.

Export **full protein CA/label axes**, but flag the actual model-observed prefix.
Unobserved residue probabilities are explicitly zero. Pure geometry proposals
still cover the full axis; model-guided proposals are limited by missing model
evidence. Report affected rows and missed true contacts. This is NOT a claim of
uncapped full-protein neural inference. ESM/protein prefix-key shifts and
label/protein length mismatches fail rather than being silently clipped.

Predeclared candidate family matrix, 19 configurations total:

| Family | Construction | Radius / budget |
| --- | --- | --- |
| Original control | Components of final predicted contact mask | 8 A / up to 16 |
| Relaxed controls | Components of p(contact)>=0.1 or >=0.2 | 6 A / up to 16 |
| Probability spatial NMS | Mean contact probability in 8 A CA neighborhoods, spatial seed suppression at 8 A; full CA spheres, not contact-mask intersections | spheres 8/12 A; K=4/8/16/32 |
| Geometry FPS | Farthest-point sampling over all CAs; ligand-independent candidate generation | spheres 8/12 A; K=4/8/16/32 |

All candidates ranked by top-20%-mean frozen contact probability, with stable
tie handling and mask deduplication. FPS construction is protein-only, but its
**ranking** still uses ligand-conditioned model probabilities. FPS is a geometry
coverage control, not a learned biological pocket detector.

Report original F1, score-selected F1, GT-oracle F1, IoU>=0.25/0.5 candidate
coverage, center-distance<4 A coverage, candidate count/size, selected size,
protein coverage fraction and true-contact coverage. The center diagnostic is
distance to the **GT contacting-residue CA centroid**, not standard ligand-center
DCC or nearest-ligand-atom DCA. GT is used only in these measurements, never in
candidate construction/ranking. Sample and exact-structure-protein-balanced
means are both saved. Oracle is NOT a deployment selector.

This PDBbind-only val does not establish multi-site HOLO4K assignment quality.
After reviewing all val configurations, a protocol may be fixed and evaluated
on held-out data once; selecting the best test configuration is prohibited.

## TRAIN negative audit

The old PDBbind-only 10.89% statistic is not reused. All 10317 current TRAIN rows
are in the denominator, including singleton groups, zero counts and QC failures.
Report PDBbind/BioLiP separately, group-balanced coverage, and the expected
anchor coverage under uniform batch sizes 8/16/32/64. Dataset-level coverage is
not the same as availability in a random training minibatch.

1. Exact residue key + residue name axis grouping, with strict length and
   duplicate-key checks. No arbitrary homolog residue-mask transfer.
2. Distinct candidate: Jaccard<=0.2, overlap count<3, and contact-mask centroids
   >=8 A apart when **both masks are mapped into each protein frame**. Never
   compare raw centroids across unrelated structure frames.
3. Geometry verification uses all corresponding CAs, independent of contact
   masks: proper-rotation Kabsch, maximum CA error<=0.05 A. Exact-coordinate
   matches bypass fitting. Compatible-but-not-identical structures with
   RMSD<=2 A and max error<=5 A are an exploratory count only, not approved
   direct-region negatives.
4. Known, distinct sanitized heavy-atom canonical isomeric SMILES required for
   conservative validity. Identical chemistry and unknown chemistry are separate
   categories, not negative examples. Local original SDFs supplied all 5160
   PDBbind TRAIN identities; the manifest includes SDF hashes and split hash.
   BioLiP identities are read from feature-recorded source SDFs on HPC.
5. Recheck after common training-visible residue truncation and ESM axis checks.
   Atom counts above 256 are flagged because full-molecule chemistry does not
   necessarily characterize the cropped atom view.

Remaining limitations: different SMILES does not prove nonbinding, tautomer/
protonation equivalence is not collapsed, partial/renumbered homologs are omitted,
and prepared label axes are not regenerated from original atom-level structures
in this audit. Consequently conservative coverage is a lower-bound diagnostic,
not a certified biological nonbinder dataset. No training negative artifact is
written automatically.

## Implementation and checks

- `scripts/diagnostics/export_strict_val_predictions.py`
- `scripts/diagnostics/strict_val_candidates.py`
- `scripts/diagnostics/audit_strict_train_negatives.py`
- `scripts/diagnostics/strict_assignment_common.py`
- `scripts/slurm/run_strict_val_candidates.slurm`: `i64m1tga800ue`, one GPU.
- `scripts/slurm/run_strict_train_negative_audit.slurm`: `emergency_cpu`, eight CPUs.

Local BIO environment: 33 related tests passed, including target-independent
proposals, geometric-frame invariance, reflection rejection, chemical identity
masking, unknown chemistry, high overlap, empty targets, train/val/test overlap,
random-batch coverage and axis-shift rejection. Slurm scripts pass `bash -n`.
Local optional PyG extension warnings do not affect these tests; GPU execution
is checked separately in the existing HPC environment.
