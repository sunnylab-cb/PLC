# Benchmark integrity repair and paired controls

Date: 2026-09-19. Scope: P0 of `holo4k_architecture_metric_audit_20260919.md`.
No training, checkpoint modification, external-set threshold selection, or source
feature-cache rewrite is performed.

## Implementation

- `scripts/infer/benchmark_integrity.py`: restricted tensor loading, residue-key
  validation, ESM alignment, finite checks, explicit SASA-only fallback, historical
  tag-axis reconstruction, and source-distance verification.
- `scripts/infer/infer_v11_benchmark.py`: CPU preflight, audited input mode,
  explicit inference precision, metric denominators, per-row QC ledger, and
  optional prediction exports.
- The V11 model inheritance path now accepts optional ESM2/ESM3 validity masks.
  Missing ESM2 rows have zero projected contribution; missing ESM3 rows do not
  enter global-prior pooling. No parameters are added. Unmasked calls retain
  their old behavior. All-valid masked predictions match unmasked predictions
  exactly in the small V11m13 forward regression test.

Raw ligand SASA/lipophilic-SASA columns 38/39 may be replaced by zero ONLY under
the explicit `--nonfinite-policy sasa_zero` control, before applying the unchanged
training normalization statistics. Every repair records coordinates in the
feature matrix and the source file hash. Other nonfinite features and valid-row
nonfinite logits are fatal. Original feature files remain unchanged.

## Important tag-axis finding

The historical benchmark tag generator on HPC is:

`/hpc2hdd/home/shuang886/jhspoolers/PLC/scripts/benchmark/tag/calculate_distances_dcenter.py`

It is NOT equivalent to the current protein feature parser or current
`preparation/tag/calculate_distances_dcenter.py`:

1. Bio.PDB model 0, non-HETATM residues with at least one heavy atom are included.
2. Residues are stably sorted by `(chain_id, residue_number)`. Insertion-code ties
   preserve source iteration order; insertion codes are retained in our keys.
3. A CA atom and canonical amino-acid identity are not required by that generator.
4. The minimum heavy-atom distance to the source ligand SDF is the label value.

The first attempted reconstruction from the protein feature parser failed on a
190-versus-191 residue case. The audit correctly blocked inference. The replacement
reconstructs the historical order and recomputes the ENTIRE raw distance vector
from the original atom+H PDB and ligand SDF. Maximum absolute error must be at
most `1e-4` Angstrom before mapping labels onto the protein feature residue axis.
This also detects equal-length ordering mistakes. Any extra tag-only residues
are counted explicitly; missing protein labels remain fatal.

The old source PDB/SDF coordinates are used for label verification and reported
diagnostics only, not added to the model as protein-ligand relative coordinates.

## ESM3 numbering defect and identity checks

The key-only audit exposed duplicate keys in normalized ESM3 caches. These are
not necessarily empty padding. For `1a4w`, source tokens 19 and 33 both claim
`L:11` but have sequence letters S and D, respectively, and both have atoms.
The cache's key-derived three-letter names report ASP for both. Token 19's CA
matches the canonical protein SER L11 exactly; token 33's nearest canonical CA
is 2.431 Angstrom away. Keeping the last dictionary entry is therefore unsafe.

For caches carrying atom37 coordinates/masks and sequence letters, the audited
runner now requires a unique same-chain CA match within 0.01 Angstrom and checks
the sequence identity. No rigid alignment, nearest-pocket selection, or ligand
coordinates are involved. Missing matches are masked, ambiguous matches or
contradictory sequence identities fail. Key reassignments, duplicate resolutions,
and three-letter-name conflicts are recorded. ESM2 caches without this metadata
continue to use exact residue keys with identity checks.

This fixes inference mapping, not the stored cache or any past training. Whether
the same extractor defects affected training inputs requires a separate audit;
it must not be presumed either absent or responsible for the performance gap.

## Paired evaluation protocol

Two unchanged strict-split seed44 checkpoints:

- Base: `pocket_gvp_fusion_seed44_chain_si30_biolipmerge_scratch_seed44_12222454_1_v11m13_20260826_055836`.
- ESM3: `pocket_gvp_fusion_seed44_chain_si30_biolipmerge_esm3gp_scratch_seed44_12620756_1_v11m13e3ft_20260903_011256`.

Both use archived `best_f1_c0_model.pt`, their saved validation contact gate,
training feature stats, strict per-ligand rows, `max_atoms=-1`, and protein cap
1800. No benchmark-driven tuning. One GPU runs all controls sequentially.

| Arm | Alignment | Residue coverage | Precision |
| --- | --- | --- | --- |
| `legacy_bf16` | Historical positional behavior | Historical minimum-length budget | BF16 |
| `keys_legacy_budget` | Verified labels + ESM key alignment + missing masks | Historical budget | BF16 |
| `keys_protein` | Same audited alignment | Protein axis up to 1800 | BF16 |

The last arm exports probabilities, labels, predictions, CA coordinates and
residue keys for later candidate diagnostics. These are not inference-time GT
inputs. Old FP16 benchmark CSVs remain untouched. Comparing them with
`legacy_bf16` bundles the precision change with explicit SASA fallback; it is NOT
an isolated precision ablation. Key alignment and missing-mask handling are
likewise evaluated as one integrity-repair intervention.

Expected old inclusion ledger: 494 COACH420 ligand rows and 7245 HOLO4K ligand
rows. These are evaluated subsets, not the nominal full benchmark populations.
Verify exact sample-key equality on completion, not just counts. Report sample
and protein-balanced metrics, all nonfinite/repair counts, BP/LAA denominators,
and affected-versus-unaffected row comparisons. Conditional BP is not a headline
unconditional pocket-detection metric.

## Deployment and verification

Dedicated remote root:
`/hpc2hdd/home/shuang886/PLC/experiments/benchmark_integrity_20260919`

Only source files were rsynced into its `code/` directory. The main HPC repository,
old outputs, weights and input caches are not overwritten. New result roots are
job-specific. GPU scheduling uses `i64m1tga800ue`; CPU uses `emergency_cpu`.

- Attempt `12792919`: failed at a historical parser sibling import; dependent
  GPU `12792924` was automatically cancelled without executing inference.
- Attempt `12792936`: failed the tag-axis length assertion; dependent GPU
  `12792940` was automatically cancelled without executing inference.
- Attempt `12792965`: COACH420 494/494 passed; stopped at HOLO4K ESM3 duplicate
  keys. No GPU evaluation submitted from this attempt.
- `12792988`: full key-only CPU audit, records every failure instead of stopping
  at the first. Still exits nonzero if any row fails. This is diagnostic evidence,
  not the dependency permitting GPU evaluation.
- `12793034`: enhanced identity-check audit using independent `code_ca/` snapshot.
- `12793043`: one-GPU `i64m1tga800ue` evaluation, dependency
  `afterok:12793034`, with automatic cancellation if its dependency fails.
  Both COMPLETED. Audit passed 494/494 COACH and 7245/7245 HOLO rows in 12m28s.
  All six frozen-checkpoint evaluation arms completed in 47m25s.
- `12792988` ultimately reported seven ESM3 duplicate-key failures in five HOLO
  proteins. The enhanced identity-check audit resolved all seven.
- P1 candidate diagnostic `12803928` COMPLETED in 39s on eight CPUs. It found
  essentially no whole-set headroom for ranking thresholded-mask components.
  See `benchmark_integrity_results_20260919.md` before starting P2 training.

Local validation: 27 tests passed across `test_benchmark_integrity.py`,
`test_delta_scale_gradient.py`, `test_internal_eval_compat.py`, and
`test_contact_probability_metrics.py`. This includes real PDB parsing, SDF
distance verification, insertion codes, missing middle residues, equal-length
reordering, restricted NumPy cache loading, SASA repair and mask propagation.
Additional tests cover complete failure accounting and coordinate/sequence-based
duplicate resolution without ambiguous or cross-chain matches.
These are not a substitute for the full remote cache audit.

Continuation added candidate diagnostics and an end-to-end prediction/SDF test;
the current combined suite has 32 passing tests. P0 numerical results and P1
interpretations are documented separately in the results report above.

## Next stage, after corrected results

P1 will compare prediction-derived candidate recall, oracle candidate F1 and
actual score-selected F1, using validation-fixed or explicitly prespecified
clustering rules. Oracle selection is diagnostic only. Report same-chemistry
copies separately without dropping them from strict metrics. Re-audit negative
coverage on the CURRENT strict merged training split before any site-assignment
training; the old PDBbind-only 10.89% figure must not be reused as current coverage.
