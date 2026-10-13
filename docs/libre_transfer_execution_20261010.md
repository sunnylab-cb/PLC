# LiBRe-inspired transfer: seed44 execution

Date: 2026-10-10. This is an implementation/run ledger, not an improvement claim.
User approved the four-arm plan. Do not compare a selected validation maximum to
the released LiBRe test score as evidence of an architectural advantage.

Latest check (2026-10-11): smoke12964495 and launcher12964516 completed
with exit0. All four tasks of training array12964718 completed35epochs normally.
No variant exceeds the unchanged baseline bestF1; CNC-off is the closest.
See `libre_transfer_results_20261011.md`. Submission-time observations below
are retained as history, not current queue status.

## Matrix

| Array index | Variant | Scientific change | Extra parameters |
|---|---|---|---:|
| 0 | plc_product_readout | LN-projected protein/readout product, 64 extra local-head input channels | 82,304 |
| 1 | plc_additive_readout_control | Same projections/head width, `(u+v)/sqrt(2)` instead of product | 82,304 |
| 2 | plc_cnc_off | Disable confident-negative clipping only | 0 |
| 3 | plc_residue_classmass | Training-only contact/far positive-class mass, weight .02, temperature .2 | 0 |

All arms use GFGF V11m13, local54/group-rescompat/co-attn/FiLM/virtual tokens,
seed44, scratch initialization, no frozen trunk, B16, BF16, LR1e-4,
120-epoch cosine horizon, and loss-based early stopping patience30. Best-F1
selection is retained, with the existing secondary AUPRC/AUROC checkpoints.
Caps remain 1800 residues / 256 atoms with alternating-end atom truncation.
No new foundation-model downloads, torch.compile, or shared P-L coordinates.

The center auxiliary head retains its original 1028D input. Product and additive
arms have identical initial parameter values, including widened-head weights;
new initialization does not consume the shared training RNG stream. Defaults
add no state-dict keys, preserving old checkpoint loading.

Classmass samples at most32 contact and32 far residues per input row (B16 gives
at most1024 total), excludes near/padding/self-pairs, skips anchors with no
same-class partner, and handles one-class/empty sets. Its FP32 objective is
`logsumexp(nonself) - logsumexp(same_class_nonself)`, not canonical SupCon.
Average classes within each input protein/pair row, then average eligible rows;
this does not deduplicate exact-protein groups. A private deterministic sampler
does not change model dropout RNG. Evaluation does not compute the auxiliary.

## Validation

Local BIO environment: 46 tests passed across transfer training (18), GDN,
transfer-log parsing, contact metrics and LiBRe ranking tests. Python syntax and
Slurm shell syntax checks passed. Local PyG optional scatter/sparse extension
warnings were emitted; full small-model CPU forward/backward still passed.

GPU smoke uses the real patched trainer, first16 TRAIN pairs, three optimizer
steps per arm, original focal/Tversky/CNC/distance losses plus the intended
auxiliary, gradient checks, FP16 validation (matching the existing evaluator),
strict state-dict reconstruction and parameter-budget assertions. This is a
connectivity/stability check, not a validation performance measurement or a
guarantee that every longer training batch fits memory.

New TB diagnostics: `Health/conditional_readout_abs_mean`,
`Gradients/conditional_readout_norm`, `Gradients/local_delta_head_norm`, and
`ResidueContrast/{raw_loss,weighted_loss,sampled_contact,sampled_far,eligible_anchors,eligible_rows}`.
Gradient norms are measured after AMP unscale and before clipping.

## Deployment

Remote experiment root:
`/hpc2hdd/home/shuang886/PLC/experiments/libre_transfer_20261010`

Base source: `protein_hybrid_gdn_20261006/code_v1`, copied without changing the
original. FLA dependencies are linked from the frozen base's `deps` directory.
Only the nine implementation/test/Slurm files were overlaid; remote dirty git
state was not pulled, cleaned or overwritten. All outputs and logs stay on HDD.

Frozen assets verified before preparing configs:

| Asset | SHA256 |
|---|---|
| Base output config | c58d7a8d4dd1a1840232e5cda9b648b62085a93da7dc14c00753e3ac5944fa69 |
| Strict split, 10317/647/647 | 170a4e6364c19662ad922ace7957e1733409dfe146ebdb74b15c95130bb622bc |
| TRAIN/VAL-aligned distance labels | c1b56f25dc15f80c81c3321ec48ad2c67fd87ec9aca02eca76d09f240ff980b6 |

`prepared/provenance.json` records source hashes, config hashes and the exact
scientific differences. Slurm verifies these before smoke/training; training
also requires a successful smoke matching that provenance digest.

- GPU smoke: `12964495`, `i64m1tga800ue`, Pending Priority at the last check.
- Submission-only continuation: `12964516`, `emergency_cpu`,
  `afterok:12964495`, one CPU / 2GB / 5min. No training/inference on this job.
- Training: not submitted yet. The continuation verifies successful smoke,
  all four arm names/parameter budgets and unchanged provenance, then submits
  the training array. A failed smoke cancels the dependent continuation; a
  stale/partial smoke fails closed. Duplicate invocation is file-lock guarded.
- Training IDs and exact sbatch command are written to `submission.json`.
- Resource policy: one GPU per task, training array `0-3%2`.
- Use `ssh hpc2` and absolute `/opt/slurm/bin/{squeue,sacct,sbatch}`.
- Login-node `nvidia-smi` availability is not an allocation/availability check.

Both UE and emergency_gpu expose fully allocated A800 nodes at this check;
Slurm reports no start-time estimate. The smoke was not duplicated across
partitions. Its training-source snapshot remains immutable. The two dispatch
files live in `launcher/` outside that snapshot, and the additional guard unit
test was run locally without replacing the already frozen test source.

After training, first compare strict VAL F1/AUPRC/fixed-budget ranking, precision,
recall, contact counts and matched epochs. A pilot +.003 F1 is a screening rule,
not significance. Only survivors receive seed confirmation and fixed-protocol
ligand-dependency / held-out benchmark evaluation. No TEST data were used here.
