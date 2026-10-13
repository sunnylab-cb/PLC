# Reviewed result snapshot, 2026-10-13

Small machine-readable exports from already collected evidence. This export
does not run inference, contact the HPC, admit new data or change a model.

| File | Scope |
| --- | --- |
| `pdbbind_coverage.json` | Original/legacy/strict PDBbind coverage and external homology audit |
| `biolip_tensor_qc_summary.json` | Actual tensor and known-chain checks of expansion candidates |
| `known_ligand_summary.json` | 1,390 extracted ligand tensors; first-stage QC only |
| `known_ligand_provenance.json` | Extraction inputs/configuration recorded by the isolated job |
| `context_noise_summary.json` | Six noise pilots plus reference, clean strict VAL seed44 |
| `self_distillation_summary.json` | Two KD pilots plus reference, strict VAL seed44 |
| `libre_transfer_summary.json` | Product/additive/CNC/classmass pilots plus reference, strict VAL seed44 |
| `provenance.json` | Original local paths, source SHA256 and exact jq export filters |

The full matched external evaluation is separately stored in
[`../libre_benchmark_12960610/final_report.json`](../libre_benchmark_12960610/final_report.json).
The [current project status](../../docs/PROJECT_STATUS.md) links the detailed
metric definitions, protocols, uncertainties and decisions.

Best-F1 checkpoint metrics, independently selected best-AP metrics and
VAL-fitted threshold metrics are intentionally separate. Tail-epoch means
are not seed statistics. VAL-only pilots do not imply TEST or external gains.
New feature extraction is not admission into TRAIN or evidence of model gain.

Raw TensorBoard series, predictions, input tensors and checkpoint files remain
outside Git. Original paths under `tmp/` identify workstation evidence;
the source hashes and export filters permit checking these exports when that
evidence is available. Remote provenance/locations remain in the dated docs.
