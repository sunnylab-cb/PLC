# PLC: Protein-Ligand Contact Prediction

Ligand-conditioned, residue-level contact prediction from protein sequence
and structure features plus ligand chemistry and graph features.

## Repository

- Current repository: https://github.com/sunnylab-cb/PLC
- Active and default branch: `master`.
- The former `hsm` branch's 235 commits were preserved without squashing or
  rewriting, through `c5a30e17d814f5b3b4f62d609e972712f400680f`.
- On the migration workstation, `plc` names this repository and `origin`
  still names the previous repository. Local `master` tracks `plc/master`.
  A fresh clone of this repository will normally call it `origin`.
- This is a Git code/history migration, not a migration of GitHub issues,
  pull requests, release assets or Actions records.

## Start Here

- [Latest result and data status, 2026-10-13](docs/PROJECT_STATUS.md).
- [Experiment planning and execution index](refine-logs/MANIFEST.md).
- [PDBbind coverage and BioLiP expansion audit](docs/pdbbind_coverage_and_biolip_expansion_20261013.md).
- [Matched LiBRe / PLC benchmark](docs/libre_benchmark_20261010.md).
- [Historical architecture and feature report](report/report.tex), with
  [PDF](report/report.pdf). This July report predates the strict-split and
  October experiments; use the dated result documents for current metrics.

## Code Layout

| Directory | Purpose |
| --- | --- |
| `src/models/`, `src/losses/` | Model families, optional probes and objectives |
| `preparation/` | Protein/ligand features, normalization and dataset preparation |
| `configs/` | Historical configurations and split manifests |
| `scripts/train/`, `scripts/infer/` | Training and evaluation entry points |
| `scripts/diagnostics/`, `scripts/benchmark/` | QC, coverage and benchmark analysis |
| `scripts/slurm/` | Experiment-specific HPC launchers |
| `tests/` | Unit and integration tests; some need the scientific Python stack |
| `docs/`, `refine-logs/`, `results/` | Dated conclusions, protocols and small summaries |

Historical configuration and launcher files are not interchangeable. Consult
the relevant experiment ledger before running a script; do not infer the
current frozen split from an old file named `splits.txt`.

## Artifact Policy

Commit source, configuration, tests and reviewed result summaries together.
Keep dataset tensors, pretrained weights, checkpoints, predictions, raw TB
and run logs on the data servers. `.gitignore` excludes these and local
credentials/settings; it does not erase already tracked historical files.

References to `tmp/` in older documents describe local evidence snapshots,
not files guaranteed to be in Git. Small October evidence exports are indexed
in [the current result snapshot](results/project_snapshot_20261013/README.md).
No checkpoint or dataset is included with this update.

On the migration workstation, publish reviewed updates with
`git push plc master`. New clones can use `git push origin master`.
