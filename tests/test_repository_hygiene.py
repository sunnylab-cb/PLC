"""Protect source/result visibility while excluding local experiment payloads."""

import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", [
    "data/protein_features/example.pt",
    "benchmark/p2rank-datasets-fixed_hres/holo4k/example.pdb",
    "outputs/run/checkpoint_epoch_100.pt",
    "outputs/run/config.yaml",
    "tmp/arbitrary_run/summary.json",
    "tmp/arbitrary_helper.py",
    "report/.mplconfig/fontlist.json",
    "report/report.aux",
    "report/report.log",
    "results/run/predictions/example.npz",
    "results/run/sample_all_results.csv",
    "src/models/__pycache__/model.cpython-312.pyc",
    ".env",
    ".claude/settings.local.json",
    "weights/model.safetensors",
])
def test_experiment_payloads_are_ignored(path):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", path], cwd=ROOT,
    )
    assert result.returncode == 0


@pytest.mark.parametrize("path", [
    "README.md",
    "docs/PROJECT_STATUS.md",
    "src/models/protein_context_noise.py",
    "scripts/benchmark/run_libre_benchmark.py",
    "scripts/slurm/run_biolip_known_chain_expansion.slurm",
    "tests/test_repository_hygiene.py",
    "configs/splits/splits.txt",
    "refine-logs/biolip_expansion_20261013/EXPERIMENT_TRACKER.md",
    "report/report.tex",
    "report/report.pdf",
    "report/figures/fig_model_architecture.png",
    "results/project_snapshot_20261013/known_ligand_summary.json",
    "results/project_snapshot_20261013/summary_metrics.csv",
    "tmp/collect_training_coverage_audit.py",
    ".env.example",
])
def test_source_and_curated_results_are_not_ignored(path):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", path], cwd=ROOT,
    )
    assert result.returncode == 1


def test_result_snapshot_has_small_provenance_linked_exports():
    snapshot = ROOT / "results/project_snapshot_20261013"
    provenance = json.loads((snapshot / "provenance.json").read_text())
    assert len(provenance["files"]) == 7
    for entry in provenance["files"]:
        assert Path(entry["export"]).name == entry["export"]
        assert len(entry["source_sha256"]) == 64
        int(entry["source_sha256"], 16)
        path = snapshot / entry["export"]
        assert path.stat().st_size < 128 * 1024
        assert isinstance(json.loads(path.read_text()), dict)
