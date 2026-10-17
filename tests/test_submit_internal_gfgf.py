import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/diagnostics"))
import submit_internal_gfgf as submit


@pytest.mark.parametrize("cohort,stages", [
    ("pdbbind_only", ["prepare", "smoke", "train"]),
    ("pdbbind_biolip_joint", ["si", "prepare", "smoke", "train"]),
])
def test_submission_is_idempotent_and_fail_closed(tmp_path, monkeypatch, cohort, stages):
    root, source = tmp_path/"experiment", tmp_path/"source"
    root.mkdir(); source.mkdir()
    (root/"deployment.json").write_text(json.dumps({"source_sha256": {}}))
    base = tmp_path/"base.yaml"
    base.write_text("test fixture\n")
    if cohort == "pdbbind_biolip_joint":
        pool = source/"qc_pool/pool_summary.json"
        pool.parent.mkdir()
        pool.write_text(json.dumps(dict(cohort=cohort, external_benchmarks_used=False,
                                       accepted_by_source={"pdbbind": 3, "biolip": 3})))
    else:
        for rel in ("pdbbind_only/pool/pool_summary.json", "pdbbind_only/split/all_pairs/summary.json"):
            path = source/rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
    commands = []

    def sbatch(command, **kwargs):
        commands.append(command)
        return str(100+len(commands))+"\n"

    monkeypatch.setattr(submit.subprocess, "check_output", sbatch)
    monkeypatch.setattr(sys, "argv", ["submit", "--experiment", str(root), "--source", str(source),
                                      "--base", str(base), "--cohort", cohort])
    submit.main()
    receipt = json.loads((root/"submission.json").read_text())
    assert set(receipt["jobs"]) == set(stages)
    assert len(commands) == len(stages)
    for index, command in enumerate(commands):
        if index:
            assert f"--dependency=afterok:{100+index}" in command
            assert "--kill-on-invalid-dep=yes" in command
        assert "scancel" not in " ".join(command)
    if cohort == "pdbbind_biolip_joint":
        assert commands[0][-1].endswith("run_joint_global_si30_prepare.slurm")
        assert any("STAGE=si" in arg for arg in commands[0])
        assert "--cpus-per-task=8" in commands[1]
        assert "--time=7-00:00:00" in commands[-1]
    submit.main()
    assert len(commands) == len(stages)
