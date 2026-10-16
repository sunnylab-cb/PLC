import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/diagnostics/biolip_ligand_takeover.py"
SPEC = importlib.util.spec_from_file_location("takeover", SOURCE)
takeover = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(takeover)


def test_job_fields_and_identity():
    info = takeover.fields("JobId=10 JobName=bio_expand_ligand_remaining "
                           "UserId=owner(123) WorkDir=/exp/code "
                           f"Command=/exp/code/{takeover.RUNNER}")
    takeover.validate_target(info, "10", Path("/exp"), "bio_expand_ligand_remaining", 123)
    for key in ("JobId", "JobName", "UserId", "WorkDir", "Command"):
        with pytest.raises(RuntimeError):
            takeover.validate_target(dict(info, **{key: "wrong"}), "10", Path("/exp"),
                                     "bio_expand_ligand_remaining", 123)


@pytest.mark.parametrize("dependency", ["afterok:20", "afterok:20(unfulfilled)"])
def test_qc_safe_dependency(dependency):
    takeover.validate_qc({"JobState": "PENDING", "Dependency": dependency}, "20")


@pytest.mark.parametrize("dependency", ["afterany:10", "afterany:20", "afterok:21",
                                       "afterok:20?afterany:10", "(null)"])
def test_qc_rejects_early_or_wrong_dependency(dependency):
    with pytest.raises(RuntimeError):
        takeover.validate_qc({"JobState": "PENDING", "Dependency": dependency}, "20")


def test_qc_must_not_be_running():
    with pytest.raises(RuntimeError):
        takeover.validate_qc({"JobState": "RUNNING", "Dependency": "afterok:20"}, "20")


def test_source_integrity(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code/source.py").write_bytes(b"original")
    deployment = {"files": {"source.py": hashlib.sha256(b"original").hexdigest()}}
    takeover.verify_source(tmp_path, deployment)
    (tmp_path / "code/source.py").write_bytes(b"changed")
    with pytest.raises(RuntimeError):
        takeover.verify_source(tmp_path, deployment)


def test_wait_requires_both_job_exit_and_free_lock(tmp_path):
    lock = tmp_path / "stage.lock"
    with patch.object(takeover, "command", side_effect=["10 COMPLETING", "20 RUNNING"]), \
            patch.object(takeover.time, "sleep") as sleep:
        takeover.wait_for_exit("10", lock)
    sleep.assert_called_once_with(5)
    assert lock.exists()


def test_wait_refuses_persistent_job(tmp_path):
    with patch.object(takeover, "command", return_value="10 RUNNING"), \
            patch.object(takeover.time, "monotonic", side_effect=[0, 1, 1000]), \
            patch.object(takeover.time, "sleep"):
        with pytest.raises(RuntimeError, match="refusing concurrent"):
            takeover.wait_for_exit("10", tmp_path / "stage.lock", timeout=10)


def test_wait_refuses_live_stage_lock_after_job_exit(tmp_path):
    lock_path = tmp_path / "stage.lock"
    with lock_path.open("a+") as handle:
        takeover.fcntl.flock(handle, takeover.fcntl.LOCK_EX | takeover.fcntl.LOCK_NB)
        with patch.object(takeover, "command", return_value=""), \
                patch.object(takeover.time, "monotonic", side_effect=[0, 1, 1000]), \
                patch.object(takeover.time, "sleep"):
            with pytest.raises(RuntimeError, match="refusing concurrent"):
                takeover.wait_for_exit("10", lock_path, timeout=10)
