import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import run_internal_pairwise_multinode as multi


def test_shards_cover_each_pair_once():
    for n in range(25):
        shards = [multi.shard_tasks(n, 7, rank, 4) for rank in range(4)]
        tasks = sorted(t for shard in shards for t in shard)
        assert tasks == multi.shard_tasks(n, 7, 0, 1)
        assert len({t[0] for t in tasks}) == len(tasks)


@pytest.mark.parametrize("labels", [None, ["train", "train", "val", "test"]])
def test_four_rank_result_and_resume_match_original(tmp_path, labels):
    core = multi.load_legacy(ROOT)
    seqs = ["AAAAAAAA", "RRRRRRRR", "WWWWWWWW", "DDDDDDDD"]
    base = dict(identity=.3, coverage=.8, edge_rule="global_only")
    reference, _ = core.run_stage(seqs, labels, base, tmp_path / "reference", 1, 1)
    out = tmp_path / "distributed"
    out.mkdir()
    contract = multi.stage_contract(base, labels)
    assert contract == json.loads((tmp_path / "reference/contract.json").read_text())
    core.save_json(out / "contract.json", contract)
    request = dict(legacy_code=str(ROOT), legacy_sha256=core.digest(Path(core.__file__)),
                   driver_sha256=core.digest(Path(multi.__file__)), sequences=seqs, labels=labels,
                   out=str(out), contract=contract, chunk_size=1, world=4,
                   workers_per_node=1, invocation="test")
    path = tmp_path / "request.json"
    core.save_json(path, request)
    for rank in range(4):
        multi.worker(path, rank, 4, 1)
    before = {p.name: (core.digest(p), p.stat().st_mtime_ns) for p in out.glob("chunk_*")}
    for rank in range(4):
        multi.worker(path, rank, 4, 1)
        progress = json.loads((out / f"rank_{rank:02d}_progress.json").read_text())
        assert progress["done"] and progress["fresh_evaluated"] == 0
    result, _ = core.run_stage(seqs, labels, base, out, 1, 1)
    assert result == reference
    assert before == {p.name: (core.digest(p), p.stat().st_mtime_ns) for p in out.glob("chunk_*")}
    with pytest.raises(ValueError, match="allocation"):
        multi.worker(path, 0, 2, 1)
