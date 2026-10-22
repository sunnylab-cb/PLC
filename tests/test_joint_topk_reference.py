import ast
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

from scripts.diagnostics.recompute_joint_topk_reference import (
    Aggregate, POLICY, budgets, metrics, score_row, validate_rows,
)


def test_budgets_clip_and_round_without_labels():
    assert budgets(3)["top80"] == 3
    assert budgets(21)["top_l20"] == 2
    assert budgets(21)["top_l10"] == 3
    assert budgets(21)["top_l5"] == 5
    with pytest.raises(ValueError):
        budgets(0)


def test_top10_is_exact_count_and_stable_at_ties():
    p = np.array([.9] * 12 + [.1] * 8)
    y = np.array([True] * 6 + [False] * 14)
    r = score_row(p, y, .553555)["top10"]
    assert r["predicted"] == 10 and r["tp"] == 6
    assert r["cutoff_tie"] == 1
    assert r["expected_tp"] == pytest.approx(5)
    assert metrics(20, 6, 10, r["expected_tp"])["f1"] == pytest.approx(.625)
    swapped = score_row(p, ~y, .553555)["top10"]
    assert swapped["predicted"] == 10 and swapped["tp"] == 4


def test_uniform_tie_expectation_matches_all_selections():
    p = np.ones(12)
    y = np.arange(12) < 4
    r = score_row(p, y, .5)["top10"]
    expected = metrics(12, 4, 10, r["expected_tp"])
    exhaustive = [metrics(12, 4, 10, sum(y[list(indices)]))
                  for indices in itertools.combinations(range(12), 10)]
    for key in expected:
        assert expected[key] == pytest.approx(np.mean([d[key] for d in exhaustive]))


@pytest.mark.parametrize("positive,predicted,tp", [(0, 0, 0), (20, 20, 20), (7, 10, 5), (7, 0, 0)])
def test_metrics_match_sklearn(positive, predicted, tp):
    y = np.array([1] * positive + [0] * (20 - positive))
    pred = np.zeros(20, dtype=bool)
    pred[:tp] = True
    pred[positive:positive + predicted - tp] = True
    m = metrics(20, positive, predicted, tp)
    assert m["f1"] == pytest.approx(f1_score(y, pred, zero_division=0))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred, zero_division=0))
    assert m["recall"] == pytest.approx(recall_score(y, pred, zero_division=0))


def test_pooling_is_not_macro_and_preserves_total_counts():
    a = Aggregate()
    for n, positive, predicted, tp in [(20, 5, 10, 5), (100, 30, 10, 1)]:
        a.add(dict(n=n, positive=positive, predicted=predicted, tp=tp, expected_tp=float(tp), cutoff_tie=0))
    d = a.finish()
    assert d["n_residues"] == 120 and d["n_predicted"] == 20
    assert d["pooled"]["f1"] == pytest.approx(12/55)
    assert d["sample_macro"]["f1"] == pytest.approx((10/15 + 2/40)/2)


@pytest.mark.parametrize("p", [[float("nan")], [float("inf")], [-.1], [1.1]])
def test_nonfinite_and_out_of_range_rejected(p):
    with pytest.raises(ValueError):
        score_row(p, [True], .5)


def test_population_and_donor_mismatch_rejected():
    spec = dict(sample_id="a", source="pdbbind", donor_id="b")
    row = dict(**spec, scores=torch.tensor([.9]), labels=torch.tensor([0], dtype=torch.uint8))
    assert len(validate_rows([row], {"samples": [spec]}, "shuffle")) == 64
    with pytest.raises(ValueError, match="population"):
        validate_rows([], {"samples": [spec]}, "real")
    with pytest.raises(ValueError, match="donor"):
        validate_rows([dict(row, donor_id="a")], {"samples": [spec]}, "shuffle")


def test_no_inference_training_or_best_budget_selection():
    source = Path("scripts/diagnostics/recompute_joint_topk_reference.py").read_text()
    names = {getattr(n.func, "id", getattr(n.func, "attr", ""))
             for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)}
    assert not names & {"find_best_f1_threshold", "backward", "AdamW", "predict_v11_logits", "create_v11_model"}
    assert POLICY["select_best_k"] is False
    assert POLICY["test_driven_iteration_allowed"] is False
    assert POLICY["primary"] == "top20"


def test_end_to_end_preserves_reference_and_refuses_overwrite(tmp_path, monkeypatch):
    from scripts.diagnostics import recompute_joint_topk_reference as runner
    from scripts.diagnostics.eval_joint_reference_test import summarize

    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    specs = [dict(sample_id="a", source="pdbbind", donor_id="b"),
             dict(sample_id="b", source="biolip", donor_id="a")]
    protocol = dict(samples=specs, val_gate=.553555)
    runner.write_json(source / "protocol.json", protocol)
    digest = runner.sha(source / "protocol.json")
    monkeypatch.setattr(runner, "REFERENCE_SHA", digest)
    reference = dict(complete=True, reference_only=True, test_driven_iteration_allowed=False,
                     protocol_sha256=digest, epoch=25, val_gate=protocol["val_gate"], summaries={})
    for mode in runner.MODES:
        rows = [dict(**spec, scores=torch.linspace(.1, .9, 21),
                     labels=torch.tensor([0, 1, 2] * 7, dtype=torch.uint8),
                     argmax=torch.tensor([2, 0, 2] * 7, dtype=torch.uint8)) for spec in specs]
        torch.save(rows, source / f"predictions_{mode}.pt")
        summary = {"all": summarize(rows, protocol["val_gate"])}
        for group in ("pdbbind", "biolip"):
            summary[group] = summarize([r for r in rows if r["source"] == group], protocol["val_gate"])
        summary["runtime"] = dict(residue_axis_sha256=validate_rows(rows, protocol, mode))
        reference["summaries"][mode] = summary
    runner.write_json(source / "reference_test_summary.json", reference)
    before = {p.name: runner.sha(p) for p in source.iterdir()}
    runner.evaluate(source, output)
    result = json.loads((output / "topk_reference_summary.json").read_text())
    assert result["complete"]
    assert result["summaries"]["real"]["all"]["top20"]["n_predicted"] == 40
    assert result["gaps"]["all"]["top20"]["real_minus_shuffle"]["pooled"]["f1"] == 0
    assert before == {p.name: runner.sha(p) for p in source.iterdir()}
    with pytest.raises(FileExistsError):
        runner.evaluate(source, output)
