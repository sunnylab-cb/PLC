from scripts.diagnostics.audit_pdbbind_coverage import qualifying_hit, split_ids, all_ids


def test_asymmetric_coverage_is_current_policy():
    v = "q t 0 20 0.40 100 20 0.20 1.0 0 30".split()
    assert qualifying_hit(v)[-1]
    v[8] = "0.79"
    assert not qualifying_hit(v)[-1]


def test_identity_threshold_and_percent_format():
    v = "q t 0 100 0.299 100 100 1 1 0 30".split()
    assert not qualifying_hit(v)[-1]
    v[4] = "30.0"
    assert qualifying_hit(v)[-1]


def test_three_column_split_preserves_all_ids(tmp_path):
    p = tmp_path / "splits.txt"
    p.write_text("# example\ntrain\t1abc\ttrain\nval\t2ABC\thigh_sim\ntest\t3abc\ttest\n")
    assert all_ids(split_ids(p)) == {"1abc", "2abc", "3abc"}
