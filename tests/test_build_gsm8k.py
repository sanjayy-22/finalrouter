import json

from data.build_gsm8k import build
from evaluation.scoring import score


def test_build_split_and_reference():
    rows = [{"question": f"q{i}", "answer": f"work...\n#### {i},000"} for i in range(20)]
    qs = build(rows, n=10, dev_frac=0.3, seed=0)
    assert len(qs) == 10 and sum(q["split"] == "dev" for q in qs) == 3
    assert len({q["query_id"] for q in qs}) == 10
    for q in qs:
        i = int(q["query_id"].split("-")[-1])
        assert q["reference"] == f"{i}000"
        assert score(q, f"#### {i}000") == 1.0
    assert build(rows, 10, 0.3, 0) == qs          # deterministic
