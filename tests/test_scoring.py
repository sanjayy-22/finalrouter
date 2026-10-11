from evaluation.scoring import (score, score_choice, score_contains, score_exact, score_numeric,
                                score_unittest, strip_thinking)


def test_strip_thinking():
    assert strip_thinking("<think>hmm\nlong</think>\nAnswer: 4") == "Answer: 4"


def test_numeric_gsm8k_style():
    assert score_numeric("so the total is 1,234 apples.", "#### 1234") == 1.0
    assert score_numeric("I think 12", "13") == 0.0
    assert score_numeric("no number", "13") == 0.0


def test_choice_and_exact_and_contains():
    assert score_choice("The answer is (B). Final: C", "C") == 1.0
    assert score_exact("  Paris. ", ["paris", "Paris, France"]) == 1.0
    assert score_contains("apple, banana, cherry", ["apple", "cherry", "kiwi"]) == 2 / 3


def test_unittest_pass_and_fail():
    tests = "assert add(2, 3) == 5"
    assert score_unittest("```python\ndef add(a, b):\n    return a + b\n```", tests) == 1.0
    assert score_unittest("```python\ndef add(a, b):\n    return a - b\n```", tests) == 0.0


def test_dispatch_and_deferred_judge():
    assert score({"scorer": "numeric", "reference": "7"}, "<think>x</think>7") == 1.0
    assert score({"scorer": "judge", "prompt": "p"}, "r", judge=None) is None
