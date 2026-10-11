"""Task-specific quality scorers. Each returns a score in [0, 1], or None if unscorable.

Query field `scorer` selects one of: exact, choice, numeric, contains, unittest, judge.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Mapping

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks before scoring."""
    return _THINK_RE.sub("", text or "").strip()


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def score_exact(response: str, reference) -> float:
    refs = reference if isinstance(reference, list) else [reference]
    r = _norm_text(response)
    return 1.0 if any(_norm_text(str(x)) == r for x in refs) else 0.0


def score_choice(response: str, reference: str) -> float:
    """Multiple choice: last standalone letter A-J in the response must equal the reference."""
    letters = re.findall(r"\b([A-J])\b", response.upper())
    return 1.0 if letters and letters[-1] == str(reference).strip().upper() else 0.0


def _last_number(text: str) -> float | None:
    if "####" in text:
        text = text.split("####")[-1]
    nums = _NUM_RE.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def score_numeric(response: str, reference, tol: float = 1e-6) -> float:
    got, want = _last_number(response), _last_number(str(reference))
    if got is None or want is None:
        return 0.0
    return 1.0 if abs(got - want) <= tol * max(1.0, abs(want)) else 0.0


def score_contains(response: str, required: list[str]) -> float:
    """Fraction of required substrings/constraints present (case-insensitive)."""
    if not required:
        return 0.0
    low = response.lower()
    return sum(1 for s in required if s.lower() in low) / len(required)


def extract_code(response: str) -> str:
    m = _CODE_BLOCK_RE.findall(response)
    return m[-1] if m else response


def score_unittest(response: str, tests: str, timeout: float = 10.0) -> float:
    """Run model-generated code against tests in a subprocess.

    WARNING: this executes untrusted model output. Run collection inside a container/sandbox
    with no network and no credentials.
    """
    code = extract_code(response)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "candidate_test.py"
        p.write_text(code + "\n\n" + tests + "\n", encoding="utf-8")
        try:
            res = subprocess.run([sys.executable, "-I", str(p)], cwd=d, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0.0
    return 1.0 if res.returncode == 0 else 0.0


JUDGE_PROMPT = """You are grading a response to a user request.
Rubric: {rubric}

User request:
{prompt}

Response:
{response}

Grade from 1 (unusable) to 10 (excellent) using only the rubric. Reply with a line "Score: <n>"."""


def score_judge(prompt: str, response: str, rubric: str, judge: Mapping) -> float | None:
    """LLM-as-judge via an OpenAI-compatible endpoint. Use a model that is NOT in the routing pool,
    and validate its scores against a manually graded sample before using them as rewards."""
    body = {
        "model": judge["name"],
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(rubric=rubric, prompt=prompt, response=response)}],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    req = urllib.request.Request(judge["base_url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                 **({"Authorization": f"Bearer {judge['api_key']}"} if judge.get("api_key") else {})})
    with urllib.request.urlopen(req, timeout=judge.get("timeout", 60)) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"Score:\s*(\d+)", text)
    if not m:
        return None
    return (min(max(int(m.group(1)), 1), 10) - 1) / 9.0


def score(query: Mapping, response: str, judge: Mapping | None = None) -> float | None:
    resp = strip_thinking(response)
    kind = query.get("scorer", "exact")
    ref = query.get("reference")
    if kind == "exact":
        return score_exact(resp, ref)
    if kind == "choice":
        return score_choice(resp, ref)
    if kind == "numeric":
        return score_numeric(resp, ref)
    if kind == "contains":
        return score_contains(resp, query.get("required", []))
    if kind == "unittest":
        return score_unittest(resp, query["tests"])
    if kind == "judge":
        if judge is None:
            return None  # deferred: no judge configured
        return score_judge(query["prompt"], resp, query.get("rubric", "helpful, correct, complete"), judge)
    raise ValueError(f"unknown scorer {kind!r}")


SCORERS: dict[str, Callable] = {"exact": score_exact, "choice": score_choice, "numeric": score_numeric,
                                "contains": score_contains, "unittest": score_unittest}
