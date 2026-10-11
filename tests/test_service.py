from fastapi.testclient import TestClient

from router.reward import RewardConfig
from service.app import create_app

CFG = {"models": [{"model_id": "s", "name": "S", "base_url": "http://x"},
                  {"model_id": "l", "name": "L", "base_url": "http://y"}]}


def fake_caller(model, prompt, decoding):
    if model["model_id"] == "l":
        return {"response": "", "latency_ms": 5.0, "input_tokens": None, "output_tokens": None,
                "failed": True, "error": "Timeout"}
    return {"response": "<think>t</think>hello", "latency_ms": 10.0, "input_tokens": 3, "output_tokens": 2,
            "failed": False, "error": None}


def test_route_feedback_cycle(tmp_path):
    app = create_app(CFG, RewardConfig(cost_range=(0, 1), latency_range=(0, 100)),
                     state_path=str(tmp_path / "state.json"), log_path=str(tmp_path / "audit.jsonl"),
                     caller=fake_caller)
    c = TestClient(app)
    r = c.post("/route", json={"prompt": "hi", "task_category": "qa"}).json()
    assert r["model"] == "s" and r["response"] == "hello"            # tie -> first model
    fb = c.post("/feedback", json={"decision_id": r["decision_id"], "quality": 1.0}).json()
    assert fb["reward"] is not None
    assert c.post("/feedback", json={"decision_id": r["decision_id"], "quality": 1.0}).status_code == 404
    assert c.get("/stats").json()["updates"]["s"] == 1
    assert (tmp_path / "state.json").exists() and (tmp_path / "audit.jsonl").read_text().count("\n") == 1


def test_failure_updates_immediately(tmp_path):
    cfg = {"models": list(reversed(CFG["models"]))}                  # make "l" the tie-break winner
    app = create_app(cfg, RewardConfig(), caller=fake_caller)
    c = TestClient(app)
    r = c.post("/route", json={"prompt": "hi"}).json()
    assert r["model"] == "l" and r["failed"]
    assert c.get("/stats").json()["updates"]["l"] == 1


def test_ui_and_demo_mode():
    from service.app import demo_caller
    app = create_app(CFG, RewardConfig(), caller=demo_caller, demo=True)
    c = TestClient(app)
    page = c.get("/")
    assert page.status_code == 200 and "Adaptive LLM Router" in page.text
    s = c.get("/stats").json()
    assert s["demo"] is True and [m["model_id"] for m in s["models"]] == ["s", "l"]
    r = c.post("/route", json={"prompt": "2+2?", "task_category": "reasoning"}).json()
    assert "[demo mode]" in r["response"] and set(r["scores"]) == {"s", "l"}
