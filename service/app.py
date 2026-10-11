"""Step 7: live router service.

  POST /route     {"prompt": "...", "task_category": "coding"}  -> picks a model, calls it, returns the answer
  POST /feedback  {"decision_id": "...", "quality": 0.8}         -> reward + LinUCB update (delayed feedback)
  GET  /stats
  GET  /                                                         -> simple web UI (service/static/index.html)

run:   ROUTER_MODELS=serving/models.json uvicorn service.app:app --port 9000
demo:  ROUTER_DEMO=1 uvicorn service.app:app --port 9000     # no GPU: placeholder answers, real routing/learning
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Mapping

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from evaluation.collect import ServerDown, call_model, compute_cost
from evaluation.io import append_jsonl
from evaluation.scoring import strip_thinking
from router.features import FeatureExtractor
from router.linucb import LinUCB
from router.reward import RewardConfig, compute_reward


STATIC_DIR = Path(__file__).parent / "static"


def demo_caller(model: Mapping, prompt: str, decoding: Mapping) -> dict:
    """Stand-in for a model server so the UI can be tried without a GPU. Answers are placeholders;
    latency/tokens are synthetic and scale with the model's cost factor."""
    size = float(model.get("cost_per_1k_tokens", 1.0))
    out_tokens = 120 + 4 * len(prompt.split())
    return {"response": f"[demo mode] {model['name']} would answer here. No model was called.",
            "latency_ms": 200.0 + 60.0 * size + 2.0 * out_tokens, "input_tokens": len(prompt.split()),
            "output_tokens": out_tokens, "failed": False, "error": None}


class RouteRequest(BaseModel):
    prompt: str
    task_category: str | None = None


class FeedbackRequest(BaseModel):
    decision_id: str
    quality: float = Field(ge=0.0, le=1.0)


def create_app(models_cfg: Mapping, reward_cfg: RewardConfig, state_path: str | None = None,
               log_path: str | None = None, alpha: float = 1.0, lam: float = 1.0,
               caller: Callable = call_model, fx: FeatureExtractor | None = None,
               demo: bool = False) -> FastAPI:
    models = {m["model_id"]: m for m in models_cfg["models"]}
    order = [m["model_id"] for m in models_cfg["models"]]
    decoding = models_cfg.get("decoding", {})
    fx = fx or FeatureExtractor()
    if state_path and Path(state_path).exists():
        policy = LinUCB.load(state_path)
    else:
        policy = LinUCB(order, fx.dim, alpha=alpha, lam=lam)
    pending: dict[str, dict] = {}
    lock = threading.Lock()
    app = FastAPI(title="Adaptive LLM Router")

    def _apply(decision: dict, quality: float | None) -> float | None:
        r = compute_reward(quality, decision["cost"], decision["latency_ms"], decision["failed"], reward_cfg)
        if r is not None:
            policy.update(decision["model"], decision["x"], r)
            if state_path:
                policy.save(state_path)
        if log_path:
            append_jsonl(log_path, {"ts": time.time(), "decision_id": decision["id"], "model": decision["model"],
                                    "context": decision["x"].tolist(), "latency_ms": decision["latency_ms"],
                                    "cost": decision["cost"], "failed": decision["failed"],
                                    "quality": quality, "reward": r})
        return r

    @app.post("/route")
    def route(req: RouteRequest):
        query = {"prompt": req.prompt, "task_category": req.task_category}
        x = fx(query)
        with lock:
            scores = policy.scores(x)
            model_id = policy.select(x, query)
        try:
            out = caller(models[model_id], req.prompt, decoding)
        except ServerDown:
            raise HTTPException(503, f"model server for {model_id} is unreachable")
        decision = {"id": uuid.uuid4().hex, "model": model_id, "x": x, "latency_ms": out["latency_ms"],
                    "cost": compute_cost(models[model_id], out), "failed": out["failed"]}
        with lock:
            if out["failed"]:
                _apply(decision, None)          # failure -> Q = 0, update immediately
            else:
                pending[decision["id"]] = decision
        return {"decision_id": decision["id"], "model": model_id, "scores": scores,
                "response": strip_thinking(out["response"]), "latency_ms": out["latency_ms"],
                "failed": out["failed"], "error": out["error"]}

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        with lock:
            decision = pending.pop(req.decision_id, None)
            if decision is None:
                raise HTTPException(404, "unknown or already-rated decision_id")
            r = _apply(decision, req.quality)
        return {"decision_id": req.decision_id, "reward": r}

    @app.get("/stats")
    def stats():
        return {"models": [{"model_id": m, "name": models[m]["name"]} for m in order], "demo": demo,
                "alpha": policy.alpha, "updates": policy.counts, "pending_feedback": len(pending),
                "features": fx.names(), "categories": list(fx.categories), "reward_config": reward_cfg.to_dict()}

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _from_env() -> FastAPI:
    demo = os.environ.get("ROUTER_DEMO", "") not in ("", "0")
    cfg = json.load(open(os.environ.get("ROUTER_MODELS", "serving/models.json")))
    rpath = os.environ.get("ROUTER_REWARD")
    # fallback ranges: cost in the models.json proxy units, latency in ms; fit real ranges with
    # router.reward.fit_ranges on dev outcomes and pass them via ROUTER_REWARD
    rcfg = RewardConfig(cost_range=(0.0, 20.0), latency_range=(0.0, 10000.0))
    if rpath:
        rc = json.load(open(rpath))
        rcfg = RewardConfig(**{k: tuple(v) if k.endswith("_range") else v for k, v in rc.items()})
    default_state = "router_state_demo.json" if demo else "router_state.json"
    return create_app(cfg, rcfg, state_path=os.environ.get("ROUTER_STATE", default_state),
                      log_path=os.environ.get("ROUTER_LOG", "logs/audit_demo.jsonl" if demo else "logs/audit.jsonl"),
                      alpha=float(os.environ.get("ROUTER_ALPHA", "1.0")),
                      caller=demo_caller if demo else call_model, demo=demo)


app = _from_env() if (os.environ.get("ROUTER_MODELS") or os.environ.get("ROUTER_DEMO")) else None
