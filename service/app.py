"""Step 7: live router service.

  POST /route     {"prompt": "...", "task_category": "coding"}  -> picks a model, calls it, returns the answer
  POST /feedback  {"decision_id": "...", "quality": 0.8}         -> reward + LinUCB update (delayed feedback)
  GET  /stats

run:  ROUTER_MODELS=serving/models.json uvicorn service.app:app --port 9000
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
from pydantic import BaseModel, Field

from evaluation.collect import ServerDown, call_model, compute_cost
from evaluation.io import append_jsonl
from evaluation.scoring import strip_thinking
from router.features import FeatureExtractor
from router.linucb import LinUCB
from router.reward import RewardConfig, compute_reward


class RouteRequest(BaseModel):
    prompt: str
    task_category: str | None = None


class FeedbackRequest(BaseModel):
    decision_id: str
    quality: float = Field(ge=0.0, le=1.0)


def create_app(models_cfg: Mapping, reward_cfg: RewardConfig, state_path: str | None = None,
               log_path: str | None = None, alpha: float = 1.0, lam: float = 1.0,
               caller: Callable = call_model, fx: FeatureExtractor | None = None) -> FastAPI:
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
        return {"alpha": policy.alpha, "updates": policy.counts, "pending_feedback": len(pending),
                "features": fx.names(), "reward_config": reward_cfg.to_dict()}

    return app


def _from_env() -> FastAPI:
    cfg = json.load(open(os.environ.get("ROUTER_MODELS", "serving/models.json")))
    rpath = os.environ.get("ROUTER_REWARD")
    rcfg = RewardConfig()
    if rpath:
        rc = json.load(open(rpath))
        rcfg = RewardConfig(**{k: tuple(v) if k.endswith("_range") else v for k, v in rc.items()})
    return create_app(cfg, rcfg, state_path=os.environ.get("ROUTER_STATE", "router_state.json"),
                      log_path=os.environ.get("ROUTER_LOG", "logs/audit.jsonl"),
                      alpha=float(os.environ.get("ROUTER_ALPHA", "1.0")))


app = _from_env() if os.environ.get("ROUTER_MODELS") else None
