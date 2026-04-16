"""
distributed/server.py — Parameter server + PPO learner (FastAPI).

Responsibilities
----------------
1.  Serve current model weights to workers (GET /weights).
2.  Accept rollout batches from workers (POST /experience).
3.  Buffer incoming experience; run PPO updates in a background thread.
4.  Checkpoint the model periodically.
5.  Expose a status endpoint for monitoring (GET /status).

The training thread wakes up whenever the buffer holds at least
config.min_train_batch new experiences.
"""
from __future__ import annotations

import base64
import collections
import logging
import os
import threading
import time
from typing import List

import numpy as np
import torch
import torch.optim as optim
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from .config import TrainConfig
from .model  import MarketPolicy

log = logging.getLogger("dist.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# =============================================================================
# Pydantic schemas
# =============================================================================

class RolloutPayload(BaseModel):
    worker_id:      str
    ctrl_id:        int
    role:           str
    policy_version: int
    # All arrays sent as base64-encoded float32 / int32 bytes
    obs_b64:        str          # (T, STATE_DIM) float32
    actions_b64:    str          # (T, 3)         int32
    rewards_b64:    str          # (T,)            float32
    dones_b64:      str          # (T,)            float32  (0/1)
    log_probs_b64:  str          # (T,)            float32
    values_b64:     str          # (T,)            float32
    last_obs_b64:   str          # (STATE_DIM,)    float32  (bootstrap)
    steps:          int


class StatusResponse(BaseModel):
    policy_version:       int
    total_updates:        int
    total_experiences:    int
    buffer_size:          int
    workers_registered:   int
    uptime_seconds:       float


# =============================================================================
# Learner state (module-level singleton, owned by the server process)
# =============================================================================

class Learner:
    def __init__(self, cfg: TrainConfig):
        self.cfg     = cfg
        self.model   = MarketPolicy(hidden_dim=cfg.hidden_dim)
        self.opt     = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        self.policy_version: int = 0
        self.total_updates:  int = 0
        self.total_exp:      int = 0

        # Rolling FIFO experience buffer
        self._buf: collections.deque = collections.deque(maxlen=cfg.max_buffer_size)
        self._buf_lock  = threading.Lock()
        self._train_cond = threading.Condition(self._buf_lock)

        # Worker registry  {worker_id: last_seen_timestamp}
        self._workers: dict = {}
        self._workers_lock  = threading.Lock()

        self._start_time = time.time()

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── Public API (called from FastAPI handlers) ──────────────────────────────

    def register_worker(self, worker_id: str):
        with self._workers_lock:
            self._workers[worker_id] = time.time()
            log.info(f"Worker registered: {worker_id}  (total: {len(self._workers)})")

    def accept_rollout(self, payload: RolloutPayload) -> dict:
        """Decode, validate, and buffer one worker's rollout."""
        lag = self.policy_version - payload.policy_version
        if lag > self.cfg.max_policy_lag:
            return {"status": "rejected", "reason": f"policy lag {lag} > {self.cfg.max_policy_lag}"}

        with self._workers_lock:
            self._workers[payload.worker_id] = time.time()

        T = payload.steps
        def _dec(b64: str, dtype, shape) -> np.ndarray:
            return np.frombuffer(base64.b64decode(b64), dtype=dtype).reshape(shape).copy()

        try:
            obs       = _dec(payload.obs_b64,      np.float32, (T, -1))
            actions   = _dec(payload.actions_b64,  np.int32,   (T, 3))
            rewards   = _dec(payload.rewards_b64,  np.float32, (T,))
            dones     = _dec(payload.dones_b64,    np.float32, (T,))
            log_probs = _dec(payload.log_probs_b64,np.float32, (T,))
            values    = _dec(payload.values_b64,   np.float32, (T,))
            last_obs  = _dec(payload.last_obs_b64, np.float32, (-1,))
        except Exception as exc:
            return {"status": "rejected", "reason": f"decode error: {exc}"}

        # Compute GAE advantages + returns server-side (requires last_obs value)
        with torch.no_grad():
            last_val = self.model(torch.FloatTensor(last_obs).unsqueeze(0))[-1].item()
        advantages, returns = _gae(rewards, values, dones, last_val,
                                   self.cfg.gamma, self.cfg.gae_lambda)

        with self._train_cond:
            for t in range(T):
                self._buf.append({
                    "obs":      obs[t],
                    "action":   actions[t],
                    "ret":      returns[t],
                    "adv":      advantages[t],
                    "log_prob": log_probs[t],
                    "value":    values[t],
                })
            self.total_exp += T
            if len(self._buf) >= self.cfg.min_train_batch:
                self._train_cond.notify()

        log.debug(f"Accepted {T} steps from {payload.worker_id}  "
                  f"(buf={len(self._buf)}, v={self.policy_version})")
        return {"status": "ok", "policy_version": self.policy_version}

    def get_weights(self) -> bytes:
        return self.model.state_bytes()

    def get_status(self) -> dict:
        with self._workers_lock:
            n_workers = len(self._workers)
        with self._buf_lock:
            buf_size = len(self._buf)
        return {
            "policy_version":     self.policy_version,
            "total_updates":      self.total_updates,
            "total_experiences":  self.total_exp,
            "buffer_size":        buf_size,
            "workers_registered": n_workers,
            "uptime_seconds":     round(time.time() - self._start_time, 1),
        }

    # ── Training loop (runs in a background thread) ───────────────────────────

    def training_loop(self):
        """Blocks until min_train_batch experiences available, then runs PPO."""
        log.info("Training loop started.")
        while True:
            with self._train_cond:
                while len(self._buf) < self.cfg.min_train_batch:
                    self._train_cond.wait()
                # Snapshot the buffer for this update
                batch = list(self._buf)[-self.cfg.min_train_batch:]

            self._ppo_update(batch)
            self.total_updates += 1
            self.policy_version += 1

            if self.total_updates % self.cfg.checkpoint_every == 0:
                self._save_checkpoint()

    def _ppo_update(self, batch: list):
        obs       = torch.FloatTensor(np.stack([e["obs"]    for e in batch]))
        actions   = torch.LongTensor (np.stack([e["action"] for e in batch]))
        returns   = torch.FloatTensor(np.array([e["ret"]    for e in batch]))
        advs      = torch.FloatTensor(np.array([e["adv"]    for e in batch]))
        old_lp    = torch.FloatTensor(np.array([e["log_prob"] for e in batch]))

        # Normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        B = len(batch)
        idx = np.arange(B)

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, B, 64):
                mb = idx[start:start + 64]
                mb_obs  = obs[mb]
                mb_act  = actions[mb]
                mb_ret  = returns[mb]
                mb_adv  = advs[mb]
                mb_olp  = old_lp[mb]

                new_lp, entropy, values = self.model.evaluate(mb_obs, mb_act)

                ratio = (new_lp - mb_olp).exp()
                clip  = self.cfg.ppo_clip
                loss_policy = -torch.min(
                    ratio * mb_adv,
                    ratio.clamp(1 - clip, 1 + clip) * mb_adv,
                ).mean()
                loss_value   = 0.5 * (values - mb_ret).pow(2).mean()
                loss_entropy = -entropy.mean()

                loss = (loss_policy
                        + self.cfg.value_coef  * loss_value
                        + self.cfg.entropy_coef * loss_entropy)

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

        log.info(f"[update {self.total_updates}] v={self.policy_version+1}  "
                 f"buf={len(self._buf)}  "
                 f"policy={loss_policy.item():.4f}  "
                 f"value={loss_value.item():.4f}  "
                 f"entropy={(-loss_entropy).item():.4f}")

    def _save_checkpoint(self):
        path = os.path.join(
            self.cfg.checkpoint_dir,
            f"policy_v{self.policy_version}.pt",
        )
        torch.save({
            "policy_version": self.policy_version,
            "total_updates":  self.total_updates,
            "model_state":    self.model.state_dict(),
            "opt_state":      self.opt.state_dict(),
        }, path)
        log.info(f"Checkpoint saved → {path}")


# =============================================================================
# Utility
# =============================================================================

import torch.nn as nn   # needed inside _ppo_update

def _gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        nv   = last_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta     = rewards[t] + gamma * nv * mask - values[t]
        last_gae  = delta + gamma * lam * mask * last_gae
        adv[t]    = last_gae
    return adv, adv + values


# =============================================================================
# FastAPI application factory
# =============================================================================

def build_app(cfg: TrainConfig) -> tuple[FastAPI, Learner]:
    app     = FastAPI(title="Market-Sim Parameter Server", version="1.0")
    learner = Learner(cfg)

    # Start training thread as daemon (dies when main process exits)
    t = threading.Thread(target=learner.training_loop, daemon=True, name="Learner")
    t.start()

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/")
    def root():
        return learner.get_status()

    @app.get("/status", response_model=StatusResponse)
    def status():
        return learner.get_status()

    @app.get("/weights")
    def weights():
        """Return current model weights as raw bytes (application/octet-stream)."""
        data = learner.get_weights()
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"X-Policy-Version": str(learner.policy_version)},
        )

    @app.post("/register")
    def register(worker_id: str):
        learner.register_worker(worker_id)
        return {
            "policy_version": learner.policy_version,
            "config": {
                "rollout_steps":   cfg.rollout_steps,
                "gamma":           cfg.gamma,
                "gae_lambda":      cfg.gae_lambda,
                "num_controllers": cfg.num_controllers,
                "role_config":     cfg.role_config,
                "max_ticks":       cfg.max_ticks,
            },
        }

    @app.post("/experience")
    def experience(payload: RolloutPayload):
        result = learner.accept_rollout(payload)
        return result

    return app, learner
