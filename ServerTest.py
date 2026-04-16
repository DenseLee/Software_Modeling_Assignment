"""
Market Simulation Server
========================
FastAPI server that exposes the MarketEnvironment over HTTP.
Supports single-agent and multi-agent sessions.
ngrok tunnel is started automatically on launch.

Run:
    python -m uvicorn ServerTest:app --host 0.0.0.0 --port 8000 --reload

Or directly:
    python ServerTest.py
"""

import os
import threading
import uuid
import asyncio
from typing import Optional

import numpy as np
import market_sim as m

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Market Simulation API", version="1.0")

# =============================================================================
# Model inference  (loaded once at startup)
# =============================================================================
# Set env vars to override which model is used:
#   MODEL_PATH   = path to .zip  (default: market_ppo_agent1_final.zip)
#   VECNORM_PATH = path to .pkl  (default: market_vecnorm_agent1_final.pkl)
# Set DETERMINISTIC=0 to sample stochastically instead of argmax.
# =============================================================================

_MODEL_PATH      = os.getenv("MODEL_PATH",   "market_ppo_agent1_final.zip")
_VECNORM_PATH    = os.getenv("VECNORM_PATH", "market_vecnorm_agent1_final.pkl")
_DETERMINISTIC   = os.getenv("DETERMINISTIC", "1") != "0"

_model      = None   # SB3 PPO
_vecnorm    = None   # VecNormalize (optional, for obs normalisation)
_infer_lock = threading.Lock()


def _load_model():
    global _model, _vecnorm

    if not os.path.exists(_MODEL_PATH):
        print(f"  [infer] Model not found: {_MODEL_PATH}  — /infer disabled")
        return

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        import gymnasium as gym

        # Minimal dummy env — only used so VecNormalize can load its stats.
        class _DummyEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    -1.0, 1.0, shape=(m.STATE_DIM,), dtype=np.float32)
                self.action_space = gym.spaces.MultiDiscrete(list(m.ACTION_SHAPE))
            def reset(self, **kw):
                return np.zeros(m.STATE_DIM, dtype=np.float32), {}
            def step(self, a):
                return np.zeros(m.STATE_DIM, dtype=np.float32), 0.0, False, False, {}

        dummy = DummyVecEnv([_DummyEnv])

        if os.path.exists(_VECNORM_PATH):
            _vecnorm = VecNormalize.load(_VECNORM_PATH, dummy)
            _vecnorm.training    = False
            _vecnorm.norm_reward = False
            _model = PPO.load(_MODEL_PATH, env=_vecnorm)
            print(f"  [infer] Loaded  model   = {_MODEL_PATH}")
            print(f"  [infer]         vecnorm = {_VECNORM_PATH}")
        else:
            _model = PPO.load(_MODEL_PATH)
            print(f"  [infer] Loaded  model   = {_MODEL_PATH}  (no vecnorm)")

        print(f"  [infer] deterministic = {_DETERMINISTIC}")

    except Exception as exc:
        print(f"  [infer] WARNING — model load failed: {exc}")
        print(f"  [infer] /infer endpoint will return HTTP 503")


_load_model()

# =============================================================================
# Session store
# =============================================================================
# Each session holds:
#   env        : MarketEnvironment  (shared tick counter)
#   num_agents : int
#   pending    : dict[ctrl_id, action]  — buffered until all agents submit
#   events     : dict[ctrl_id, asyncio.Event]  — signal per waiting agent
#   result     : the last step() output, shared across all agents in a tick
# =============================================================================
_sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


# =============================================================================
# Schemas
# =============================================================================

class ResetRequest(BaseModel):
    num_agents: int = 1
    max_ticks:  int = 300
    seed:       Optional[int] = None

class ResetResponse(BaseModel):
    session_id: str
    obs:        dict[int, list[float]]   # ctrl_id → obs vector
    state_dim:  int
    action_nvec: list[int]

class StepRequest(BaseModel):
    session_id: str
    ctrl_id:    int = 1
    action:     list[int]               # [type_idx, slot_a, slot_b]

class StepResponse(BaseModel):
    session_id:  str
    ctrl_id:     int
    obs:         list[float]
    reward:      float
    terminated:  bool
    truncated:   bool
    tick:        int
    info:        dict


class InferRequest(BaseModel):
    obs:           list[float]
    deterministic: bool = True

class InferResponse(BaseModel):
    action:        list[int]   # [type_idx, slot_a, slot_b]
    deterministic: bool


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "active_sessions": len(_sessions),
        "state_dim": m.STATE_DIM,
        "action_nvec": list(m.ACTION_SHAPE),
    }


@app.get("/sessions")
def list_sessions():
    return {
        sid: {
            "num_agents": s["num_agents"],
            "tick": s["env"].tick,
            "max_ticks": s["env"].max_ticks,
        }
        for sid, s in _sessions.items()
    }


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    """
    Run the loaded PPO model on a single observation vector.

    - obs must be float[STATE_DIM] (530 values), exactly as returned by /reset or /step.
    - VecNormalize normalisation is applied automatically if the .pkl was loaded.
    - Returns action = [type_idx (0-12), slot_a (0-15), slot_b (0-7)].

    Unity workflow:
        obs  ← /reset or /step response
        action ← /infer(obs)
        ...  ← /step(action)
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Check MODEL_PATH / VECNORM_PATH env vars and server logs.",
        )
    if len(req.obs) != m.STATE_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"obs must have {m.STATE_DIM} values, got {len(req.obs)}.",
        )

    obs = np.array(req.obs, dtype=np.float32).reshape(1, -1)

    with _infer_lock:
        if _vecnorm is not None:
            obs = _vecnorm.normalize_obs(obs)
        action, _ = _model.predict(obs, deterministic=req.deterministic)

    return InferResponse(
        action=action[0].tolist(),
        deterministic=req.deterministic,
    )


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    """
    Create a new simulation session (or reset an existing one).
    Returns the initial observation for every agent.
    """
    if req.num_agents < 1 or req.num_agents > 8:
        raise HTTPException(status_code=400, detail="num_agents must be 1–8")

    env = m.MarketEnvironment(
        num_controllers=req.num_agents,
        max_ticks=req.max_ticks,
        seed=req.seed,
    )
    obs_dict, info = env.reset(seed=req.seed)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "env":        env,
        "num_agents": req.num_agents,
        "pending":    {},        # ctrl_id → action list
        "result":     None,      # filled once all agents submit
        "lock":       asyncio.Lock(),
        "ready_evt":  asyncio.Event(),
    }

    return ResetResponse(
        session_id=session_id,
        obs={cid: list(vec) for cid, vec in obs_dict.items()},
        state_dim=m.STATE_DIM,
        action_nvec=list(m.ACTION_SHAPE),
    )


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    """
    Submit one agent's action.

    Single-agent sessions: steps immediately and returns the result.
    Multi-agent sessions: buffers the action; the last agent to submit
    triggers the tick and all pending requests return together.
    """
    session = _get_session(req.session_id)
    env: m.MarketEnvironment = session["env"]
    num_agents: int = session["num_agents"]

    if req.ctrl_id not in range(1, num_agents + 1):
        raise HTTPException(status_code=400,
                            detail=f"ctrl_id must be 1–{num_agents}")

    if len(req.action) != 3:
        raise HTTPException(status_code=400,
                            detail="action must be [type_idx, slot_a, slot_b]")

    lock: asyncio.Lock  = session["lock"]
    ready: asyncio.Event = session["ready_evt"]

    async with lock:
        session["pending"][req.ctrl_id] = req.action
        all_ready = len(session["pending"]) == num_agents

        if all_ready:
            # This agent is last — drive the tick
            collected = dict(session["pending"])
            session["pending"].clear()

            obs_dict, rewards, terminated, truncated, info = env.step(collected)
            session["result"] = (obs_dict, rewards, terminated, truncated, info)

            # Signal all waiting agents
            ready.set()
            ready.clear()   # reset for the next tick

    if not all_ready:
        # Wait for the driving agent to finish (max 10 s before timeout)
        try:
            await asyncio.wait_for(ready.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408,
                                detail="Timed out waiting for other agents to submit actions.")

    obs_dict, rewards, terminated, truncated, info = session["result"]

    obs    = obs_dict.get(req.ctrl_id, [0.0] * m.STATE_DIM)
    reward = rewards.get(req.ctrl_id, 0.0)

    return StepResponse(
        session_id=req.session_id,
        ctrl_id=req.ctrl_id,
        obs=list(obs),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        tick=env.tick,
        info=info,
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Remove a session to free memory."""
    _get_session(session_id)
    del _sessions[session_id]
    return {"deleted": session_id}


# =============================================================================
# Entry point — starts uvicorn + ngrok tunnel
# =============================================================================
NGROK_STATIC_DOMAIN = "free-kailee-sleepily.ngrok-free.dev"

if __name__ == "__main__":
    import uvicorn
    from pyngrok import ngrok


    tunnel = ngrok.connect(8000, domain=NGROK_STATIC_DOMAIN)
    print(f"\n  Public URL: https://{NGROK_STATIC_DOMAIN}")
    print(f"  Docs:       https://{NGROK_STATIC_DOMAIN}/docs\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
