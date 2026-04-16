"""
distributed/worker.py — Rollout collector + HTTP client.

Each worker:
  1. Registers with the parameter server.
  2. Downloads current model weights.
  3. Runs a local MarketEnvironment for `rollout_steps` ticks.
  4. POSTs the rollout batch to the server.
  5. Syncs weights every `weight_sync_every` rollouts.
  6. Repeats indefinitely.

Multiple workers can run on the same machine (different ctrl_ids or processes).
Workers are stateless with respect to the model — the server is the source of truth.
"""
from __future__ import annotations

import base64
import logging
import socket
import os
import sys
import time
from typing import Optional

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_sim.environment import MarketEnvironment, _ownedBuildings
from market_sim._globals    import globalControllers

from .model  import MarketPolicy
from .config import TrainConfig

log = logging.getLogger("dist.worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# =============================================================================
# Worker
# =============================================================================

class Worker:
    """
    A single worker process: collects rollouts from one MarketEnvironment
    and ships them to the parameter server.

    Parameters
    ----------
    server_url  : e.g. "https://free-xxx.ngrok-free.app"
    ctrl_id     : which controller in the environment this worker trains (1 or 2)
    cfg         : TrainConfig (can be fetched from server at startup)
    worker_id   : unique identifier string (defaults to hostname + pid)
    """

    def __init__(
        self,
        server_url: str,
        ctrl_id:    int         = 1,
        cfg:        Optional[TrainConfig] = None,
        worker_id:  Optional[str] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.ctrl_id    = ctrl_id
        # Sanitise to ASCII — hostnames with non-ASCII chars break URL paths
        raw_host = socket.gethostname()
        safe_host = raw_host.encode("ascii", errors="ignore").decode() or "worker"
        safe_host = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_host)
        self.worker_id  = worker_id or f"{safe_host}_c{ctrl_id}_pid{os.getpid()}"
        self.model      = MarketPolicy()
        self.policy_version = -1
        self.rollout_count  = 0

        # Register and fetch config from server
        self.cfg = cfg or TrainConfig()
        self._register()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _register(self):
        url = f"{self.server_url}/register"
        for attempt in range(10):
            try:
                r = requests.post(url, params={"worker_id": self.worker_id}, timeout=15)
                r.raise_for_status()
                data = r.json()
                self.policy_version = data["policy_version"]
                # Merge server config into local cfg (server is authoritative)
                sc = data.get("config", {})
                if sc:
                    self.cfg.rollout_steps   = sc.get("rollout_steps",   self.cfg.rollout_steps)
                    self.cfg.gamma           = sc.get("gamma",           self.cfg.gamma)
                    self.cfg.gae_lambda      = sc.get("gae_lambda",      self.cfg.gae_lambda)
                    self.cfg.num_controllers = sc.get("num_controllers", self.cfg.num_controllers)
                    self.cfg.role_config     = sc.get("role_config",     self.cfg.role_config)
                    self.cfg.max_ticks       = sc.get("max_ticks",       self.cfg.max_ticks)
                log.info(f"Registered as {self.worker_id} | server policy v={self.policy_version}")
                break
            except Exception as exc:
                log.warning(f"Register attempt {attempt+1}/10 failed: {exc}")
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"Could not register with server at {self.server_url}")

    def _sync_weights(self):
        """Pull latest weights from the parameter server."""
        url = f"{self.server_url}/weights"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            self.model.load_bytes(r.content)
            self.policy_version = int(r.headers.get("X-Policy-Version", self.policy_version))
            log.info(f"Weights synced — policy v={self.policy_version}")
        except Exception as exc:
            log.warning(f"Weight sync failed: {exc}")

    # ── Environment management ─────────────────────────────────────────────────

    def _make_env(self) -> tuple[MarketEnvironment, dict]:
        """Create a fresh MarketEnvironment and return (env, obs_dict)."""
        env = MarketEnvironment(
            num_controllers=self.cfg.num_controllers,
            max_ticks=self.cfg.max_ticks,
            role_config=self.cfg.role_config,
        )
        obs_dict, _ = env.reset()
        return env, obs_dict

    # ── Rollout collection ─────────────────────────────────────────────────────

    def _collect_rollout(self, env: MarketEnvironment, obs_dict: dict):
        """
        Collect `cfg.rollout_steps` steps for ctrl_id using the current policy.

        Returns
        -------
        (trajectory, next_obs_dict, done)
          trajectory : dict of numpy arrays, each length rollout_steps
          next_obs   : obs array for ctrl_id after the last step (for bootstrap)
          done       : bool — whether the episode ended
        """
        T = self.cfg.rollout_steps

        obs_list       = []
        action_list    = []
        reward_list    = []
        done_list      = []
        log_prob_list  = []
        value_list     = []

        obs = np.array(obs_dict[self.ctrl_id], dtype=np.float32)
        done = False

        for _ in range(T):
            action, log_prob, value = self.model.act(obs)

            # Build actions for ALL controllers (others use idle)
            all_actions = {
                cid: (action if cid == self.ctrl_id else [0, 0, 0])
                for cid in range(1, self.cfg.num_controllers + 1)
            }

            next_obs_dict, rewards, terminated, truncated, _ = env.step(all_actions)
            reward = float(rewards.get(self.ctrl_id, 0.0))
            episode_done = terminated or truncated

            obs_list.append(obs.copy())
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(float(episode_done))
            log_prob_list.append(log_prob)
            value_list.append(value)

            obs = np.array(next_obs_dict[self.ctrl_id], dtype=np.float32)

            if episode_done:
                done = True
                break

        trajectory = {
            "obs":       np.stack(obs_list).astype(np.float32),
            "actions":   np.array(action_list, dtype=np.int32),
            "rewards":   np.array(reward_list, dtype=np.float32),
            "dones":     np.array(done_list,   dtype=np.float32),
            "log_probs": np.array(log_prob_list, dtype=np.float32),
            "values":    np.array(value_list,  dtype=np.float32),
            "last_obs":  obs,   # bootstrap obs
        }
        return trajectory, next_obs_dict, done

    # ── Submission ────────────────────────────────────────────────────────────

    @staticmethod
    def _enc(arr: np.ndarray) -> str:
        return base64.b64encode(arr.tobytes()).decode()

    def _submit_rollout(self, traj: dict) -> dict:
        """POST a rollout batch to the parameter server."""
        T = len(traj["rewards"])

        # Get role for this ctrl_id from cfg
        rc   = self.cfg.role_config or {}
        role = rc.get(self.ctrl_id, {}).get("role", "local")

        payload = {
            "worker_id":      self.worker_id,
            "ctrl_id":        self.ctrl_id,
            "role":           role,
            "policy_version": self.policy_version,
            "steps":          T,
            "obs_b64":        self._enc(traj["obs"]),
            "actions_b64":    self._enc(traj["actions"]),
            "rewards_b64":    self._enc(traj["rewards"]),
            "dones_b64":      self._enc(traj["dones"]),
            "log_probs_b64":  self._enc(traj["log_probs"]),
            "values_b64":     self._enc(traj["values"]),
            "last_obs_b64":   self._enc(traj["last_obs"].astype(np.float32)),
        }

        for attempt in range(5):
            try:
                r = requests.post(
                    f"{self.server_url}/experience",
                    json=payload,
                    timeout=30,
                )
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                log.warning(f"Submit attempt {attempt+1}/5 failed: {exc}")
                time.sleep(2 ** attempt)
        return {"status": "failed"}

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        """
        Main worker loop. Runs indefinitely until interrupted.
        """
        log.info(f"Worker {self.worker_id} starting (ctrl_id={self.ctrl_id})")
        self._sync_weights()

        env, obs_dict = self._make_env()
        ep_steps      = 0
        ep_return     = 0.0
        ep_count      = 0

        while True:
            traj, obs_dict, done = self._collect_rollout(env, obs_dict)

            T = len(traj["rewards"])
            ep_steps  += T
            ep_return += traj["rewards"].sum()

            result = self._submit_rollout(traj)
            self.rollout_count += 1

            status = result.get("status", "?")
            log.info(
                f"Rollout {self.rollout_count:>4} | steps={T:>3} | "
                f"ret={traj['rewards'].sum():+.3f} | submit={status} | "
                f"local_v={self.policy_version}"
            )

            if done:
                ep_count += 1
                log.info(
                    f"  Episode {ep_count} done — "
                    f"total_steps={ep_steps}  total_return={ep_return:.3f}"
                )
                ep_steps = ep_return = 0.0
                env, obs_dict = self._make_env()

            # Sync weights periodically
            if self.rollout_count % self.cfg.weight_sync_every == 0:
                self._sync_weights()
