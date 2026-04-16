"""
distributed/model.py — Actor-critic policy for the market simulation.

Action space: MultiDiscrete([N_ACTIONS, MAX_BUILDINGS, ACTION_SUB])
  Three independent categorical heads; log_prob = sum of the three.

Observation: Box(STATE_DIM,) float32
"""
from __future__ import annotations
import io
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Resolve dims from the local market_sim package
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_sim.environment import STATE_DIM, N_ACTIONS, MAX_BUILDINGS, ACTION_SUB


class MarketPolicy(nn.Module):
    """Shared-parameter actor-critic for all agent roles (ruler + local)."""

    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = 256):
        super().__init__()
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim

        # Shared feature backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor: three independent categorical distributions
        self.head_type     = nn.Linear(hidden_dim, N_ACTIONS)
        self.head_building = nn.Linear(hidden_dim, MAX_BUILDINGS)
        self.head_sub      = nn.Linear(hidden_dim, ACTION_SUB)

        # Critic: scalar state-value estimate
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Policy heads need smaller scale to start near-uniform
        for head in (self.head_type, self.head_building, self.head_sub):
            nn.init.orthogonal_(head.weight, gain=0.01)
        # Value head: small scale
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor):
        """
        obs : (batch, state_dim)
        Returns (logits_type, logits_building, logits_sub, value)
        """
        x = self.backbone(obs)
        return (
            self.head_type(x),
            self.head_building(x),
            self.head_sub(x),
            self.value_head(x).squeeze(-1),
        )

    # ── Inference (single step) ────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = False):
        """
        Select an action for one observation.

        Returns
        -------
        action   : list[int]  — [type_idx, building_idx, sub_idx]
        log_prob : float      — sum of log-probs (for PPO importance sampling)
        value    : float      — critic estimate
        """
        obs = torch.FloatTensor(obs_np).unsqueeze(0)
        lt, lb, ls, value = self.forward(obs)

        if deterministic:
            at, ab, as_ = lt.argmax(-1), lb.argmax(-1), ls.argmax(-1)
            log_prob = 0.0
        else:
            dt, db, ds = Categorical(logits=lt), Categorical(logits=lb), Categorical(logits=ls)
            at, ab, as_ = dt.sample(), db.sample(), ds.sample()
            log_prob = (dt.log_prob(at) + db.log_prob(ab) + ds.log_prob(as_)).item()

        return [at.item(), ab.item(), as_.item()], log_prob, value.item()

    # ── Evaluation (batch) ────────────────────────────────────────────────────

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Batch evaluation for PPO update.

        Parameters
        ----------
        obs     : (T, state_dim)
        actions : (T, 3)  — [type, building, sub]

        Returns
        -------
        log_probs : (T,)
        entropy   : (T,)
        values    : (T,)
        """
        lt, lb, ls, values = self.forward(obs)
        dt, db, ds = Categorical(logits=lt), Categorical(logits=lb), Categorical(logits=ls)

        log_probs = (
            dt.log_prob(actions[:, 0]) +
            db.log_prob(actions[:, 1]) +
            ds.log_prob(actions[:, 2])
        )
        entropy = dt.entropy() + db.entropy() + ds.entropy()
        return log_probs, entropy, values

    # ── Serialisation ─────────────────────────────────────────────────────────

    def state_bytes(self) -> bytes:
        """Serialise state_dict to raw bytes (for HTTP transfer)."""
        buf = io.BytesIO()
        torch.save(self.state_dict(), buf)
        return buf.getvalue()

    def load_bytes(self, data: bytes):
        """Load state_dict from raw bytes received over HTTP."""
        buf = io.BytesIO(data)
        state = torch.load(buf, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
