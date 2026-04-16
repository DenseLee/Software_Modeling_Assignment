"""
distributed/config.py — Training hyperparameters and topology config.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    # ── Network ───────────────────────────────────────────────────────────────
    hidden_dim:         int   = 256

    # ── PPO ───────────────────────────────────────────────────────────────────
    learning_rate:      float = 3e-4
    gamma:              float = 0.99
    gae_lambda:         float = 0.95
    ppo_clip:           float = 0.2
    ppo_epochs:         int   = 4
    value_coef:         float = 0.5
    entropy_coef:       float = 0.01
    max_grad_norm:      float = 0.5

    # ── Rollout / batching ────────────────────────────────────────────────────
    rollout_steps:      int   = 64     # steps each worker collects per submission
    min_train_batch:    int   = 512    # experiences needed before a training update
    max_buffer_size:    int   = 8192   # rolling experience buffer cap (FIFO)
    max_policy_lag:     int   = 15     # reject rollouts older than this many updates

    # ── Weight sync ───────────────────────────────────────────────────────────
    weight_sync_every:  int   = 1      # worker pulls weights after every N rollouts

    # ── Environment ───────────────────────────────────────────────────────────
    max_ticks:          int   = 300
    num_controllers:    int   = 2
    role_config:        dict  = field(default_factory=lambda: {
        1: {"role": "ruler", "name": "Kingdom"},
        2: {"role": "local", "ruler_id": 1, "name": "Merchant"},
    })

    # ── Server ────────────────────────────────────────────────────────────────
    server_port:        int            = 8000
    ngrok_domain:       Optional[str]  = None   # set your static ngrok domain here
    #   e.g.  ngrok_domain = "free-kailee-sleepily.ngrok-free.app"

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_every:   int   = 50     # save model every N training updates
    checkpoint_dir:     str   = "./distributed_checkpoints/"
