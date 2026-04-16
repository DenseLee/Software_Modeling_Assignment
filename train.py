"""
ML Training — Market Simulation
================================
Trains a PPO agent on the MarketEnvironment using Stable Baselines3.

Requirements:
    pip install stable-baselines3 gymnasium tensorboard

Run:
    python train.py

Outputs:
    checkpoints/market_ppo_<N>_steps.zip   — periodic saves
    market_policy_final.zip                 — final model
    tb_logs/                                — TensorBoard logs
        tensorboard --logdir tb_logs

Load a trained model:
    from stable_baselines3 import PPO
    model = PPO.load("market_policy_final")
    obs, info = env.reset()
    action, _ = model.predict(obs, deterministic=True)
"""

import multiprocessing as _mp
# Worker subprocesses (SubprocVecEnv) only run the simulation — they don't
# need GPU.  Blocking CUDA here prevents each worker from loading ~2 GB of
# CUDA DLLs, which exhausts the Windows paging file with 10+ processes.
if _mp.current_process().name != "MainProcess":
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import market_sim as m

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


# =============================================================================
# Gymnasium wrapper
# =============================================================================

class MarketGymEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper around market_sim.SingleAgentEnv.

    Observation: Box(530,) float32  in [-1, 1]
    Action:      MultiDiscrete([13, 16, 8])
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ctrl_id:     int  = 1,
        num_agents:  int  = 1,
        max_ticks:   int  = 300,
        seed:        int  = None,
        role_config: dict = None,
    ):
        super().__init__()
        self._ctrl_id   = ctrl_id
        self._multi     = num_agents > 1 or role_config is not None

        if self._multi:
            # Use MarketEnvironment directly — SingleAgentEnv's barrier deadlocks
            # when only one of N controllers ever calls step().
            self._base_env = m.MarketEnvironment(
                num_controllers=num_agents,
                max_ticks=max_ticks,
                seed=seed,
                role_config=role_config or {},
            )
        else:
            base = m.MarketEnvironment(num_controllers=1, max_ticks=max_ticks, seed=seed)
            self._single_env = m.SingleAgentEnv(env=base, ctrl_id=ctrl_id)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(m.STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(list(m.ACTION_SHAPE))

    def reset(self, seed=None, options=None):
        if self._multi:
            obs_dict, info = self._base_env.reset(seed=seed)
            obs = obs_dict.get(self._ctrl_id, [0.0] * m.STATE_DIM)
        else:
            obs, info = self._single_env.reset(seed=seed)
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        if self._multi:
            # Submit only this agent's action; other controllers get no action
            # (MarketEnvironment.step skips missing ctrl_ids gracefully).
            obs_dict, rewards, terminated, truncated, info = self._base_env.step(
                {self._ctrl_id: action.tolist()}
            )
            obs    = obs_dict.get(self._ctrl_id, [0.0] * m.STATE_DIM)
            reward = rewards.get(self._ctrl_id, 0.0)
        else:
            obs, reward, terminated, truncated, info = self._single_env.step(action.tolist())
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def action_masks(self) -> list:
        """Return valid-action boolean mask for MaskablePPO."""
        try:
            if self._multi:
                ctrl = m.globalControllers.get(self._ctrl_id)
            else:
                ctrl = list(m.globalControllers.values())[0]
            if ctrl is not None:
                return m.computeActionMask(ctrl)
        except Exception:
            pass
        # Fallback: all actions valid
        return [True] * (m.N_ACTIONS + m.MAX_BUILDINGS + 8)

    def render(self):
        pass


# =============================================================================
# Training
# =============================================================================

def make_env(seed: int = 0, num_agents: int = 1,
             role_config: dict = None, max_ticks: int = 300):
    """Factory for DummyVecEnv / SubprocVecEnv."""
    def _init():
        env = MarketGymEnv(
            max_ticks=max_ticks,
            num_agents=num_agents,
            role_config=role_config,
        )
        env = Monitor(env)
        return env
    return _init


def train(
    total_timesteps: int   = 500_000,
    n_envs:          int   = 1,
    seed:            int   = 42,
    validate_env:    bool  = True,
    # Hyperparameters
    ent_coef:        float = 0.01,
    net_arch:        list  = None,     # defaults to [256, 256]
    n_steps:         int   = 2048,
    batch_size:      int   = 64,
    learning_rate:   float = 3e-4,
    # Environment config
    num_agents:      int   = 1,
    role_config:     dict  = None,
    max_ticks:       int   = 300,
    # Resume / fine-tune
    resume_model:    str   = None,     # path to .zip to warm-start from
    resume_vecnorm:  str   = None,     # path to .pkl for obs normalisation stats
):
    arch = net_arch or [256, 256]

    # ── Optional: validate the env before training ────────────────────────
    if validate_env:
        print("Validating environment...")
        check_env(MarketGymEnv(), warn=True)
        print("  check_env passed.\n")

    env_kwargs = dict(num_agents=num_agents, role_config=role_config, max_ticks=max_ticks)

    # ── Build vectorised env ──────────────────────────────────────────────
    VecEnvCls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec_env = VecEnvCls([make_env(seed + i, **env_kwargs) for i in range(n_envs)])

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # ── Eval env ──────────────────────────────────────────────────────────
    eval_env = DummyVecEnv([make_env(seed + 999, **env_kwargs)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # ── Callbacks ─────────────────────────────────────────────────────────
    # Save ~20 checkpoints spread across the full run regardless of length.
    ckpt_freq = max(total_timesteps // (n_envs * 20), 1_000)
    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path="./checkpoints/",
        name_prefix="market_ppo",
        save_vecnormalize=True,
    )

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=max(20_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    # ── MaskablePPO model ─────────────────────────────────────────────────
    print(f"  net_arch={arch}  n_steps={n_steps}  batch_size={batch_size}"
          f"  ent_coef={ent_coef}  lr={learning_rate}")
    print(f"  num_agents={num_agents}  bootstrap={'yes' if role_config else 'no'}"
          f"  max_ticks={max_ticks}\n")

    if resume_model:
        # Warm-start: load weights from checkpoint, swap in the new (longer) env.
        # Carry obs-normalisation stats forward so observations stay on the same
        # scale.  Reward normalisation is reset because the reward range shifts
        # with longer episodes.
        print(f"Resuming from: {resume_model}")
        model = MaskablePPO.load(
            resume_model,
            env=vec_env,
            device="cpu",
            custom_objects={
                "n_steps": n_steps,
                "batch_size": batch_size,
                "ent_coef": ent_coef,
                "learning_rate": learning_rate,
                "tensorboard_log": "./tb_logs/",
            },
        )
        if resume_vecnorm:
            # Copy obs running stats from previous VecNormalize; keep reward stats fresh.
            import pickle
            with open(resume_vecnorm, "rb") as fh:
                prev_vn = pickle.load(fh)
            vec_env.obs_rms  = prev_vn.obs_rms
            eval_env.obs_rms = prev_vn.obs_rms
            print(f"Loaded obs normalisation from: {resume_vecnorm}")
        model.set_env(vec_env)
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            seed=seed,
            device="cpu",          # MLP policies are faster on CPU; avoids CUDA paging-file exhaustion
            tensorboard_log="./tb_logs/",
            policy_kwargs={"net_arch": arch},
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
        )

    print(f"Training MaskablePPO for {total_timesteps:,} timesteps across {n_envs} env(s)...")
    print("  TensorBoard: tensorboard --logdir tb_logs\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
    )

    # ── Save final model + normalisation stats ────────────────────────────
    model.save("market_policy_final")
    vec_env.save("market_vecnormalize_final.pkl")
    print("\nSaved: market_policy_final.zip + market_vecnormalize_final.pkl")

    return model, vec_env


# =============================================================================
# Multi-agent training — independent PPO per agent, shared env tick
# =============================================================================

def train_multi(
    num_agents:      int   = 2,
    total_timesteps: int   = 500_000,
    n_envs:          int   = 1,
    max_ticks:       int   = 300,
    seed:            int   = 42,
    validate_env:    bool  = True,
    bootstrap:       bool  = False,
    # Tuned hyperparameters
    ent_coef:        float = 0.003,
    net_arch_size:   int   = 512,
    n_steps:         int   = 4096,
    batch_size:      int   = 256,
    learning_rate:   float = 1e-4,
):
    """
    Train a single shared PPO model across num_agents * n_envs parallel envs,
    with optional bootstrapped economies and tuned hyperparameters.

    bootstrap=True: each env starts with pre-built buildings and balanced
    supply (ruler + local merchant role), giving the agent a richer starting
    state and faster progression through the tech tree.

    Tuned defaults vs. single-agent train():
      ent_coef      0.01  → 0.003   less random exploration, more exploitation
      net_arch      [256,256] → [512,512]  larger network for richer obs
      n_steps       2048  → 4096    longer rollouts, better credit assignment
      batch_size    64    → 256     bigger minibatches, more stable GPU updates
      learning_rate 3e-4  → 1e-4    conservative lr for fine-tuning regime
    """
    total_envs = num_agents * n_envs

    # Build role_config: agent 1 = ruler (has tax policy + market ownership),
    # agents 2+ = local merchants sharing the ruler's market.
    role_config = None
    if bootstrap:
        role_config = {1: {"role": "ruler", "name": "Kingdom"}}
        for i in range(2, num_agents + 1):
            role_config[i] = {"role": "local", "ruler_id": 1,
                              "name": f"Merchant_{i}"}

    train(
        total_timesteps=total_timesteps,
        n_envs=total_envs,
        seed=seed,
        validate_env=validate_env,
        ent_coef=ent_coef,
        net_arch=[net_arch_size, net_arch_size],
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_agents=num_agents,
        role_config=role_config,
        max_ticks=max_ticks,
    )


# =============================================================================
# Inference + verbose replay + CSV export
# =============================================================================

_ACTION_NAMES = [
    "IDLE", "BUILD", "SELECT_CHAIN", "ADD_CHAIN", "PROPOSE_TRADE",
    "SET_EMBARGO", "LIFT_EMBARGO", "BUILD_INFRA", "DEMOLISH",
    "REMOVE_CHAIN", "SET_PRIORITY", "CANCEL_TRADE", "RESPOND_TRADE",
    "SUBSIDIZE",
]

def run_inference(
    model_path:   str  = "market_policy_final",
    vecnorm_path: str  = "market_vecnormalize_final.pkl",
    steps:        int  = 300,
    verbose:      bool = False,
    csv_path:     str  = None,
    bootstrap:    bool = False,
    num_agents:   int  = 1,
):
    """
    Load a trained model and run inference for up to `steps` ticks.

    verbose    — print a status line every tick showing the decoded action
                 and key market metrics.
    csv_path   — if given, write a CSV with per-tick metrics that can be
                 loaded into Excel / pandas / any graphing tool.
    bootstrap  — pre-populate the economy with buildings and supply, matching
                 the bootstrapped training conditions.
    num_agents — number of controllers in the env (should match training).
    """
    import csv

    # ── Build role_config when bootstrapping (mirrors train_multi) ────────
    role_config = None
    if bootstrap and num_agents > 1:
        role_config = {1: {"role": "ruler", "name": "Kingdom"}}
        for i in range(2, num_agents + 1):
            role_config[i] = {"role": "local", "ruler_id": 1,
                              "name": f"Merchant_{i}"}

    # ── Build a raw (unwrapped) env so we can read market state ──────────
    raw_env = MarketGymEnv(max_ticks=steps, num_agents=num_agents,
                           role_config=role_config)
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training   = False
    vec_env.norm_reward = False

    model = MaskablePPO.load(model_path, env=vec_env)

    obs        = vec_env.reset()
    total_reward = 0.0
    rows         = []          # for CSV

    # Column header matches what we collect below
    CSV_COLS = [
        "tick", "action_type", "action_slot_a", "action_slot_b",
        "reward", "total_reward",
        "population", "gdp", "treasury",
        "happiness", "literacy", "unemployment",
        "diversity", "tech_level",
        "num_buildings", "infra_level",
        "bldg_workers", "bldg_capacity", "bldg_worker_fill",
        "subsidy_rate", "avg_funds_t1",
        "invalid_streak",
        "terminated", "truncated",
    ]

    if verbose:
        print(f"\n{'='*76}")
        print(f"  INFERENCE REPLAY  —  model: {model_path}")
        print(f"{'='*76}")
        print(f"  {'tick':>4}  {'action':<18}  {'pop':>8}  {'gdp':>10}  "
              f"{'hap':>5}  {'lit':>5}  {'rwd':>8}  {'bldgs':>5}")
        print(f"  {'─'*74}")

    for t in range(steps):
        # ── Read state BEFORE the step (avoids reading reset state on done) ──
        ctrl = None
        try:
            ctrl = list(m.globalControllers.values())[0]
        except IndexError:
            pass

        if ctrl:
            pop          = sum(pc.count for pc in ctrl.pop_classes)
            gdp          = m.computeGDP(ctrl)
            treasury     = ctrl.treasury
            hap          = m.computeAvgHappiness(ctrl)
            lit          = m.computeAvgLiteracy(ctrl)
            unemp        = m.computeUnemploymentRate(ctrl)
            div_         = m.computeDiversityScore(ctrl)
            tech         = ctrl.techLevel
            bldgs        = len(ctrl.market.buildingList)
            infra_l      = ctrl.market.infrastructure.level
            inv_streak   = getattr(ctrl, '_invalid_streak', 0)
            bld_list     = list(ctrl.market.buildingList.values())
            bldg_workers = sum(b.totalWorkers() for b in bld_list)
            bldg_cap     = sum(b.target_workers for b in bld_list)
            sub_rate     = getattr(ctrl, '_subsidy_rate', 0.0)
            t1_pc        = next((pc for pc in ctrl.pop_classes if pc.tier == 1), None)
            avg_funds_t1 = round(t1_pc.avg_funds, 2) if t1_pc else 0.0
        else:
            pop = gdp = treasury = hap = lit = unemp = div_ = 0.0
            tech = bldgs = infra_l = inv_streak = 0
            bldg_workers = bldg_cap = sub_rate = avg_funds_t1 = 0

        action_masks = get_action_masks(vec_env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, done, info = vec_env.step(action)

        r          = float(reward[0])
        total_reward += r
        act        = action[0]                    # shape (3,) for this agent
        act_name   = _ACTION_NAMES[int(act[0]) % len(_ACTION_NAMES)]
        terminated = bool(info[0].get("terminated", done[0]))
        truncated  = bool(info[0].get("truncated",  False))

        rows.append({
            "tick":           t + 1,
            "action_type":    act_name,
            "action_slot_a":  int(act[1]),
            "action_slot_b":  int(act[2]),
            "reward":         round(r, 6),
            "total_reward":   round(total_reward, 6),
            "population":     round(pop, 1),
            "gdp":            round(gdp, 2),
            "treasury":       round(treasury, 2),
            "happiness":      round(hap, 4),
            "literacy":       round(lit, 4),
            "unemployment":   round(unemp, 4),
            "diversity":      round(div_, 4),
            "tech_level":     tech,
            "num_buildings":  bldgs,
            "infra_level":    infra_l,
            "bldg_workers":   round(bldg_workers, 1),
            "bldg_capacity":  bldg_cap,
            "bldg_worker_fill": round(bldg_workers / bldg_cap, 4) if bldg_cap > 0 else 0.0,
            "subsidy_rate":   round(sub_rate, 2),
            "avg_funds_t1":   avg_funds_t1,
            "invalid_streak": inv_streak,
            "terminated":     int(terminated),
            "truncated":      int(truncated),
        })

        if verbose:
            print(f"  {t+1:>4}  {act_name:<18}  {pop:>8,.0f}  {gdp:>10,.1f}  "
                  f"{hap:>5.3f}  {lit:>5.3f}  {r:>+8.4f}  {bldgs:>5}")

        if done[0]:
            ep = info[0].get("episode", {})
            if verbose and ep:
                print(f"\n  Episode ended — {ep.get('reason', '?')}  "
                      f"return: {total_reward:+.4f}")
            break

    if verbose:
        print(f"\n  Total ticks: {len(rows)}  |  Total reward: {total_reward:+.4f}")
        print(f"{'='*76}\n")
    else:
        print(f"Inference complete — {len(rows)} steps — total reward: {total_reward:+.4f}")

    # ── Write CSV ─────────────────────────────────────────────────────────
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved → {csv_path}")

    return rows


# =============================================================================
# Curriculum training
# =============================================================================

# Three progressive stages.  Each unlocks more of the action space.
# ent_coef decreases stage-by-stage: more exploration early, more exploitation late.
CURRICULUM_STAGES = [
    {
        "name":      "Stage 1 — Construction",
        # IDLE, BUILD_BUILDING, BUILD_INFRA, DEMOLISH_BUILDING
        "allowed":   {0, 1, 7, 8},
        "threshold": 160.0,   # advance when rolling ep_rew_mean >= this
        "ent_coef":  0.01,
    },
    {
        "name":      "Stage 2 — Chain Management",
        # + SELECT_CHAIN, ADD_CHAIN, REMOVE_CHAIN, SET_PRIORITY
        "allowed":   {0, 1, 2, 3, 7, 8, 9, 10},
        "threshold": 185.0,
        "ent_coef":  0.005,
    },
    {
        "name":      "Stage 3 — Full Game",
        # All 13 actions including trade and diplomacy
        "allowed":   set(range(13)),
        "threshold": float("inf"),   # runs to timeout — no reward exit for final stage
        "ent_coef":  0.003,
    },
]


class CurriculumActionWrapper(gym.Wrapper):
    """
    Remaps disallowed action types to IDLE (0, 0, 0).
    Lets the policy explore the full action space while the env only
    executes the actions appropriate for the current curriculum stage.
    """
    def __init__(self, env, allowed_actions: set):
        super().__init__(env)
        self._allowed = frozenset(int(a) for a in allowed_actions)

    def step(self, action):
        if int(action[0]) not in self._allowed:
            action = action.copy()
            action[0] = 0
            action[1] = 0
            action[2] = 0
        return self.env.step(action)


class CurriculumStageCallback(BaseCallback):
    """
    Stops the current training stage when either:
      • Rolling ep_rew_mean (last 100 episodes) >= reward_threshold, OR
      • Wall-clock time >= timeout_sec AND ep_rew_mean >= min_reward_for_timeout.

    The ``min_reward_for_timeout`` gate prevents a timeout from advancing the
    stage when the agent has not learned the current skill yet (e.g. it is still
    stuck spamming no-op actions with negative reward).  Training continues past
    the soft timeout until the agent clears the minimum bar, up to the absolute
    hard_timeout_sec cap, after which the stage advances unconditionally.

    Sets self.advance_reason so the caller knows why training stopped.
    """

    def __init__(
        self,
        reward_threshold:      float,
        timeout_sec:           float,
        min_reward_for_timeout: float = 80.0,
        hard_timeout_sec:      float  = None,   # defaults to 3× timeout_sec
        verbose:               int    = 1,
    ):
        super().__init__(verbose)
        self.reward_threshold       = reward_threshold
        self.timeout_sec            = timeout_sec
        self.min_reward_for_timeout = min_reward_for_timeout
        self.hard_timeout_sec       = hard_timeout_sec if hard_timeout_sec is not None \
                                      else timeout_sec * 3.0
        self.advance_reason   = ""
        self._start_time: float = 0.0
        # Check every this many steps to keep overhead low
        self._check_every = 4096

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self._check_every != 0:
            return True

        elapsed  = time.time() - self._start_time
        mean_rew = (np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                    if len(self.model.ep_info_buffer) >= 20 else None)

        # ── Hard timeout — advance unconditionally ────────────────────────────
        if elapsed >= self.hard_timeout_sec:
            self.advance_reason = (
                f"hard timeout ({elapsed / 3600:.2f}h)"
                + (f", mean_rew={mean_rew:.1f}" if mean_rew is not None else "")
            )
            if self.verbose:
                print(f"\n  [Curriculum] {self.advance_reason} — advancing to next stage")
            return False

        # ── Soft timeout — only advance if agent has cleared the minimum bar ──
        if elapsed >= self.timeout_sec:
            if mean_rew is not None and mean_rew >= self.min_reward_for_timeout:
                self.advance_reason = (
                    f"timeout ({elapsed / 3600:.2f}h), mean_rew={mean_rew:.1f}"
                )
                if self.verbose:
                    print(f"\n  [Curriculum] {self.advance_reason} — advancing to next stage")
                return False
            # Agent hasn't learned enough yet — keep training
            if self.verbose and self.num_timesteps % (self._check_every * 10) == 0:
                bar = f"mean_rew={mean_rew:.1f}" if mean_rew is not None else "too few eps"
                print(f"\n  [Curriculum] soft timeout reached but {bar} "
                      f"< min={self.min_reward_for_timeout:.1f} — extending stage")

        # ── Reward threshold ──────────────────────────────────────────────────
        if (self.reward_threshold < float("inf") and mean_rew is not None):
            if mean_rew >= self.reward_threshold:
                self.advance_reason = (
                    f"reward {mean_rew:.1f} >= threshold {self.reward_threshold}"
                )
                if self.verbose:
                    print(f"\n  [Curriculum] {self.advance_reason} — advancing to next stage")
                return False

        return True


def train_curriculum(
    n_envs:                  int   = 2,   # 2×num_agents = 4 subprocesses; safe on Windows
    num_agents:              int   = 2,
    seed:                    int   = 42,
    bootstrap:               bool  = True,
    net_arch_size:           int   = 512,
    n_steps:                 int   = 4096,
    batch_size:              int   = 256,
    learning_rate:           float = 1e-4,
    stage_timeout_sec:       float = 3600.0,
    # Minimum rolling mean reward before a timeout is allowed to advance the
    # stage.  Keeps the agent in the current stage if it hasn't learned the
    # basics yet (e.g. still spamming no-op actions with negative reward).
    min_reward_for_timeout:  float = 80.0,
    max_ticks:               int   = 300,
):
    """
    Three-stage curriculum PPO training with automatic progression.

    Stages advance when ep_rew_mean >= threshold OR stage_timeout_sec elapses.
    Model weights and VecNormalize obs stats carry over between stages so the
    agent does not forget what it learned.

    Per-stage saves:
        market_policy_construction.zip
        market_policy_chain_management.zip
        market_policy_full_game.zip
    Final save (compatible with /infer endpoint):
        market_policy_final.zip + market_vecnormalize_final.pkl
    """
    # Build role_config once — same across all stages
    role_config = None
    if bootstrap:
        role_config = {1: {"role": "ruler", "name": "Kingdom"}}
        for i in range(2, num_agents + 1):
            role_config[i] = {"role": "local", "ruler_id": 1, "name": f"Merchant_{i}"}

    total_envs = n_envs * num_agents   # e.g. 5 × 2 = 10 parallel envs

    prev_model:   MaskablePPO  = None
    prev_vecnorm: VecNormalize = None

    for s_idx, stage in enumerate(CURRICULUM_STAGES):
        print(f"\n{'='*64}")
        print(f"  {stage['name']}  (stage {s_idx + 1} / {len(CURRICULUM_STAGES)})")
        print(f"  allowed actions : {sorted(stage['allowed'])}")
        print(f"  reward threshold: {stage['threshold']}")
        print(f"  ent_coef        : {stage['ent_coef']}")
        print(f"  timeout         : {stage_timeout_sec / 3600:.1f}h")
        print(f"{'='*64}\n")

        allowed = stage["allowed"]

        def _make_env(seed_: int = 0, _allowed=allowed):
            def _init():
                env = MarketGymEnv(
                    num_agents=num_agents,
                    role_config=role_config,
                    max_ticks=max_ticks,
                )
                env = CurriculumActionWrapper(env, _allowed)
                env = Monitor(env)
                return env
            return _init

        VecCls  = SubprocVecEnv if total_envs > 1 else DummyVecEnv
        vec_env = VecCls([_make_env(seed + i) for i in range(total_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0, clip_reward=10.0)

        # Carry over normalisation stats so obs scaling stays consistent
        if prev_vecnorm is not None:
            vec_env.obs_rms = prev_vecnorm.obs_rms
            vec_env.ret_rms = prev_vecnorm.ret_rms

        eval_env = DummyVecEnv([_make_env(seed + 999)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        if prev_vecnorm is not None:
            eval_env.obs_rms = prev_vecnorm.obs_rms

        stage_slug = stage["name"].split("—")[1].strip().lower().replace(" ", "_")

        checkpoint_cb = CheckpointCallback(
            save_freq=max(20_000 // total_envs, 1),
            save_path=f"./checkpoints/stage{s_idx + 1}/",
            name_prefix="market_ppo",
            save_vecnormalize=True,
        )
        eval_cb = MaskableEvalCallback(
            eval_env,
            best_model_save_path=f"./best_model/stage{s_idx + 1}/",
            log_path=f"./eval_logs/stage{s_idx + 1}/",
            eval_freq=max(40_000 // total_envs, 1),
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        )
        curriculum_cb = CurriculumStageCallback(
            reward_threshold=stage["threshold"],
            timeout_sec=stage_timeout_sec,
            min_reward_for_timeout=min_reward_for_timeout,
        )

        model = MaskablePPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            seed=seed + s_idx,
            device="cpu",      # MLP policies are faster on CPU; avoids CUDA paging-file exhaustion
            tensorboard_log="./tb_logs/",
            policy_kwargs={"net_arch": [net_arch_size, net_arch_size]},
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=stage["ent_coef"],
            learning_rate=learning_rate,
        )

        # Transfer weights from previous stage — agent keeps what it learned
        if prev_model is not None:
            print("  Transferring weights from previous stage...\n")
            model.set_parameters(prev_model.get_parameters(), exact_match=False)

        model.learn(
            total_timesteps=20_000_000,  # large cap — callback drives the real exit
            callback=[checkpoint_cb, eval_cb, curriculum_cb],
            reset_num_timesteps=True,
        )

        # Save this stage's model
        model.save(f"market_policy_{stage_slug}")
        vec_env.save(f"market_vecnorm_{stage_slug}.pkl")
        reason = curriculum_cb.advance_reason or "completed"
        print(f"\n  Stage {s_idx + 1} done — {reason}")
        print(f"  Saved: market_policy_{stage_slug}.zip\n")

        prev_model   = model
        prev_vecnorm = vec_env

    # Final save — same filenames as standard train() so /infer works unchanged
    prev_model.save("market_policy_final")
    prev_vecnorm.save("market_vecnormalize_final.pkl")
    print("\nAll stages complete.")
    print("Final model: market_policy_final.zip + market_vecnormalize_final.pkl")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train / run Market Sim PPO agent")
    parser.add_argument("--mode",         choices=["train", "infer"], default="train")
    parser.add_argument("--multi",        action="store_true",
                        help="Multi-agent training (num_agents controllers per env)")
    parser.add_argument("--num-agents",   type=int,   default=2)
    parser.add_argument("--timesteps",    type=int,   default=500_000)
    parser.add_argument("--n-envs",       type=int,   default=1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--model",        type=str,   default="market_policy_final")
    parser.add_argument("--steps",        type=int,   default=300,
                        help="Max ticks during inference")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--csv",          type=str,   default=None)
    parser.add_argument("--no-validate",   action="store_true")
    # Curriculum
    parser.add_argument("--curriculum",    action="store_true",
                        help="Three-stage curriculum with auto-progression")
    parser.add_argument("--stage-timeout", type=float, default=3600.0,
                        help="Max seconds per curriculum stage (default 3600 = 1h)")
    parser.add_argument("--min-reward-for-timeout", type=float, default=80.0,
                        help="Min rolling mean reward before a timeout can advance the "
                             "stage (default 80.0); prevents advancing when the agent "
                             "hasn't mastered the current skill yet")
    # Episode length
    parser.add_argument("--max-ticks",   type=int,   default=300,
                        help="Episode length in ticks (default 300; try 600 for longer horizon)")
    # Economy bootstrap
    parser.add_argument("--bootstrap",    action="store_true",
                        help="Start each env with pre-built economies (ruler+local role_config)")
    # Hyperparameter overrides
    parser.add_argument("--ent-coef",     type=float, default=None,
                        help="Entropy coef (default 0.003 for multi, 0.01 for single)")
    parser.add_argument("--net-arch-size",type=int,   default=512,
                        help="Hidden layer size, two layers used (default 512 for multi)")
    parser.add_argument("--n-steps",      type=int,   default=4096,
                        help="Rollout steps per env per update (default 4096 for multi)")
    parser.add_argument("--batch-size",   type=int,   default=256,
                        help="PPO minibatch size (default 256 for multi)")
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="Learning rate (default 1e-4 for multi)")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to model .zip to warm-start training from")
    parser.add_argument("--resume-vecnorm", type=str, default=None,
                        help="Path to vecnorm .pkl to carry obs stats forward during resume")
    # Overnight convenience flag
    parser.add_argument("--overnight",    action="store_true",
                        help="Safe overnight preset for R7 8745H: 6 envs, 10M timesteps, "
                             "below-normal process priority, scaled checkpoint interval")
    args = parser.parse_args()

    # ── Overnight preset ──────────────────────────────────────────────────────
    if args.overnight:
        if args.n_envs == 1:
            args.n_envs = 6          # 6 of 8 cores used; 2 free for Windows + thermals
        if args.timesteps == 500_000:
            args.timesteps = 10_000_000   # ~6-8 h on R7 8745H with 6 envs
        # Lower process priority so Windows stays responsive and the laptop
        # doesn't thermal-throttle from competing with foreground tasks.
        try:
            import ctypes
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
            print("[overnight] Process priority set to BELOW_NORMAL")
        except Exception as _e:
            print(f"[overnight] Could not set process priority: {_e}")
        print(f"[overnight] n_envs={args.n_envs}  timesteps={args.timesteps:,}")
        print("[overnight] Checkpoints every ~500k timesteps in ./checkpoints/\n")

    if args.mode == "train":
        if args.curriculum:
            print("Note: curriculum training is disabled — running standard multi-agent training.")
        if args.curriculum or args.multi:
            train_multi(
                num_agents=args.num_agents,
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
                max_ticks=args.max_ticks,
                seed=args.seed,
                validate_env=not args.no_validate,
                bootstrap=args.bootstrap,
                ent_coef=args.ent_coef if args.ent_coef is not None else 0.003,
                net_arch_size=args.net_arch_size,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )
        else:
            train(
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
                seed=args.seed,
                validate_env=not args.no_validate,
                ent_coef=args.ent_coef if args.ent_coef is not None else 0.01,
                net_arch=[args.net_arch_size, args.net_arch_size],
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                max_ticks=args.max_ticks,
                resume_model=args.resume,
                resume_vecnorm=args.resume_vecnorm,
            )
    else:
        run_inference(
            model_path=args.model,
            steps=args.steps,
            verbose=args.verbose,
            csv_path=args.csv,
            bootstrap=args.bootstrap,
            num_agents=args.num_agents,
        )
