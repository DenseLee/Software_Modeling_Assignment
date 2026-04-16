"""
sim_analysis.py
Run the MarketEnvironment for 1000 steps with random actions and print a
detailed economic analysis of the results.
"""

import sys, os, random, statistics
sys.path.insert(0, os.path.dirname(__file__))
import market_sim as m

STEPS = 1000
SEED  = 42

base_env = m.MarketEnvironment(num_controllers=1, max_ticks=STEPS, seed=SEED)
env = m.SingleAgentEnv(env=base_env, ctrl_id=1)
obs, info = env.reset(seed=SEED)

# ── metric history ────────────────────────────────────────────────────────────
history = []

def snap(tick, reward, terminated, truncated):
    ctrl = list(m.globalControllers.values())[0] if m.globalControllers else None
    if ctrl is None:
        return
    history.append({
        "tick":         tick,
        "reward":       reward,
        "gdp":          m.computeGDP(ctrl),
        "treasury":     ctrl.treasury,
        "population":   sum(pc.count for pc in ctrl.pop_classes),
        "happiness":    m.computeAvgHappiness(ctrl),
        "literacy":     m.computeAvgLiteracy(ctrl),
        "unemployment": m.computeUnemploymentRate(ctrl),
        "diversity":    m.computeDiversityScore(ctrl),
        "tech_level":   ctrl.techLevel,
        "buildings":    len(ctrl.market.buildingList),
        "infra_level":  ctrl.market.infrastructure.level,
        "coin_value":   m.computeCoinValue(),
        "terminated":   terminated,
        "truncated":    truncated,
    })

# ── action space bounds ───────────────────────────────────────────────────────
n_act, n_slot_a, n_slot_b = m.ACTION_SHAPE
rng = random.Random(SEED)

print(f"Running {STEPS} steps (random policy, seed={SEED}) …")
total_reward = 0.0
for t in range(STEPS):
    action = [rng.randrange(n_act), rng.randrange(n_slot_a), rng.randrange(n_slot_b)]
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    snap(t + 1, reward, terminated, truncated)
    if terminated or truncated:
        print(f"  Episode ended early at tick {t+1}.")
        break

# ── helper ────────────────────────────────────────────────────────────────────
def col(key):
    return [r[key] for r in history]

def pct_change(series):
    if len(series) < 2 or series[0] == 0:
        return None
    return (series[-1] - series[0]) / abs(series[0]) * 100

# ── print report ──────────────────────────────────────────────────────────────
ticks = len(history)
print(f"\n{'='*60}")
print(f"  MARKET SIM — ANALYSIS  ({ticks} ticks, random policy)")
print(f"{'='*60}")

def row(label, start, end, unit="", fmt=".2f"):
    change = end - start
    pct    = (change / abs(start) * 100) if start != 0 else 0.0
    sign   = "+" if change >= 0 else ""
    print(f"  {label:<22} start={start:{fmt}}{unit}  end={end:{fmt}}{unit}  Δ={sign}{change:{fmt}}{unit} ({sign}{pct:.1f}%)")

gdp  = col("gdp");        row("GDP",           gdp[0],  gdp[-1],  " units")
pop  = col("population"); row("Population",    pop[0],  pop[-1],  "",       ".0f")
trs  = col("treasury");   row("Treasury",      trs[0],  trs[-1],  " coins")
hap  = col("happiness");  row("Avg Happiness", hap[0],  hap[-1],  "",       ".4f")
lit  = col("literacy");   row("Avg Literacy",  lit[0],  lit[-1],  "",       ".4f")
ue   = col("unemployment");row("Unemployment", ue[0],   ue[-1],   "",       ".4f")
div_ = col("diversity");  row("Diversity",     div_[0], div_[-1], "",       ".4f")

tech = col("tech_level")
bldg = col("buildings")
infra= col("infra_level")
coin = col("coin_value")

print(f"\n  {'Tech level':<22} {tech[0]} → {tech[-1]}")
print(f"  {'Buildings':<22} {bldg[0]} → {bldg[-1]}")
print(f"  {'Infra level':<22} {infra[0]} → {infra[-1]}")
print(f"  {'Coin value':<22} {coin[0]:.4f} → {coin[-1]:.4f}")

print(f"\n  {'Total reward':<22} {total_reward:+.4f}")
print(f"  {'Avg reward/tick':<22} {total_reward/ticks:+.6f}")
print(f"  {'Reward range':<22} min={min(col('reward')):+.4f}  max={max(col('reward')):+.4f}")

# GDP trend — split into quarters
q = ticks // 4
gdp_q = [statistics.mean(gdp[i*q:(i+1)*q]) for i in range(4)]
print(f"\n  GDP by quarter (mean per tick):")
for i, v in enumerate(gdp_q):
    print(f"    Q{i+1}: {v:.2f}")

# Happiness trend
hap_q = [statistics.mean(hap[i*q:(i+1)*q]) for i in range(4)]
print(f"\n  Happiness by quarter:")
for i, v in enumerate(hap_q):
    print(f"    Q{i+1}: {v:.4f}")

# unemployment trend
ue_q = [statistics.mean(ue[i*q:(i+1)*q]) for i in range(4)]
print(f"\n  Unemployment by quarter:")
for i, v in enumerate(ue_q):
    print(f"    Q{i+1}: {v:.4f}")

print(f"\n{'='*60}\n")
