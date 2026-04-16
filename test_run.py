"""
test_run.py — 100-tick simulation test with full per-tick CSV export.

Outputs:
  run_controllers.csv  — one row per (tick, controller)
  run_buildings.csv    — one row per (tick, building)
"""
import csv
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Validation: confirm required symbols exist before running
# ---------------------------------------------------------------------------
def _validate():
    errors = []

    try:
        from market_sim.structures import (
            BUILDING_FUNDS_CAP, BUILDING_SEED_CAPITAL, Building,
        )
        b = Building(0, __import__("market_sim.structures", fromlist=["BT_FARM"]).BT_FARM)
        if b.funds != 0.0:
            errors.append(f"Building.funds default should be 0.0, got {b.funds}")
        if BUILDING_FUNDS_CAP != 2000.0:
            errors.append(f"BUILDING_FUNDS_CAP expected 2000.0, got {BUILDING_FUNDS_CAP}")
        if BUILDING_SEED_CAPITAL != 200.0:
            errors.append(f"BUILDING_SEED_CAPITAL expected 200.0, got {BUILDING_SEED_CAPITAL}")
    except ImportError as e:
        errors.append(f"structures import: {e}")

    try:
        from market_sim.tick import (
            _ownedBuildings, updateMarketPrices, tickWageBalancing, tickProduction,
            _SPEC_HORIZON, _SPEC_EMA_ALPHA, BUILDING_FUNDS_CAP as TK_CAP,
        )
    except ImportError as e:
        errors.append(f"tick import: {e}")

    try:
        from market_sim.environment import MarketEnvironment, _ownedBuildings
    except ImportError as e:
        errors.append(f"environment import: {e}")

    try:
        from market_sim.metrics import (
            computeGDP, computeReward, computeProductionEfficiency,
            computeDiversityScore, computeUnemploymentRate, computeAvgLiteracy,
            computeAvgHappiness, computePrivateProfit, computeSupplyChainBalance,
            computeEconomicStrength,
        )
    except ImportError as e:
        errors.append(f"metrics import: {e}")

    try:
        from market_sim.taxation import tickTaxCollection
    except ImportError as e:
        errors.append(f"taxation import: {e}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("Validation OK — all symbols present, Building.funds=0 default confirmed.")

_validate()

# ---------------------------------------------------------------------------
# Imports (all validated above)
# ---------------------------------------------------------------------------
from market_sim.environment import MarketEnvironment, _ownedBuildings
from market_sim._globals import globalControllers, globalItemIndex
from market_sim.metrics import (
    computeGDP, computeReward, computeProductionEfficiency,
    computeDiversityScore, computeUnemploymentRate, computeAvgLiteracy,
    computeAvgHappiness, computePrivateProfit, computeSupplyChainBalance,
    computeEconomicStrength,
)
from market_sim.types import itemTag
from market_sim.structures import BUILDING_FUNDS_CAP

# ---------------------------------------------------------------------------
# Simulation setup
# ---------------------------------------------------------------------------
TICKS = 100
SEED  = 42

role_config = {
    1: {"role": "ruler", "name": "Kingdom"},
    2: {"role": "local", "ruler_id": 1, "name": "Merchant"},
}

env = MarketEnvironment(
    num_controllers=2,
    max_ticks=TICKS,
    seed=SEED,
    role_config=role_config,
)
obs, info = env.reset()

# ---------------------------------------------------------------------------
# CSV setup
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).parent
ctrl_csv_path = OUT_DIR / "run_controllers.csv"
bldg_csv_path = OUT_DIR / "run_buildings.csv"

CTRL_FIELDS = [
    "tick", "ctrl_id", "ctrl_name", "role",
    "treasury", "population", "tech_level",
    "gdp",
    "n_owned_buildings", "n_market_buildings",
    "unemployment_pct", "avg_happiness", "avg_literacy",
    "prod_efficiency", "diversity_score", "supply_chain_balance",
    "private_profit",
    "last_tax_revenue", "cumulative_tax_revenue",
    "economic_strength",
    "reward",
    # Market-wide (same value for controllers sharing a market)
    "food_supply", "metal_supply", "lumber_supply", "tools_supply",
    "infra_level", "infra_wear", "infra_eff_capacity", "infra_max_capacity",
    "crop_failure_remaining", "long_winter_remaining",
    # Coin & speculative
    "food_avg_price", "metal_avg_price",
    "wheat_supply", "wheat_price", "wheat_delta_ema", "wheat_spec_demand",
]

BLDG_FIELDS = [
    "tick", "ctrl_id", "ctrl_name", "building_id", "building_type",
    "owner_name",
    "chain_output_tag", "chain_input_tags",
    "workers_total", "worker_efficiency",
    "funds", "funds_pct_of_cap",
    "production_progress", "cycle_count", "stall_ticks",
    "efficiency_rating",
    "operating_revenue", "operating_cost",
    "last_revenue_est", "last_cost_wages",
    "unprofitable_ticks",
]

ctrl_file = open(ctrl_csv_path, "w", newline="", encoding="utf-8")
bldg_file = open(bldg_csv_path, "w", newline="", encoding="utf-8")
ctrl_writer = csv.DictWriter(ctrl_file, fieldnames=CTRL_FIELDS)
bldg_writer = csv.DictWriter(bldg_file, fieldnames=BLDG_FIELDS)
ctrl_writer.writeheader()
bldg_writer.writeheader()

# Precompute stable item IDs
_name_to_id = {i.name: i.id for i in globalItemIndex.values()}
wheat_id = _name_to_id.get("wheat")

# ---------------------------------------------------------------------------
# Helper: speculative demand for a given itemTrackUnit
# ---------------------------------------------------------------------------
_SPEC_HORIZON = 50.0

def _spec_demand(itu) -> float:
    if itu is None or itu.supply_delta_ema >= 0:
        return itu.demand if itu else 0.0
    ticks_to_empty = itu.supply / max(-itu.supply_delta_ema, 0.001)
    if ticks_to_empty >= _SPEC_HORIZON:
        return itu.demand
    urgency = 1.0 - ticks_to_empty / _SPEC_HORIZON
    return itu.demand + urgency * max(itu.supply - itu.demand, 0.0)


# ---------------------------------------------------------------------------
# Tick loop
# ---------------------------------------------------------------------------
print(f"Running {TICKS} ticks...")

for tick in range(1, TICKS + 1):
    actions = {
        1: {"type": 0, "params": {}},
        2: {"type": 0, "params": {}},
    }
    obs, rewards, terminated, truncated, step_info = env.step(actions)

    for cid, ctrl in sorted(globalControllers.items()):
        mkt   = ctrl.market
        owned = _ownedBuildings(ctrl)
        infra = mkt.infrastructure

        # Market-wide item helpers
        def _tag_supply(tag):
            return sum(u.supply for u in mkt.getItemsByTag(tag))

        def _tag_avg_price(tag):
            units = mkt.getItemsByTag(tag)
            return sum(u.itemPrice for u in units) / len(units) if units else 0.0

        wheat_itu = mkt.itemTracker.get(wheat_id)

        ctrl_writer.writerow({
            "tick":                   tick,
            "ctrl_id":                cid,
            "ctrl_name":              ctrl.name,
            "role":                   ctrl.ai_role,
            "treasury":               round(ctrl.treasury, 4),
            "population":             round(ctrl.population, 2),
            "tech_level":             ctrl.techLevel,
            "gdp":                    round(computeGDP(ctrl), 4),
            "n_owned_buildings":      len(owned),
            "n_market_buildings":     len(mkt.buildingList),
            "unemployment_pct":       round(computeUnemploymentRate(ctrl) * 100, 4),
            "avg_happiness":          round(computeAvgHappiness(ctrl), 6),
            "avg_literacy":           round(computeAvgLiteracy(ctrl), 6),
            "prod_efficiency":        round(computeProductionEfficiency(ctrl), 6),
            "diversity_score":        round(computeDiversityScore(ctrl), 6),
            "supply_chain_balance":   round(computeSupplyChainBalance(ctrl), 6),
            "private_profit":         round(computePrivateProfit(ctrl), 4),
            "last_tax_revenue":       round(ctrl._last_tax_revenue, 4),
            "cumulative_tax_revenue": round(ctrl._cumulative_tax_revenue, 4),
            "economic_strength":      round(computeEconomicStrength(ctrl), 6),
            "reward":                 round(rewards.get(cid, 0.0), 6),
            "food_supply":            round(_tag_supply(itemTag.food), 4),
            "metal_supply":           round(_tag_supply(itemTag.metal), 4),
            "lumber_supply":          round(_tag_supply(itemTag.lumber), 4),
            "tools_supply":           round(_tag_supply(itemTag.tools), 4),
            "infra_level":            infra.level,
            "infra_wear":             round(infra.wear, 6),
            "infra_eff_capacity":     round(infra.effective_capacity, 4),
            "infra_max_capacity":     round(infra.max_capacity, 4),
            "crop_failure_remaining": ctrl.crop_failure_remaining,
            "long_winter_remaining":  ctrl.long_winter_remaining,
            "food_avg_price":         round(_tag_avg_price(itemTag.food), 6),
            "metal_avg_price":        round(_tag_avg_price(itemTag.metal), 6),
            "wheat_supply":           round(wheat_itu.supply, 4) if wheat_itu else 0,
            "wheat_price":            round(wheat_itu.itemPrice, 6) if wheat_itu else 0,
            "wheat_delta_ema":        round(wheat_itu.supply_delta_ema, 4) if wheat_itu else 0,
            "wheat_spec_demand":      round(_spec_demand(wheat_itu), 4) if wheat_itu else 0,
        })

        # Building rows — only owned buildings for this controller
        for b in sorted(owned, key=lambda x: x.id):
            chain = b.activeChain()
            owner = globalControllers.get(b.owner_id)
            bldg_writer.writerow({
                "tick":               tick,
                "ctrl_id":            cid,
                "ctrl_name":          ctrl.name,
                "building_id":        b.id,
                "building_type":      b.building_type.name,
                "owner_name":         owner.name if owner else "?",
                "chain_output_tag":   list(chain.outputDict.keys())[0].name if chain and chain.outputDict else "",
                "chain_input_tags":   "|".join(t.name for t in chain.inputDict) if chain else "",
                "workers_total":      round(b.totalWorkers(), 2),
                "worker_efficiency":  round(b.workerEfficiency(), 4),
                "funds":              round(b.funds, 4),
                "funds_pct_of_cap":   round(b.funds / BUILDING_FUNDS_CAP * 100, 2),
                "production_progress":round(b.production_progress, 4),
                "cycle_count":        chain.cycle_count if chain else 0,
                "stall_ticks":        chain.stall_ticks if chain else 0,
                "efficiency_rating":  round(chain.efficiency_rating, 6) if chain else 0,
                "operating_revenue":  round(chain.operating_revenue, 4) if chain else 0,
                "operating_cost":     round(chain.operating_cost, 4) if chain else 0,
                "last_revenue_est":   round(b.last_revenue, 4),
                "last_cost_wages":    round(b.last_cost, 4),
                "unprofitable_ticks": b.unprofitable_ticks,
            })

    if terminated or truncated:
        reason = step_info.get("episode", {}).get("reason", "unknown")
        print(f"  Episode ended early at tick {tick}: {reason}")
        break

ctrl_file.close()
bldg_file.close()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print(f"\nDone. {tick} ticks simulated.")
print(f"  {ctrl_csv_path}")
print(f"  {bldg_csv_path}")

ctrls = list(globalControllers.values())
ruler  = ctrls[0]
local  = ctrls[1]
n_r    = len(_ownedBuildings(ruler))
n_l    = len(_ownedBuildings(local))
n_tot  = len(ruler.market.buildingList)

print(f"\nShared market sanity:")
print(f"  Same object        : {ruler.market is local.market}")
print(f"  Kingdom owns       : {n_r}  |  Merchant owns: {n_l}  |  Total: {n_tot}  |  Accounted: {n_r+n_l==n_tot}")

print(f"\nFinal state:")
for ctrl in ctrls:
    owned = _ownedBuildings(ctrl)
    print(f"  {ctrl.name:<12} treasury={ctrl.treasury:>10.1f}  cum_tax={ctrl._cumulative_tax_revenue:>8.1f}  buildings={len(owned)}")
    for b in sorted(owned, key=lambda x: x.id):
        chain = b.activeChain()
        out   = list(chain.outputDict.keys())[0].name if chain and chain.outputDict else "none"
        print(f"    #{b.id:02d} {b.building_type.name:<10}  out={out:<12}  funds={b.funds:>7.1f}  workers={b.totalWorkers():>4.0f}")
