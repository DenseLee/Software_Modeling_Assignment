"""Heuristic agent runner and test simulation.

Demonstrates the two-tier AI hierarchy:
  • RulerAI  — controller 1 — focuses on economic strength and strategic supply
  • LocalAI  — controller 2 — focuses on private profit and supply-chain balance

Used for manual testing and env validation — no ML model required.

Depends on: everything.
"""
from __future__ import annotations

from ._globals   import globalControllers
from .tick       import computeCoinValue
from .metrics    import (
    computeGDP, computeDiversityScore, computeAvgHappiness, computeAvgLiteracy,
    computeUnemploymentRate, computeProductionEfficiency, computeEventPenalty,
    computeEventMitigationScore, computeTierRatio,
    computePrivateProfit, computeEconomicStrength, computeSupplyChainBalance,
)
from .seasons    import SEASON_NAMES, getEffectiveSeason
from .ai_agents  import LocalAI, RulerAI
from .environment import MarketEnvironment

_PRINT_INTERVAL = 50

# Singleton AI instances (stateless — safe to share)
_LOCAL_AI = LocalAI()
_RULER_AI = RulerAI()


# =============================================================================
# DISPATCHER
# =============================================================================
def _agentAction(ctrl, tick: int) -> dict:
    """Route ctrl to the correct AI based on its ai_role."""
    if ctrl.ai_role == "ruler":
        return _RULER_AI.act(ctrl, tick)
    return _LOCAL_AI.act(ctrl, tick)


# =============================================================================
# STATUS LINE
# =============================================================================
def _statusLine(ctrl, reward: float) -> str:
    coin     = max(computeCoinValue(), 0.01)
    infra    = ctrl.market.infrastructure
    season   = SEASON_NAMES.get(getEffectiveSeason(ctrl), "?")
    event_flag = " [EVENT]" if ctrl.active_events else ""
    role_tag   = f"[{ctrl.ai_role.upper()}]"

    # Tax line — ruler shows revenue collected; local shows tax burden
    if ctrl.ai_role == "ruler":
        tax_str = f"tax_in={ctrl._last_tax_revenue:>6.1f}"
    else:
        recent = [r for r in getattr(
            globalControllers.get(ctrl.ruler_id or -1), 'tax_history', []
        ) if r.local_ctrl_id == ctrl.id]
        paid   = recent[-1].total if recent else 0.0
        tax_str = f"tax_out={paid:>5.1f}"

    return (
        f"  {ctrl.name:<12} {role_tag:<8} | tech={ctrl.techLevel} "
        f"| {season:<6} "
        f"| pop={ctrl.population:>8.1f} "
        f"| GDP={computeGDP(ctrl):>10.1f} "
        f"| tsy={ctrl.treasury:>9.0f} "
        f"| div={computeDiversityScore(ctrl):.2f} "
        f"| hap={computeAvgHappiness(ctrl):.2f} "
        f"| unemp={computeUnemploymentRate(ctrl):.1%} "
        f"| bldgs={len(ctrl.market.buildingList)} "
        f"| infra=L{infra.level}({infra.effective_capacity:.0f}/{infra.max_capacity:.0f}) "
        f"| coin={coin:.2f} "
        f"| {tax_str} "
        f"| rwd={reward:+.4f}"
        f"{event_flag}"
    )


# =============================================================================
# TEST SIMULATION
# =============================================================================
def runTestSimulation(steps: int = 300,
                      verbose_interval: int = _PRINT_INTERVAL) -> MarketEnvironment:
    """Stepped test loop using the two-tier heuristic agent hierarchy.

    Controller 1 = Ruler  (RulerAI: economic strength + strategic resources)
    Controller 2 = Local  (LocalAI: private profit + supply balance)

    The local controller pays taxes to the ruler each tick; the ruler's
    tech advancement rate scales with accumulated tax revenue.
    """
    env = MarketEnvironment(
        num_controllers=2,
        max_ticks=steps,
        role_config={
            1: {"role": "ruler", "name": "Kingdom"},
            2: {"role": "local", "name": "Merchant_Co", "ruler_id": 1},
        },
    )
    obs, info = env.reset()   # bootstrap happens inside reset()

    last_rewards = {cid: 0.0 for cid in globalControllers}

    ruler_ctrl = next((c for c in globalControllers.values() if c.ai_role == "ruler"), None)
    local_ctrl = next((c for c in globalControllers.values() if c.ai_role == "local"), None)

    print(f"\n{'='*80}")
    print(f"  MARKET SIMULATION  —  {steps} ticks  —  2-tier AI")
    if ruler_ctrl and local_ctrl:
        print(f"  Ruler: {ruler_ctrl.name}  |  Local: {local_ctrl.name} "
              f"(under {ruler_ctrl.name})")
    print(f"{'='*80}")

    for tick in range(steps):
        actions = {
            ctrl_id: _agentAction(ctrl, tick)
            for ctrl_id, ctrl in globalControllers.items()
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
        done         = terminated or truncated
        last_rewards = rewards

        if tick % verbose_interval == 0 or tick == steps - 1:
            print(f"\n── Tick {tick:>4} {'─'*60}")
            for ctrl in globalControllers.values():
                print(_statusLine(ctrl, rewards.get(ctrl.id, 0.0)))

        if done:
            ep = info.get("episode", {})
            if ep:
                print(f"\n  Episode ended — reason: {ep.get('reason', '?')}")
                print(f"  Total ticks: {ep.get('length', tick+1)}")
                for cid, ret in ep.get("returns", {}).items():
                    role = globalControllers[cid].ai_role
                    print(f"  Agent {cid} ({role}) return: {ret:+.4f}")
            break

    # ── Final summary ────────────────────────────────────────────────────────
    coin = max(computeCoinValue(), 0.001)
    print(f"\n{'='*80}")
    print("  FINAL SUMMARY")
    print(f"{'='*80}")

    for ctrl in globalControllers.values():
        role = ctrl.ai_role.upper()
        print(f"\n  [{role}] {ctrl}")
        print(f"  GDP:              {computeGDP(ctrl):>14,.2f}")
        print(f"  Treasury:         {ctrl.treasury/coin:>14,.1f} coins")
        print(f"  Diversity:        {computeDiversityScore(ctrl):>14.3f}")
        print(f"  Avg Happiness:    {computeAvgHappiness(ctrl):>14.3f}")
        print(f"  Avg Literacy:     {computeAvgLiteracy(ctrl):>14.3f}")
        print(f"  Upper-tier ratio: {computeTierRatio(ctrl):>14.4f}")
        print(f"  Unemployment:     {computeUnemploymentRate(ctrl):>14.1%}")
        print(f"  Supply balance:   {computeSupplyChainBalance(ctrl):>14.3f}")
        print(f"  Season:           {SEASON_NAMES.get(getEffectiveSeason(ctrl), '?'):>14}")

        if ctrl.ai_role == "ruler":
            tp = ctrl.tax_policy
            if tp:
                print(f"  Tax policy:       income={tp.income_tax_rate:.0%}  "
                      f"building={tp.building_tax_rate:.0%}  "
                      f"trade={tp.trade_tax_rate:.0%}")
            print(f"  Cumulative tax:   {ctrl._cumulative_tax_revenue:>14,.1f}")
            print(f"  Econ strength:    {computeEconomicStrength(ctrl):>14.3f}")
        else:
            ruler = globalControllers.get(ctrl.ruler_id or -1)
            if ruler:
                recent = [r for r in ruler.tax_history
                          if r.local_ctrl_id == ctrl.id]
                if recent:
                    print(f"  Last tax paid:    {recent[-1].total:>14.2f}")
            print(f"  Private profit:   {computePrivateProfit(ctrl):>14.2f}")

        if ctrl.crop_failure_remaining:
            print(f"  Crop failure:     {ctrl.crop_failure_remaining:>12} ticks left")
        if ctrl.long_winter_remaining:
            print(f"  Long winter:      {ctrl.long_winter_remaining:>12} ticks left")

        print(f"  Buildings:        {len(ctrl.market.buildingList):>14}")
        print(f"  Items known:      {len(ctrl.market.itemTracker):>14}")

        active_chains = sum(1 for b in ctrl.market.buildingList.values() if b.activeChain())
        all_outputs   = {tag
                         for b in ctrl.market.buildingList.values()
                         if b.activeChain()
                         for tag in b.activeChain().outputDict.keys()}
        print(f"  Active chains:    {active_chains:>14}")
        print(f"  Output tags:      {sorted(t.name for t in all_outputs)}")

        print(f"  Population classes:")
        for pc in ctrl.pop_classes:
            print(f"    {pc.name:<12}  count={pc.count:>10,.0f}  "
                  f"funds={pc.avg_funds:>8.2f}  lit={pc.literacy:.3f}  "
                  f"hap={pc.happiness:.2f}  sat={pc.satisfaction:.2f}")

        top5 = sorted(ctrl.market.itemTracker.values(),
                      key=lambda u: u.itemPrice * u.supply, reverse=True)[:5]
        print(f"  Top-5 by value:")
        for u in top5:
            print(f"    {u.itemInstance.name:<30} price={u.itemPrice:>9.2f}  "
                  f"supply={u.supply:>8.1f}")

    print(f"\n  Coin value at end: {coin:.4f}")
    print(f"{'='*80}\n")
    return env
