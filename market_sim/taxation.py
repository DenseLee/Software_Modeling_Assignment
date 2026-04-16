"""Tax system: TaxPolicy, collection, and ruler-local economy linkage.

TaxPolicy is attached to a ruler MarketController. Each tick, taxes are
collected from local controllers that name this ruler (via ruler_id) and
transferred into the ruler treasury.

The ruler's cumulative tax revenue directly amplifies the technology
advancement rate — incentivising the ruler AI to grow local economies
rather than over-tax and stunt them.

Depends on: (no package imports — types only accessed via dicts at runtime)
"""
from __future__ import annotations
from dataclasses import dataclass


# =============================================================================
# TAX POLICY
# =============================================================================
@dataclass
class TaxPolicy:
    """Tax rates applied by a ruler to its local economies each tick.

    Rates are fractions [0, max]. At extreme rates, local controllers
    resist (treasury capped at 50%), so diminishing returns set in naturally.
    """
    income_tax_rate:   float = 0.05   # fraction of all building wage bills
    building_tax_rate: float = 0.03   # fraction of estimated building revenues
    trade_tax_rate:    float = 0.02   # fraction of active trade deal value/tick

    def clamp(self) -> TaxPolicy:
        self.income_tax_rate   = max(0.0, min(self.income_tax_rate,   0.50))
        self.building_tax_rate = max(0.0, min(self.building_tax_rate, 0.40))
        self.trade_tax_rate    = max(0.0, min(self.trade_tax_rate,    0.30))
        return self

    @property
    def effective_rate(self) -> float:
        """Simple headline rate: weighted average of all three rates."""
        return (self.income_tax_rate * 0.5
                + self.building_tax_rate * 0.3
                + self.trade_tax_rate * 0.2)


# Phase-tuned presets (used by RulerAI)
TAX_POLICY_EARLY  = TaxPolicy(income_tax_rate=0.02, building_tax_rate=0.02, trade_tax_rate=0.01)
TAX_POLICY_MID    = TaxPolicy(income_tax_rate=0.05, building_tax_rate=0.04, trade_tax_rate=0.02)
TAX_POLICY_LATE   = TaxPolicy(income_tax_rate=0.08, building_tax_rate=0.06, trade_tax_rate=0.03)


# =============================================================================
# TAX RECORD (per-tick snapshot)
# =============================================================================
@dataclass
class TaxRecord:
    """Single-tick tax snapshot for one local controller."""
    tick:               int
    local_ctrl_id:      int
    income_collected:   float = 0.0
    building_collected: float = 0.0
    trade_collected:    float = 0.0

    @property
    def total(self) -> float:
        return self.income_collected + self.building_collected + self.trade_collected


# =============================================================================
# COLLECTION FUNCTION
# =============================================================================
def tickTaxCollection(local_ctrl, ruler_ctrl) -> TaxRecord:
    """Collect taxes from local_ctrl; transfer to ruler_ctrl.

    Called once per tick for each (local, ruler) pair from globalTick.

    Three revenue streams:
      income:   fraction of total wage bills paid to workers this tick
      building: fraction of estimated building output revenues
      trade:    fraction of active trade deal turnover per tick

    Hard cap: at most 50% of the local treasury moves per tick, preventing
    the ruler from bankrupting its own subjects (which would kill tax income).
    """
    policy = ruler_ctrl.tax_policy
    if policy is None:
        return TaxRecord(local_ctrl._current_tick, local_ctrl.id)

    record = TaxRecord(tick=local_ctrl._current_tick, local_ctrl_id=local_ctrl.id)

    # 1. Income tax — wages already paid out this tick (local's own buildings only)
    local_buildings = [b for b in local_ctrl.market.buildingList.values()
                       if b.owner_id == local_ctrl.id]
    total_wages = sum(
        b.worker_wage.get(tier, 0.0) * count
        for b in local_buildings
        for tier, count in b.workers.items()
    )
    record.income_collected = total_wages * policy.income_tax_rate

    # 2. Building revenue tax — estimated from chain output × price (local's own buildings only)
    total_bldg_revenue = sum(
        b.last_revenue for b in local_buildings
    )
    record.building_collected = total_bldg_revenue * policy.building_tax_rate

    # 3. Trade tax — value of goods flowing through active deals
    trade_turnover = sum(
        deal.quantity_per_tick * deal.price_per_unit
        for deal in local_ctrl.activeTradeDeals.values()
        if deal.active
    )
    record.trade_collected = trade_turnover * policy.trade_tax_rate

    # Cap to prevent collapse: max 50% of treasury per tick
    gross = record.total
    cap   = max(0.0, local_ctrl.treasury * 0.50)
    if gross > cap and gross > 0:
        scale = cap / gross
        record.income_collected   *= scale
        record.building_collected *= scale
        record.trade_collected    *= scale

    actual = record.total
    local_ctrl.treasury -= actual
    ruler_ctrl.treasury += actual

    return record
