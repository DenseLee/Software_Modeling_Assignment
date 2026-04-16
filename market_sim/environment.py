"""MarketEnvironment (multi-agent) and SingleAgentEnv (SB3/CleanRL wrapper).

Depends on: everything.
"""
from __future__ import annotations
import math
from typing import Optional

from .types       import itemTag, itemFlag, ActionType, ActionResult
from .structures  import (
    ALL_BUILDING_TYPES, BT_WORKSHOP,
    MAX_TECH_LEVEL, TECH_TIER_NAMES,
    productionGraphs,
)
from ._globals    import globalItemIndex, globalControllers, registerController
from . import _globals
from .controller  import MarketController, _computeMinLivingWage
from .tick        import (
    tickInfrastructureDrain, tickLogisticsInput, tickSubsistence,
    tickConsumption, tickPopQueue, tickProductionProgress, tickLogisticsOutput,
    tickWageBalancing, tickBuildingViability, tickProduction,
    updateMarketPrices, globalTick, computeCoinValue,
    _ownedBuildings,
)
from .metrics     import (
    computeGDP, computeReward, computeProductionEfficiency, computeDiversityScore,
    computeEventPenalty, computeEventMitigationScore, computeUnemploymentRate,
    computeAvgLiteracy, computeAvgHappiness,
    _POP_CAP,
)
from .seasons     import getEffectiveSeason


# =============================================================================
# OBSERVATION / ACTION SPACE DIMENSIONS
# =============================================================================
N_ACTIONS        = len(ActionType)       # 13
MAX_BUILDINGS    = 16
MAX_CHAINS       = 4
MAX_TRADE_SLOTS  = 5
ACTION_SUB       = 8
N_BUILDING_TYPES = len(ALL_BUILDING_TYPES)   # 9
N_ITEM_TAGS      = len(itemTag)              # 37
N_AUTHORED_ITEMS = 61
N_POP_CLASSES    = 4

_OBS_SCALARS   = 23   # +2: subsidy_rate, pop_trend
_OBS_POP       = N_POP_CLASSES * 6           # 24
_OBS_ITEMS     = N_AUTHORED_ITEMS * 5         # 305
_OBS_BUILDINGS = MAX_BUILDINGS * 10           # 160
_OBS_TRADES    = MAX_TRADE_SLOTS * 4          # 20
STATE_DIM      = (_OBS_SCALARS + _OBS_POP
                  + _OBS_ITEMS + _OBS_BUILDINGS + _OBS_TRADES)   # 530

ACTION_SHAPE   = (N_ACTIONS, MAX_BUILDINGS, ACTION_SUB)

# Episode termination thresholds
_MIN_POP_THRESHOLD    = 50.0
_BANKRUPTCY_THRESHOLD = -500.0
_MAX_BANKRUPTCY_TICKS = 20
_SUCCESS_POP_THRESHOLD = 50_000.0


# =============================================================================
# HELPERS
# =============================================================================
def _pop_trend(ctrl) -> float:
    """Population trend over the last 5 ticks, normalised to [-1, +1].
    +1 = growing at ≥1%/tick, -1 = shrinking at ≥1%/tick, 0 = stable.
    """
    hist = getattr(ctrl, '_pop_history', [])
    if len(hist) < 2:
        return 0.0
    window = hist[-min(5, len(hist)):]
    if window[0] <= 0:
        return 0.0
    rate = (window[-1] - window[0]) / window[0]   # total change / base
    return max(-1.0, min(1.0, rate * 100.0))       # scale: 1 % change → ±1


# =============================================================================
# BUILDING PROFIT ESTIMATOR
# =============================================================================
def _estimateBuildingProfit(ctrl, building) -> float:
    """Estimate expected net profit per tick for a newly constructed building.

    Priority:
      1. Same-type peers with real revenue data → average (last_revenue - last_cost).
      2. Full-employment projection from production rules:
           revenue   = out_qty/cycle × cycles_per_tick × best_output_price × glut_factor
           in_cost   = sum(in_qty/cycle × cycles_per_tick × cheapest_input_price) × avail_factor
           wage_cost = target_workers × min_living_wage_tier1
           net       = revenue - in_cost - wage_cost
         Returns the best net profit across all compatible rules.
         glut_factor  ∈ [0.2, 1.0]: penalises saturated output markets.
         avail_factor ∈ [0.1, 1.0]: penalises scarce input supply (stall risk).
    Returns a non-negative float (0.0 when nothing can be estimated).
    """
    btype  = building.building_type
    market = ctrl.market

    # ── Tier 1: peer data ───────────────────────────────────────────────────
    peers = [
        b for b in market.buildingList.values()
        if b.building_type is btype and b.last_revenue > 0
    ]
    if peers:
        avg = sum(b.last_revenue - b.last_cost for b in peers) / len(peers)
        return max(0.0, avg)

    # ── Tier 2: full-employment projection ─────────────────────────────────
    # Cycle quantities mirror production.py::requestProductionChainCreation
    _OUT = {1: 80, 2: 50, 3: 30, 4: 20, 5: 12}
    _IN  = {1: 40, 2: 30, 3: 20, 4: 15, 5: 10}

    # Wage at full employment (tier-1 workers dominate headcount)
    wage_per_worker   = _computeMinLivingWage(ctrl, 1)
    full_wage_cost    = building.target_workers * wage_per_worker

    # Find all production rules this building type can run
    compatible = [
        rule
        for rules in productionGraphs.values()
        for rule in rules
        if (rule.outputTag in btype.allowed_output_tags
            and all(t in btype.allowed_input_tags for t in rule.inputTags)
            and rule.level <= btype.max_chain_level
            and market.hasItemWithTag(rule.outputTag)
            and all(market.hasItemWithTag(t) for t in rule.inputTags))
    ]
    if not compatible:
        return 0.0

    best_net = 0.0
    for rule in compatible:
        lvl             = rule.level
        out_qty         = _OUT.get(lvl, 12)
        in_qty          = _IN.get(lvl, 10)
        ticks_per_cycle = lvl * 10          # timeToProduce = level × 10
        cycles_per_tick = 1.0 / ticks_per_cycle   # at workerEfficiency = 1.0

        # ── Output revenue ──────────────────────────────────────────────
        out_units = market.getItemsByTag(rule.outputTag)
        if not out_units:
            continue
        out_price = max(u.itemPrice for u in out_units)

        # Glut factor: if supply >> demand, we're adding to a saturated market
        total_supply = sum(u.supply  for u in out_units)
        total_demand = sum(u.demand  for u in out_units)
        # Neutral at supply == demand; degrades below 0.2 when supply ≫ demand
        glut_factor  = max(0.2, min(1.0, 2.0 * total_demand / max(total_supply, 1.0)))

        revenue_per_tick = out_qty * cycles_per_tick * out_price * glut_factor

        # ── Input cost + availability ───────────────────────────────────
        input_cost_per_tick = 0.0
        avail_factor        = 1.0
        for in_tag in rule.inputTags:
            in_units = market.getItemsByTag(in_tag)
            if not in_units:
                avail_factor = 0.0
                break
            in_price             = min(u.itemPrice for u in in_units)
            input_cost_per_tick += in_qty * cycles_per_tick * in_price

            # Stall risk: if the market can't sustain 20 ticks of our consumption
            in_supply       = sum(u.supply for u in in_units)
            needed_20_ticks = in_qty * cycles_per_tick * 20
            if in_supply < needed_20_ticks:
                avail_factor = min(avail_factor,
                                   max(0.1, in_supply / needed_20_ticks))

        revenue_per_tick    *= avail_factor
        input_cost_per_tick *= avail_factor   # fewer inputs bought when we stall

        net = revenue_per_tick - input_cost_per_tick - full_wage_cost
        best_net = max(best_net, net)

    return max(0.0, best_net)


# =============================================================================
# MARKET ENVIRONMENT
# =============================================================================
class MarketEnvironment:
    # Action type constants (mirrors ActionType enum values)
    IDLE          = ActionType.IDLE.value
    BUILD         = ActionType.BUILD_BUILDING.value
    SELECT_CHAIN  = ActionType.SELECT_CHAIN.value
    ADD_CHAIN     = ActionType.ADD_CHAIN.value
    PROPOSE_TRADE = ActionType.PROPOSE_TRADE.value
    SET_EMBARGO   = ActionType.SET_EMBARGO.value
    LIFT_EMBARGO  = ActionType.LIFT_EMBARGO.value
    BUILD_INFRA   = ActionType.BUILD_INFRA.value
    DEMOLISH      = ActionType.DEMOLISH_BUILDING.value
    REMOVE_CHAIN  = ActionType.REMOVE_CHAIN.value
    SET_PRIORITY  = ActionType.SET_PRIORITY.value
    CANCEL_TRADE  = ActionType.CANCEL_TRADE.value
    RESPOND_TRADE = ActionType.RESPOND_TRADE.value
    SUBSIDIZE     = ActionType.SUBSIDIZE.value

    observation_space = {"shape": (STATE_DIM,), "dtype": "float32",
                         "low": -1.0, "high": 1.0}
    action_space      = {"shape": ACTION_SHAPE, "dtype": "int64",
                         "nvec": list(ACTION_SHAPE)}

    def __init__(self, num_controllers: int = 2, max_ticks: int = 1000,
                 seed: int = None,
                 role_config: dict = None):
        """
        Args:
            num_controllers: how many MarketController agents to create.
            max_ticks:       episode length before truncation.
            seed:            RNG seed for reproducibility.
            role_config:     optional dict mapping controller_id (1-based) to a
                             config dict with keys ``role`` and optionally
                             ``ruler_id`` and ``name``.  Example::

                                 {1: {"role": "ruler", "name": "Kingdom"},
                                  2: {"role": "local", "ruler_id": 1,
                                      "name": "Merchant_Co"}}

                             When provided, bootstrapEconomy() is called for
                             every controller after reset so the simulation
                             starts with pre-built buildings and balanced supply.
        """
        self.num_controllers = num_controllers
        self.max_ticks       = max_ticks
        self.tick            = 0
        self.seed            = seed
        self.role_config     = role_config or {}
        self._prev_gdp:               dict = {}
        self._episode_return:         dict = {}
        self._consecutive_bankruptcy: dict = {}
        self._termination_reason:     str  = ""

    def reset(self, seed: int = None) -> tuple:
        from ._rng import _rng
        from .structures import productionGraphs
        from .types import TradeDeal

        effective_seed = seed if seed is not None else self.seed
        if effective_seed is not None:
            _rng.seed(effective_seed)

        # Clear procedural items (authored items use IDs 1–99)
        for iid in [k for k in globalItemIndex if k >= 100]:
            del globalItemIndex[iid]
        _globals._nextItemId = 100

        globalControllers.clear()
        TradeDeal._next_id = 0

        self.tick                    = 0
        self._prev_gdp               = {}
        self._episode_return         = {}
        self._consecutive_bankruptcy = {}
        self._termination_reason     = ""

        # Track ruler markets so locals can share them
        ruler_markets: dict = {}   # ruler_id → Market

        for i in range(self.num_controllers):
            cfg      = self.role_config.get(i + 1, {})
            ruler_id = cfg.get("ruler_id", None)
            # Share the ruler's market when the ruler has already been created
            shared_market = ruler_markets.get(ruler_id) if ruler_id else None

            ctrl = MarketController(
                id=i + 1,
                name=cfg.get("name", f"Agent_{i+1}"),
                starting_tech=1,
                starting_population=1000 + i * 200,
                shared_market=shared_market,
            )
            # Apply role config before bootstrap so buildings match the role
            ctrl.ai_role  = cfg.get("role",     "local")
            ctrl.ruler_id = ruler_id
            if ctrl.ai_role == "ruler":
                from .taxation import TaxPolicy
                ctrl.tax_policy = TaxPolicy()
                ruler_markets[ctrl.id] = ctrl.market   # register for locals
            registerController(ctrl)
            self._prev_gdp[ctrl.id]               = 0.0
            self._episode_return[ctrl.id]         = 0.0
            self._consecutive_bankruptcy[ctrl.id] = 0

        # Bootstrap: set initial supplies and place baseline buildings
        if self.role_config:
            self._bootstrap()

        obs  = self._getStateArrays()
        info = {"tick": 0, "state_dim": STATE_DIM, "action_shape": ACTION_SHAPE}
        return obs, info

    def _bootstrap(self) -> None:
        """Place initial buildings and boost commodity supplies for all controllers."""
        from .setup import bootstrapEconomy
        for ctrl in sorted(globalControllers.values(), key=lambda c: c.id):
            bootstrapEconomy(ctrl)
            # Seed initial prices from the now-populated supply
            updateMarketPrices(ctrl.market)

    # ── Action-shaping constants ──────────────────────────────────────────────
    _CANCEL_TRADE_PENALTY       = -0.05   # flat penalty per cancelled trade deal
    _DEMOLISH_PENALTY           = -0.10   # flat penalty per demolished building (any age)
    _PROFITABLE_DEMOLISH_HORIZON= 8       # ticks of lost profit charged when demolishing a profitable building (pool-empty case)
    _PROFITABLE_DEMOLISH_CAP    = -8.0    # max penalty for demolishing a profitable building
    _CHAIN_SWITCH_PENALTY       = -0.04   # penalty for switching a building's chain too soon
    _CHAIN_SWITCH_COOLDOWN      = 15      # ticks a building must keep its chain before free switch
    _INVALID_ACTION_BASE        = -0.02   # base penalty per invalid-action streak tick (×streak, uncapped)

    def step(self, actions) -> tuple:
        """Apply actions and advance the simulation by one tick."""
        if not isinstance(actions, dict):
            actions = {1: actions}

        action_log:    dict = {}
        action_shaping: dict = {}   # ctrl_id → extra reward from action penalties

        for ctrl_id, action in actions.items():
            ctrl = globalControllers.get(ctrl_id)
            if ctrl is None:
                continue
            if not isinstance(action, dict):
                raw_type = int(action[0]) % N_ACTIONS   # model's raw choice before decode
                action   = self._decodeAction(action, ctrl)
            else:
                raw_type = action.get("type", self.IDLE)

            shaping = 0.0
            t = action.get("type", self.IDLE)

            # ── Escalating invalid-action penalty ─────────────────────────
            # If the model chose a non-IDLE action that fell back to IDLE
            # (mask/decode mismatch or residual invalid choice), increment
            # the streak counter and apply an unbounded penalty that grows
            # with each consecutive invalid tick.  Any valid action resets it.
            if t == self.IDLE and raw_type != self.IDLE:
                streak = getattr(ctrl, '_invalid_streak', 0) + 1
                ctrl._invalid_streak = streak
                shaping += self._INVALID_ACTION_BASE * streak
            else:
                ctrl._invalid_streak = 0

            # ── CANCEL_TRADE penalty ──────────────────────────────────────
            if t == self.CANCEL_TRADE:
                shaping += self._CANCEL_TRADE_PENALTY

            # ── DEMOLISH penalty ──────────────────────────────────────────
            # Two-layer penalty:
            #  1. Flat -0.10 on every demolish.
            #  2. Bonus-pool clawback: every BUILD adds to a per-episode pool;
            #     every DEMOLISH claws back up to 5.0 from that pool.  This
            #     ensures BUILD→DEMOLISH is always net -0.10 regardless of
            #     whether the demolished building is the one just built.
            #     Previously the per-building ID lookup could be gamed by
            #     demolishing a *different* old building, leaving the new
            #     building's bonus unclaimed in the dict.
            #  3. If the pool is already empty, charge for lost future earnings
            #     when demolishing a profitable building.
            if t == self.DEMOLISH:
                shaping += self._DEMOLISH_PENALTY
                pool   = getattr(ctrl, "_build_bonus_pool", 0.0)
                bld_id = action.get("params", {}).get("building_id")
                bld    = ctrl.market.buildingList.get(bld_id) if bld_id is not None else None

                if pool > 0.0:
                    # Clawback from pool: caps at one build-bonus worth (5.0)
                    # so each demolish costs exactly one unmatched bonus.
                    clawback = min(pool, 5.0)
                    shaping -= clawback
                    ctrl._build_bonus_pool = max(0.0, pool - clawback)
                elif bld is not None:
                    # Profitable-building penalty: charge for lost future earnings
                    profit_per_tick = bld.last_revenue - bld.last_cost
                    if profit_per_tick > 0.0:
                        penalty = -(profit_per_tick * self._PROFITABLE_DEMOLISH_HORIZON)
                        shaping += max(penalty, self._PROFITABLE_DEMOLISH_CAP)

            # ── PROPOSE_TRADE cooldown ────────────────────────────────────
            if t == self.PROPOSE_TRADE:
                ctrl._last_propose_tick = self.tick

            # ── BUILD_INFRA cooldown ──────────────────────────────────────
            if t == self.BUILD_INFRA:
                ctrl._last_infra_tick = self.tick

            # ── Chain-switching cooldown penalty ──────────────────────────
            # Track last-changed tick per building on the controller.
            # Switching a chain before the cooldown expires incurs a penalty
            # that decays linearly to 0 at the cooldown boundary.
            if t in (self.SELECT_CHAIN, self.ADD_CHAIN, self.REMOVE_CHAIN):
                if not hasattr(ctrl, "_chain_last_changed"):
                    ctrl._chain_last_changed = {}
                bld_id   = action.get("params", {}).get("building_id")
                if bld_id is not None:
                    last = ctrl._chain_last_changed.get(bld_id, -self._CHAIN_SWITCH_COOLDOWN)
                    elapsed = self.tick - last
                    if elapsed < self._CHAIN_SWITCH_COOLDOWN:
                        fraction = 1.0 - elapsed / self._CHAIN_SWITCH_COOLDOWN
                        shaping += self._CHAIN_SWITCH_PENALTY * fraction
                    ctrl._chain_last_changed[bld_id] = self.tick

            result = self._dispatch(ctrl, action)

            # ── BUILD profit bonus + pool tracking ───────────────────────
            # Immediately credit an estimate of the building's expected
            # net profit per tick × a short horizon so the agent doesn't
            # have to wait 30+ ticks for GDP to reflect the investment.
            # Capped at +5.0 to stay on the same scale as other rewards.
            # Every bonus paid is added to _build_bonus_pool; every DEMOLISH
            # claws back up to 5.0 from the pool so cycling is always -0.10
            # net regardless of which building gets demolished.
            #
            # Overbuilding penalty: the bonus is scaled by projected worker
            # fill rate of the new building given available free workers.
            # Below _BUILD_BREAKEVEN_FILL the shaping turns negative so the
            # model learns not to build into an already understaffed economy.
            _BUILD_HORIZON        = 8
            _BUILD_BREAKEVEN_FILL = 0.25   # fill rate where shaping = 0
            _BUILD_MAX_PENALTY    = -2.0   # shaping at proj_fill = 0
            if t == self.BUILD and result.success and result.data is not None:
                est   = _estimateBuildingProfit(ctrl, result.data)

                # Economy-wide worker fill rate across ALL buildings
                # (including the new one that was just added).
                # If existing buildings are already poorly staffed, adding
                # another building spreads workers even thinner.
                all_blds       = list(ctrl.market.buildingList.values())
                total_employed = sum(b.totalWorkers() for b in all_blds)
                total_cap      = sum(b.target_workers for b in all_blds)
                economy_fill   = total_employed / total_cap if total_cap > 0 else 1.0

                if economy_fill >= _BUILD_BREAKEVEN_FILL:
                    # Linear scale: 0 at breakeven, full bonus at fill=1
                    scale = (economy_fill - _BUILD_BREAKEVEN_FILL) / (1.0 - _BUILD_BREAKEVEN_FILL)
                    bonus = min(est * _BUILD_HORIZON * scale, 5.0)
                else:
                    # Penalty: deepens linearly as fill drops below breakeven
                    shortfall = 1.0 - (economy_fill / _BUILD_BREAKEVEN_FILL)
                    bonus = _BUILD_MAX_PENALTY * shortfall

                shaping += bonus
                # Only track positive bonuses in the pool (penalties don't
                # create a DEMOLISH obligation — they're already a cost).
                if bonus > 0.0:
                    ctrl._build_bonus_pool = getattr(ctrl, "_build_bonus_pool", 0.0) + bonus

            action_log[ctrl_id]    = result.message
            action_shaping[ctrl_id] = shaping

        # Full economic tick (order matters).
        # Market-level ticks (infra, population, prices) run once per market;
        # building-level ticks run once per controller on owned buildings only.
        ticked_markets:        set = set()
        prices_updated_markets: set = set()
        for ctrl in sorted(globalControllers.values(), key=lambda c: c.id):
            ctrl._current_tick = self.tick
            market_key = id(ctrl.market)
            if market_key not in ticked_markets:
                tickInfrastructureDrain(ctrl)
                tickSubsistence(ctrl)
                tickConsumption(ctrl)
                tickPopQueue(ctrl)
                ticked_markets.add(market_key)
            tickLogisticsInput(ctrl)
            tickProductionProgress(ctrl)
            tickLogisticsOutput(ctrl)
            tickWageBalancing(ctrl)
            tickBuildingViability(ctrl)
            tickProduction(ctrl)

        # Price update — once per market after all controllers have acted
        for ctrl in sorted(globalControllers.values(), key=lambda c: c.id):
            market_key = id(ctrl.market)
            if market_key not in prices_updated_markets:
                updateMarketPrices(ctrl.market)
                prices_updated_markets.add(market_key)

        globalTick()
        self.tick += 1

        rewards: dict = {}
        for ctrl in globalControllers.values():
            r = computeReward(ctrl, self._prev_gdp.get(ctrl.id, 0.0))
            r += action_shaping.get(ctrl.id, 0.0)

            # ── Population health shaping ─────────────────────────────────
            # Track recent population to detect death spirals.
            pop_hist = getattr(ctrl, '_pop_history', [])
            pop_hist.append(ctrl.population)
            if len(pop_hist) > 10:
                pop_hist.pop(0)
            ctrl._pop_history = pop_hist

            # Death spiral: 5+ consecutive ticks of declining population.
            if len(pop_hist) >= 5:
                window = pop_hist[-5:]
                if all(window[i] > window[i + 1] for i in range(4)):
                    total_decline = window[0] - window[-1]
                    decline_rate  = total_decline / max(window[0], 1.0)
                    r += max(-decline_rate * 5.0, -0.5)   # cap at -0.5/tick

            # Sustained low happiness penalty (below 0.4 hurts per tick)
            avg_hap = computeAvgHappiness(ctrl)
            if avg_hap < 0.4:
                r += (avg_hap - 0.4) * 0.5   # up to -0.2/tick at hap=0

            rewards[ctrl.id] = r
            self._episode_return[ctrl.id] = self._episode_return.get(ctrl.id, 0.0) + r
            self._prev_gdp[ctrl.id]       = computeGDP(ctrl)

        terminated, truncated = self._checkEpisodeDone()
        obs  = self._getStateArrays()
        info: dict = {"tick": self.tick, "log": action_log}
        if terminated or truncated:
            info["episode"] = {
                "returns": dict(self._episode_return),
                "length":  self.tick,
                "reason":  self._termination_reason,
            }
        return obs, rewards, terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _dispatch(self, ctrl: MarketController, action: dict) -> ActionResult:
        t      = action.get("type", self.IDLE)
        params = action.get("params", {})

        if t == self.IDLE:
            return ActionResult(True, "idle")
        if t == self.BUILD:
            return ctrl.action_buildBuilding(
                building_type=params.get("building_type", BT_WORKSHOP),
                initial_chains=params.get("chains", []))
        if t == self.SELECT_CHAIN:
            return ctrl.action_selectProductionChain(params["building_id"],
                                                     params["chain_index"])
        if t == self.ADD_CHAIN:
            explicit_id = params.get("building_id")
            target_bld  = (ctrl.market.buildingList.get(explicit_id)
                           if explicit_id is not None else None)
            result = ctrl.action_requestProductionChain(
                params["output_tag"], params["input_tags"],
                target_building_id=target_bld.id if target_bld else None)
            if not result.success:
                return result
            chain = result.data
            if target_bld is None:
                for bld in _ownedBuildings(ctrl):
                    if bld.isChainCompatible(chain):
                        if not bld.productionChains:
                            target_bld = bld
                            break
                        if target_bld is None:
                            target_bld = bld
            if target_bld is None:
                return ActionResult(False, "no compatible building found for chain")
            return ctrl.action_addChainToBuilding(target_bld.id, chain)
        if t == self.PROPOSE_TRADE:
            target = globalControllers.get(params.get("target_id"))
            if target is None:
                return ActionResult(False, "target not found")
            return ctrl.action_proposeTradeDeal(
                target, params["item_tag"],
                params.get("quantity", 10), params.get("price", 1.0))
        if t == self.SET_EMBARGO:
            target = globalControllers.get(params.get("target_id"))
            return (ctrl.action_setEmbargo(target) if target
                    else ActionResult(False, "target not found"))
        if t == self.LIFT_EMBARGO:
            target = globalControllers.get(params.get("target_id"))
            return (ctrl.action_liftEmbargo(target) if target
                    else ActionResult(False, "target not found"))
        if t == self.BUILD_INFRA:
            return ctrl.action_buildInfrastructure()
        if t == self.DEMOLISH:
            return ctrl.action_demolishBuilding(params["building_id"])
        if t == self.REMOVE_CHAIN:
            return ctrl.action_removeChainFromBuilding(params["building_id"],
                                                       params["chain_index"])
        if t == self.SET_PRIORITY:
            return ctrl.action_setPriority(params["building_id"],
                                           params["chain_index"], params["priority"])
        if t == self.CANCEL_TRADE:
            return ctrl.action_cancelTradeDeal(params["deal_id"])
        if t == self.RESPOND_TRADE:
            return ctrl.action_respondToTradeDeal(params["deal_id"], params["accept"])
        if t == self.SUBSIDIZE:
            level = max(0, min(5, int(params.get("level", 1))))
            _SUBSIDY_LEVELS = {0: 0.0, 1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50}
            ctrl._subsidy_rate = _SUBSIDY_LEVELS[level]
            return ActionResult(True,
                f"subsidy set to {int(ctrl._subsidy_rate * 100)}%")
        return ActionResult(False, f"unknown action type {t}")

    def _checkEpisodeDone(self) -> tuple:
        for ctrl in globalControllers.values():
            if ctrl.population < _MIN_POP_THRESHOLD:
                self._termination_reason = f"collapse:pop:{ctrl.name}"
                return True, False
            if ctrl.treasury < _BANKRUPTCY_THRESHOLD:
                self._consecutive_bankruptcy[ctrl.id] = (
                    self._consecutive_bankruptcy.get(ctrl.id, 0) + 1)
                if self._consecutive_bankruptcy[ctrl.id] >= _MAX_BANKRUPTCY_TICKS:
                    self._termination_reason = f"collapse:bankruptcy:{ctrl.name}"
                    return True, False
            else:
                self._consecutive_bankruptcy[ctrl.id] = 0
            if (ctrl.techLevel >= MAX_TECH_LEVEL
                    and ctrl.population >= _SUCCESS_POP_THRESHOLD):
                self._termination_reason = f"success:{ctrl.name}"
                return True, False
        if self.tick >= self.max_ticks:
            self._termination_reason = "truncated:time_limit"
            return False, True
        return False, False

    def _encodeState(self, ctrl: MarketController) -> list:
        """Encode ctrl's state into a fixed-length float list of size STATE_DIM."""
        vec: list = []
        infra     = ctrl.market.infrastructure
        tick_frac = self.tick / max(self.max_ticks, 1)
        season    = getEffectiveSeason(ctrl)

        # Scalars (21)
        vec += [
            ctrl.techLevel / MAX_TECH_LEVEL,
            math.log1p(ctrl.population)         / math.log1p(_POP_CAP),
            math.log1p(max(ctrl.treasury, 0.0)) / math.log1p(1e7),
            min(computeCoinValue(), 10.0)        / 10.0,
            math.log1p(computeGDP(ctrl))         / math.log1p(1e8),
            computeDiversityScore(ctrl),
            computeProductionEfficiency(ctrl),
            computeEventPenalty(ctrl),
            computeEventMitigationScore(ctrl),
            computeUnemploymentRate(ctrl),
            computeAvgLiteracy(ctrl),
            infra.level                          / 10.0,
            infra.effective_capacity             / max(infra.max_capacity, 1.0),
            infra.wear,
            min(ctrl.crop_failure_remaining, 30) / 30.0,
            min(ctrl.long_winter_remaining,  25) / 25.0,
            season                               / 3.0,
            tick_frac,
            len(ctrl.market.buildingList)        / MAX_BUILDINGS,
            len(ctrl.activeTradeDeals)           / MAX_TRADE_SLOTS,
            1.0 if ctrl.treasury >= 0.0 else 0.0,
            # Subsidy rate and population trend (2 new scalars)
            getattr(ctrl, '_subsidy_rate', 0.0) / 0.50,   # 0→1 normalised
            _pop_trend(ctrl),                              # -1 dying, +1 growing
        ]

        # Pop classes (4 × 6 = 24)
        tier_map = {pc.tier: pc for pc in ctrl.pop_classes}
        for tier in range(1, N_POP_CLASSES + 1):
            pc = tier_map.get(tier)
            if pc:
                vec += [
                    math.log1p(pc.count)      / math.log1p(_POP_CAP),
                    math.log1p(pc.avg_funds)  / math.log1p(10_000.0),
                    pc.literacy,
                    pc.satisfaction,
                    pc.happiness,
                    min(max(pc.wage_ratio, 0.0), 3.0) / 3.0,
                ]
            else:
                vec += [0.0] * 6

        # Authored items (61 × 5 = 305)
        for item_id in range(1, N_AUTHORED_ITEMS + 1):
            itu = ctrl.market.itemTracker.get(item_id)
            if itu is not None:
                vec += [
                    math.log1p(itu.itemPrice)  / math.log1p(10_000.0),
                    math.log1p(itu.supply)     / math.log1p(5_000.0),
                    math.log1p(itu.demand)     / math.log1p(5_000.0),
                    max(-1.0, min(itu.itemScarcity, 2.0)) / 2.0,
                    1.0,
                ]
            else:
                vec += [0.0, 0.0, 0.0, 0.0, 0.0]

        # Building slots (16 × 10 = 160) — owned buildings only
        _btype_names = [bt.name for bt in ALL_BUILDING_TYPES]
        buildings    = sorted(_ownedBuildings(ctrl), key=lambda b: b.id)
        for slot in range(MAX_BUILDINGS):
            if slot < len(buildings):
                b     = buildings[slot]
                chain = b.activeChain()
                try:
                    btype_idx = _btype_names.index(b.building_type.name)
                except ValueError:
                    btype_idx = 0
                out_tag_val = (list(chain.outputDict.keys())[0].value
                               if chain and chain.outputDict else 0)
                if chain and (abs(chain.operating_revenue) + abs(chain.operating_cost)) > 0:
                    margin = ((chain.operating_revenue - chain.operating_cost)
                              / (abs(chain.operating_revenue)
                                 + abs(chain.operating_cost) + 1e-6))
                    profit_margin = (max(-1.0, min(margin, 1.0)) + 1.0) * 0.5
                else:
                    profit_margin = 0.5
                stall_rate = 0.0
                if chain:
                    total      = chain.cycle_count + chain.stall_ticks
                    stall_rate = chain.stall_ticks / max(total, 1)
                vec += [
                    1.0,
                    btype_idx / max(N_BUILDING_TYPES - 1, 1),
                    b.workerEfficiency(),
                    math.log1p(max(b.funds, 0.0)) / math.log1p(10_000.0),
                    min(b.unprofitable_ticks / 20.0, 1.0),
                    out_tag_val / max(N_ITEM_TAGS, 1),
                    chain.efficiency_rating if chain else 0.0,
                    math.log1p(chain.cycle_count if chain else 0) / 10.0,
                    stall_rate,
                    profit_margin,
                ]
            else:
                vec += [0.0] * 10

        # Trade deal slots (5 × 4 = 20)
        deals = list(ctrl.activeTradeDeals.values())[:MAX_TRADE_SLOTS]
        for slot in range(MAX_TRADE_SLOTS):
            if slot < len(deals):
                d = deals[slot]
                vec += [
                    1.0,
                    d.item_tag.value / max(N_ITEM_TAGS, 1),
                    math.log1p(d.quantity_per_tick) / math.log1p(100.0),
                    math.log1p(d.price_per_unit)    / math.log1p(1_000.0),
                ]
            else:
                vec += [0.0, 0.0, 0.0, 0.0]

        assert len(vec) == STATE_DIM, (
            f"_encodeState length mismatch: got {len(vec)}, expected {STATE_DIM}")
        return vec

    def _decodeAction(self, action_vec, ctrl: MarketController) -> dict:
        """Decode flat [type_idx, slot_a, slot_b] into a rich action dict."""
        if len(action_vec) < 3:
            return {"type": self.IDLE, "params": {}}
        t      = int(action_vec[0]) % N_ACTIONS
        slot_a = int(action_vec[1])
        slot_b = int(action_vec[2])

        if t == self.IDLE:
            return {"type": self.IDLE, "params": {}}
        if t == self.BUILD:
            affordable = [bt for bt in ALL_BUILDING_TYPES if ctrl.treasury >= bt.construction_cost]
            if not affordable:
                return {"type": self.IDLE, "params": {}}
            btype = affordable[slot_a % len(affordable)]
            return {"type": self.BUILD, "params": {"building_type": btype}}
        if t == self.BUILD_INFRA:
            return {"type": self.BUILD_INFRA, "params": {}}
        if t == self.ADD_CHAIN:
            rules = ctrl.getVisibleRules()
            if not rules:
                return {"type": self.IDLE, "params": {}}
            rule = rules[slot_a % len(rules)]
            blds = sorted(_ownedBuildings(ctrl), key=lambda b: b.id)
            if not blds:
                return {"type": self.IDLE, "params": {}}
            bld = blds[slot_b % len(blds)]
            return {"type": self.ADD_CHAIN, "params": {
                "building_id": bld.id,
                "output_tag":  rule.outputTag,
                "input_tags":  rule.inputTags,
            }}

        blds = sorted(_ownedBuildings(ctrl), key=lambda b: b.id)
        if not blds:
            return {"type": self.IDLE, "params": {}}
        bld = blds[slot_a % len(blds)]

        if t == self.SELECT_CHAIN:
            if not bld.productionChains:
                return {"type": self.IDLE, "params": {}}
            return {"type": self.SELECT_CHAIN, "params": {
                "building_id": bld.id,
                "chain_index": slot_b % len(bld.productionChains),
            }}
        if t == self.DEMOLISH:
            return {"type": self.DEMOLISH, "params": {"building_id": bld.id}}
        if t == self.REMOVE_CHAIN:
            if len(bld.productionChains) < 2:   # keep at least one chain alive
                return {"type": self.IDLE, "params": {}}
            return {"type": self.REMOVE_CHAIN, "params": {
                "building_id": bld.id,
                "chain_index": slot_b % len(bld.productionChains),
            }}
        if t == self.SET_PRIORITY:
            if not bld.productionChains:
                return {"type": self.IDLE, "params": {}}
            return {"type": self.SET_PRIORITY, "params": {
                "building_id": bld.id,
                "chain_index": slot_b % len(bld.productionChains),
                "priority":    slot_b,
            }}

        others = [c for c in globalControllers.values() if c.id != ctrl.id]
        if not others:
            return {"type": self.IDLE, "params": {}}
        target = others[slot_a % len(others)]

        if t == self.PROPOSE_TRADE:
            tags = list(itemTag)
            tag  = tags[slot_b % len(tags)]
            return {"type": self.PROPOSE_TRADE, "params": {
                "target_id": target.id, "item_tag": tag,
                "quantity": 10.0, "price": 1.0,
            }}
        if t == self.SET_EMBARGO:
            return {"type": self.SET_EMBARGO, "params": {"target_id": target.id}}
        if t == self.LIFT_EMBARGO:
            return {"type": self.LIFT_EMBARGO, "params": {"target_id": target.id}}
        if t == self.CANCEL_TRADE:
            deals = list(ctrl.activeTradeDeals.keys())
            if not deals:
                return {"type": self.IDLE, "params": {}}
            return {"type": self.CANCEL_TRADE, "params": {
                "deal_id": deals[slot_a % len(deals)]}}
        if t == self.RESPOND_TRADE:
            deals = list(ctrl.pendingTradeDeals.keys())
            if not deals:
                return {"type": self.IDLE, "params": {}}
            return {"type": self.RESPOND_TRADE, "params": {
                "deal_id": deals[slot_a % len(deals)],
                "accept":  slot_b > 0,
            }}
        if t == self.SUBSIDIZE:
            # slot_b selects level 0-5; wrap into range
            level = slot_b % 6
            return {"type": self.SUBSIDIZE, "params": {"level": level}}
        return {"type": self.IDLE, "params": {}}

    def _getStateArrays(self) -> dict:
        return {ctrl.id: self._encodeState(ctrl)
                for ctrl in sorted(globalControllers.values(), key=lambda c: c.id)}

    @staticmethod
    def get_state_labels() -> list:
        """Human-readable label for every index in the STATE_DIM vector."""
        labels: list = []
        labels += [
            "tech_level", "population", "treasury", "coin_value", "gdp",
            "diversity", "prod_efficiency", "event_penalty", "event_mitigation",
            "unemployment", "avg_literacy", "infra_level", "infra_cap_ratio",
            "infra_wear", "crop_fail_frac", "long_winter_frac", "season",
            "tick_frac", "num_buildings_frac", "num_trade_deals_frac", "solvent",
        ]
        for tier in range(1, N_POP_CLASSES + 1):
            for feat in ["count", "funds", "literacy", "satisfaction",
                         "happiness", "wage_ratio"]:
                labels.append(f"pop_t{tier}_{feat}")
        _id_to_name = {i.id: i.name for i in globalItemIndex.values()}
        for item_id in range(1, N_AUTHORED_ITEMS + 1):
            n = _id_to_name.get(item_id, f"item_{item_id}")
            for feat in ["price", "supply", "demand", "scarcity", "present"]:
                labels.append(f"{n}_{feat}")
        for slot in range(MAX_BUILDINGS):
            for feat in ["active", "type", "worker_eff", "funds", "unprofitable",
                         "output_tag", "chain_eff", "cycles", "stall_rate",
                         "profit_margin"]:
                labels.append(f"bld{slot:02d}_{feat}")
        for slot in range(MAX_TRADE_SLOTS):
            for feat in ["active", "tag", "qty", "price"]:
                labels.append(f"trade{slot}_{feat}")
        assert len(labels) == STATE_DIM
        return labels


# =============================================================================
# ACTION MASK
# =============================================================================

_MAX_PENDING_TRADES     = 3   # cap on unresolved outgoing proposals before PROPOSE_TRADE is masked
_PROPOSE_TRADE_COOLDOWN = 10  # min ticks between outgoing proposals (prevents consecutive-tick spam)
_BUILD_INFRA_COOLDOWN   = 20  # min ticks between infrastructure upgrades (prevents infra spam)

def computeActionMask(ctrl) -> list:
    """Return a flat boolean list for MaskablePPO.

    Layout mirrors ACTION_SHAPE = (N_ACTIONS, MAX_BUILDINGS, ACTION_SUB):
        first  N_ACTIONS    bools → which action types are available
        next   MAX_BUILDINGS bools → which building slots are occupied
        last   ACTION_SUB   bools → sub-action params (always all True)

    An action type is masked (False) when it can never succeed this tick,
    e.g. CANCEL_TRADE when there are no active deals.
    """
    buildings   = _ownedBuildings(ctrl)
    has_bldgs   = len(buildings) > 0
    has_chains  = any(b.productionChains for b in buildings)
    can_remove_chain = any(len(b.productionChains) >= 2 for b in buildings)
    has_rules   = bool(ctrl.getVisibleRules())
    has_active_deals  = bool(ctrl.activeTradeDeals)
    has_pending_deals = bool(ctrl.pendingTradeDeals)
    has_embargo = bool(ctrl.embargos)
    _last_infra       = getattr(ctrl, '_last_infra_tick', -_BUILD_INFRA_COOLDOWN)
    infra_ready       = (getattr(ctrl, '_current_tick', 0) - _last_infra) >= _BUILD_INFRA_COOLDOWN
    can_afford_infra  = (ctrl.treasury >= ctrl.market.infrastructure.upgrade_cost()
                         and ctrl.techLevel >= ctrl.market.infrastructure.level  # tech-level gate
                         and infra_ready)
    # Only valid if the controller can afford at least one building type at the current slot
    affordable_types  = [bt for bt in ALL_BUILDING_TYPES if ctrl.treasury >= bt.construction_cost]
    can_build         = bool(affordable_types) and not at_bldg_cap
    others_exist = any(c.id != ctrl.id for c in globalControllers.values())
    at_bldg_cap  = len(buildings) >= MAX_BUILDINGS

    # PROPOSE_TRADE: only valid if there are other controllers AND pending
    # outgoing proposals haven't already hit the cap AND a per-proposal
    # cooldown has elapsed (prevents consecutive-tick proposal spam).
    outgoing_proposals = sum(
        1 for d in ctrl.pendingTradeDeals.values()
        if d.from_controller_id == ctrl.id
    )
    last_propose  = getattr(ctrl, '_last_propose_tick', -_PROPOSE_TRADE_COOLDOWN)
    propose_ready = (getattr(ctrl, '_current_tick', 0) - last_propose) >= _PROPOSE_TRADE_COOLDOWN
    can_propose   = others_exist and outgoing_proposals < _MAX_PENDING_TRADES and propose_ready

    # SET_EMBARGO: only valid if the target is NOT already embargoed.
    # (Toggling embargo on/off every tick is a no-op reward hack.)
    non_embargoed_others = any(
        c.id != ctrl.id and c.id not in ctrl.embargos
        for c in globalControllers.values()
    )

    # ── Dimension 0: action types ─────────────────────────────────────────
    action_mask = [
        True,                                    # 0  IDLE          always ok
        can_build,                               # 1  BUILD  (at_bldg_cap already in can_build)
        has_bldgs and has_chains,                # 2  SELECT_CHAIN
        has_bldgs and has_rules,                 # 3  ADD_CHAIN
        can_propose,                             # 4  PROPOSE_TRADE
        non_embargoed_others,                    # 5  SET_EMBARGO
        has_embargo,                             # 6  LIFT_EMBARGO
        can_afford_infra,                        # 7  BUILD_INFRA
        has_bldgs,                               # 8  DEMOLISH
        can_remove_chain,                        # 9  REMOVE_CHAIN
        has_bldgs and has_chains,                # 10 SET_PRIORITY
        has_active_deals,                        # 11 CANCEL_TRADE
        has_pending_deals,                       # 12 RESPOND_TRADE
        True,                                    # 13 SUBSIDIZE      always available
    ]

    # Safety: always keep IDLE valid so the agent is never fully masked
    if not any(action_mask):
        action_mask[0] = True

    # ── Dimension 1: building slots ───────────────────────────────────────
    occupied = [False] * MAX_BUILDINGS
    for i in range(min(len(buildings), MAX_BUILDINGS)):
        occupied[i] = True
    # Always leave at least one slot open for BUILD to target
    if not any(occupied):
        occupied[0] = True

    # ── Dimension 2: sub-action params — always fully valid ───────────────
    sub_mask = [True] * ACTION_SUB

    return action_mask + occupied + sub_mask


# =============================================================================
# SINGLE-AGENT WRAPPER
# =============================================================================
class SingleAgentEnv:
    """Gymnasium-compatible single-agent view over a shared MarketEnvironment.

    Multiple SingleAgentEnv instances can wrap the SAME MarketEnvironment
    (pass the env kwarg). They share one step counter; the last agent to call
    step() drives the actual tick and all blocked callers return simultaneously.

    Usage (single agent, env created internally):
        env = SingleAgentEnv(ctrl_id=1)

    Usage (two agents, one shared env):
        base = MarketEnvironment(num_controllers=2, max_ticks=300)
        env0 = SingleAgentEnv(base, ctrl_id=1)
        env1 = SingleAgentEnv(base, ctrl_id=2)
    """
    observation_space = {"shape": (STATE_DIM,), "dtype": "float32",
                         "low": -1.0, "high": 1.0}
    action_space      = {"shape": ACTION_SHAPE, "dtype": "int64",
                         "nvec": list(ACTION_SHAPE)}

    def __init__(self, env: MarketEnvironment = None, ctrl_id: int = 1):
        import threading
        if env is None:
            env = MarketEnvironment(num_controllers=1, max_ticks=300)
        self._env     = env
        self._ctrl_id = ctrl_id
        if not hasattr(env, "_sync_lock"):
            env._sync_lock     = threading.Lock()
            env._sync_barrier  = threading.Barrier(env.num_controllers)
            env._sync_pending  = {}
            env._last_step_out = None

    @property
    def _lock(self):    return self._env._sync_lock
    @property
    def _barrier(self): return self._env._sync_barrier
    @property
    def _pending(self): return self._env._sync_pending

    def reset(self, seed: int = None) -> tuple:
        import threading
        obs_dict, info = self._env.reset(seed=seed)
        self._env._sync_barrier  = threading.Barrier(self._env.num_controllers)
        self._env._sync_pending  = {}
        self._env._last_step_out = None
        return obs_dict.get(self._ctrl_id, [0.0] * STATE_DIM), info

    def step(self, action) -> tuple:
        with self._lock:
            self._pending[self._ctrl_id] = action
            all_ready = len(self._pending) == self._env.num_controllers
        if all_ready:
            with self._lock:
                collected = dict(self._pending)
                self._pending.clear()
            self._env._last_step_out = self._env.step(collected)
            self._barrier.wait()
        else:
            self._barrier.wait()
        obs_dict, rewards, terminated, truncated, info = self._env._last_step_out
        obs    = obs_dict.get(self._ctrl_id, [0.0] * STATE_DIM)
        reward = rewards.get(self._ctrl_id, 0.0)
        return obs, reward, terminated, truncated, info

    @property
    def ctrl_id(self) -> int: return self._ctrl_id
    @property
    def tick(self) -> int:    return self._env.tick

    def __repr__(self):
        return (f"SingleAgentEnv(ctrl_id={self._ctrl_id}, "
                f"tick={self._env.tick}/{self._env.max_ticks})")


def makeSingleAgentEnvs(num_agents: int = 2, max_ticks: int = 300,
                        seed: int = None) -> list:
    """Factory: one MarketEnvironment, one SingleAgentEnv wrapper per agent."""
    base = MarketEnvironment(num_controllers=num_agents, max_ticks=max_ticks, seed=seed)
    return [SingleAgentEnv(env=base, ctrl_id=i + 1) for i in range(num_agents)]
