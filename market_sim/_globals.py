"""Module-level simulation state shared across all subsystems.

All mutable globals live here. Other modules import this module by name
and access attributes directly (e.g. ``_globals._nextItemId += 1``) so
that mutations are visible everywhere — never do ``from ._globals import
_nextItemId`` if you intend to write to it.
"""
from __future__ import annotations

globalItemIndex:   dict = {}   # item_id → item
globalControllers: dict = {}   # controller_id → MarketController
_nextItemId:       int  = 100  # procedural items start at 100; authored items use 1–99


def registerController(ctrl) -> None:
    globalControllers[ctrl.id] = ctrl
