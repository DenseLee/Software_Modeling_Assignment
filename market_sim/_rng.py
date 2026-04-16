"""Shared seeded RNG used across the entire simulation.

Always import _rng from here — never create a separate Random() instance.
Seed via _rng.seed(N) inside MarketEnvironment.reset() for reproducibility.
"""
import random as _random_module

_rng = _random_module.Random()
