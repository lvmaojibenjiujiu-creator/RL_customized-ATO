from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from rl_ato.env import ATOEnv, ControlAction, Observation


class BasePolicy(ABC):
    name = "policy"

    @abstractmethod
    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        raise NotImplementedError


def backlog_penalty_allocations(
    env: ATOEnv,
    obs: Observation,
) -> List[Tuple[int, int, float]]:
    scores = {
        (int(product), int(cohort_period)): float(env.instance.backlog_costs[int(product)])
        for product, cohort_period, remaining, _bom in obs.revealed
        if float(remaining) > 1e-9
    }
    return env.greedy_allocate(scores)


def post_allocation_inventory(
    env: ATOEnv,
    allocations: List[Tuple[int, int, float]],
) -> np.ndarray:
    if env.scenario is None:
        raise RuntimeError("environment is not initialized")
    inventory = np.asarray(env.inventory, dtype=float).copy()
    for product, cohort_period, quantity in allocations:
        inventory -= (
            env.scenario.realized_bom[int(product), int(cohort_period)]
            * float(quantity)
        )
    return np.maximum(inventory, 0.0)


def post_allocation_backlog(
    env: ATOEnv,
    allocations: List[Tuple[int, int, float]],
) -> np.ndarray:
    remaining = np.asarray(env.remaining, dtype=float).copy()
    for product, cohort_period, quantity in allocations:
        i = int(product)
        s = int(cohort_period)
        remaining[i, s] = max(0.0, remaining[i, s] - float(quantity))
    return remaining
