from __future__ import annotations

import numpy as np

from rl_ato.env import ATOEnv, ControlAction, Observation
from rl_ato.scenario import ProblemInstance

from .base import (
    BasePolicy,
    backlog_penalty_allocations,
    post_allocation_backlog,
    post_allocation_inventory,
)


class DTPPolicy(BasePolicy):
    name = "DTP"

    def __init__(self, instance: ProblemInstance):
        self.instance = instance

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        if env.scenario is None:
            raise RuntimeError("environment is not initialized")
        allocations = backlog_penalty_allocations(env, obs)
        end_inventory = post_allocation_inventory(env, allocations)
        remaining = post_allocation_backlog(env, allocations)
        target = np.zeros(self.instance.J, dtype=float)
        lookahead_end = int(obs.t + self.instance.max_replenishment_lead_time)
        for product in range(self.instance.I):
            for cohort_period in range(obs.t + 1):
                quantity = float(remaining[product, cohort_period])
                if quantity <= 1e-9 or not env.revealed[product, cohort_period]:
                    continue
                due = (
                    cohort_period
                    + int(self.instance.design_lead_times[product])
                    + int(self.instance.delivery_window)
                )
                if due <= lookahead_end:
                    target += (
                        env.scenario.realized_bom[product, cohort_period]
                        * quantity
                    )
        inventory_position = end_inventory + np.asarray(obs.outstanding, dtype=float)
        orders = np.ceil(np.maximum(target - inventory_position, 0.0) - 1e-9)
        return ControlAction(
            allocations=allocations,
            orders=np.maximum(orders, 0.0),
        )
