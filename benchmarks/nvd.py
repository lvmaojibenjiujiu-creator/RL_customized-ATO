from __future__ import annotations

import numpy as np
from scipy.stats import norm

from rl_ato.env import ATOEnv, ControlAction, Observation
from rl_ato.scenario import ProblemInstance

from .base import BasePolicy, backlog_penalty_allocations, post_allocation_inventory


class NVDPolicy(BasePolicy):
    name = "NVD"

    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        self.base_stock = self._base_stock(instance)

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        allocations = backlog_penalty_allocations(env, obs)
        end_inventory = post_allocation_inventory(env, allocations)
        inventory_position = end_inventory + np.asarray(obs.outstanding, dtype=float)
        orders = np.ceil(np.maximum(self.base_stock - inventory_position, 0.0) - 1e-9)
        return ControlAction(
            allocations=allocations,
            orders=np.maximum(orders, 0.0),
        )

    @staticmethod
    def _base_stock(instance: ProblemInstance) -> np.ndarray:
        target = np.zeros(instance.J, dtype=float)
        expected_lead_time = float(instance.expected_lead_time)
        if instance.demand_pattern.lower().startswith("season"):
            periods = np.arange(instance.T, dtype=float)
            means = instance.demand_lambdas[:, None] * (
                1.0
                + instance.seasonal_beta
                * np.sin(
                    2.0 * np.pi * periods[None, :] / max(1, instance.seasonal_cycle)
                    + instance.seasonal_phases[:, None]
                )
            )
            means = np.maximum(means, 1e-9)
            family_means = means.mean(axis=1)
            family_variances = means.mean(axis=1) + means.var(axis=1)
        else:
            family_means = np.asarray(instance.demand_lambdas, dtype=float)
            family_variances = np.asarray(instance.demand_lambdas, dtype=float)
        for component in range(instance.J):
            products = np.flatnonzero(instance.support[:, component])
            if products.size == 0:
                continue
            underage_cost = max(
                float(instance.backlog_costs[product])
                / max(float(instance.template_bom[product, component]), 1e-9)
                for product in products
            )
            probability = float(
                np.clip(
                    underage_cost
                    / (underage_cost + float(instance.holding_costs[component])),
                    1e-9,
                    1.0 - 1e-9,
                )
            )
            mean = 0.0
            variance = 0.0
            for product in products:
                protection = max(
                    0.0,
                    expected_lead_time
                    - float(instance.design_lead_times[product]),
                )
                coefficient = float(instance.template_bom[product, component])
                demand_mean = float(family_means[product])
                demand_variance = float(family_variances[product])
                mean += demand_mean * coefficient * protection
                variance += demand_variance * coefficient * coefficient * protection
            target[component] = max(
                0.0,
                float(norm.ppf(probability, loc=mean, scale=np.sqrt(max(variance, 0.0))))
                if variance > 1e-12
                else mean,
            )
        return target
