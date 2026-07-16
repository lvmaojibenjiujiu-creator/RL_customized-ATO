from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from rl_ato.env import ATOEnv, ControlAction, Observation
from rl_ato.scenario import ProblemInstance, Scenario

from .base import BasePolicy, post_allocation_inventory
from .nvd import NVDPolicy


@dataclass(frozen=True)
class SAAOBCAResult:
    base_stock: np.ndarray
    objective: float
    calibration_episodes: int
    step_sizes: Tuple[int, ...]
    upper_bound: np.ndarray
    upper_quantile: float
    starts_evaluated: int


class SAAOBCAPolicy(BasePolicy):
    name = "SAA-OBCA"

    def __init__(
        self,
        instance: ProblemInstance,
        base_stock: np.ndarray,
        beta_late: float = 1.0,
    ):
        values = np.asarray(base_stock, dtype=float)
        if values.shape != (instance.J,):
            raise ValueError(f"base_stock must have shape ({instance.J},)")
        if np.any(values < -1e-9) or np.any(np.abs(values - np.rint(values)) > 1e-9):
            raise ValueError("base_stock must be a nonnegative integer vector")
        if beta_late < 0.0:
            raise ValueError("beta_late must be nonnegative")
        self.instance = instance
        self.base_stock = np.maximum(np.rint(values), 0.0)
        self.beta_late = float(beta_late)

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        allocations = solve_obca(env, obs, self.beta_late)
        end_inventory = post_allocation_inventory(env, allocations)
        inventory_position = end_inventory + np.asarray(obs.outstanding, dtype=float)
        orders = np.ceil(np.maximum(self.base_stock - inventory_position, 0.0) - 1e-9)
        return ControlAction(
            allocations=allocations,
            orders=np.maximum(orders, 0.0),
        )


def calibrate_saa_obca(
    instance: ProblemInstance,
    scenarios: Sequence[Scenario],
    step_sizes: Sequence[int] = (8, 4, 2, 1),
    beta_late: float = 1.0,
    upper_quantile: float = 0.99,
) -> SAAOBCAResult:
    if len(scenarios) == 0:
        raise ValueError("at least one calibration scenario is required")
    if beta_late < 0.0:
        raise ValueError("beta_late must be nonnegative")
    if not 0.0 < upper_quantile <= 1.0:
        raise ValueError("upper_quantile must be in (0, 1]")
    if len(step_sizes) == 0 or any(float(value) <= 0.0 for value in step_sizes):
        raise ValueError("at least one positive step size is required")
    deltas = tuple(sorted({max(1, int(round(value))) for value in step_sizes}, reverse=True))

    nvd_stock = np.rint(np.maximum(NVDPolicy(instance).base_stock, 0.0))
    mean_stock = np.rint(
        np.maximum(
            (instance.template_bom.T @ instance.demand_lambdas)
            * float(instance.expected_lead_time),
            0.0,
        )
    )
    upper_bound = _base_stock_upper_bound(instance, scenarios, upper_quantile)
    starts = [
        0.25 * nvd_stock,
        0.50 * nvd_stock,
        0.75 * nvd_stock,
        nvd_stock,
        1.25 * nvd_stock,
        0.25 * mean_stock,
        0.50 * mean_stock,
        mean_stock,
    ]
    objective_values: Dict[Tuple[int, ...], float] = {}

    def objective(candidate: np.ndarray) -> float:
        integer_candidate = np.clip(np.rint(candidate), 0.0, upper_bound)
        key = tuple(int(value) for value in integer_candidate)
        if key not in objective_values:
            policy = SAAOBCAPolicy(instance, integer_candidate, beta_late=beta_late)
            costs = [
                _calibration_policy_cost(
                    policy,
                    instance,
                    scenario,
                )
                for scenario in scenarios
            ]
            objective_values[key] = float(np.mean(costs))
        return objective_values[key]

    best_stock: np.ndarray | None = None
    best_objective = float("inf")
    for start in starts:
        stock, value = _coordinate_search(
            np.clip(np.rint(start), 0.0, upper_bound),
            upper_bound,
            deltas,
            objective,
        )
        if value < best_objective - 1e-9:
            best_stock = stock
            best_objective = value
    if best_stock is None:
        raise RuntimeError("calibration did not produce a base-stock vector")
    return SAAOBCAResult(
        base_stock=np.rint(best_stock).astype(int),
        objective=float(best_objective),
        calibration_episodes=len(scenarios),
        step_sizes=deltas,
        upper_bound=np.rint(upper_bound).astype(int),
        upper_quantile=float(upper_quantile),
        starts_evaluated=len(starts),
    )


def solve_obca(
    env: ATOEnv,
    obs: Observation,
    beta_late: float = 1.0,
) -> List[Tuple[int, int, float]]:
    cohorts = [
        (int(product), int(cohort_period), float(remaining), np.asarray(bom, dtype=float))
        for product, cohort_period, remaining, bom in obs.revealed
        if float(remaining) > 1e-9
    ]
    if not cohorts:
        return []
    os.environ["LC_ALL"] = "C"
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("saa_obca")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.0
    variables = {}
    for index, (_product, _cohort_period, remaining, _bom) in enumerate(cohorts):
        variables[index] = model.addVar(
            lb=0.0,
            ub=float(np.floor(remaining + 1e-9)),
            vtype=GRB.INTEGER,
            name=f"x_{index}",
        )
    weights = []
    for product, cohort_period, _remaining, _bom in cohorts:
        due = (
            cohort_period
            + int(env.instance.design_lead_times[product])
            + int(env.instance.delivery_window)
        )
        overdue = 1.0 if obs.t > due else 0.0
        weights.append(
            float(env.instance.backlog_costs[product])
            * (1.0 + float(beta_late) * overdue)
        )
    model.setObjective(
        gp.quicksum(weights[index] * variables[index] for index in variables),
        GRB.MAXIMIZE,
    )
    for component in range(env.instance.J):
        terms = [
            float(cohorts[index][3][component]) * variables[index]
            for index in variables
            if float(cohorts[index][3][component]) > 1e-12
        ]
        if terms:
            model.addConstr(
                gp.quicksum(terms) <= float(obs.inventory[component]),
                name=f"component_{component}",
            )
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status {model.Status}")
    return [
        (
            cohorts[index][0],
            cohorts[index][1],
            float(round(variables[index].X)),
        )
        for index in variables
        if variables[index].X > 1e-9
    ]


def _calibration_policy_cost(
    policy: SAAOBCAPolicy,
    instance: ProblemInstance,
    scenario: Scenario,
) -> float:
    env = ATOEnv(instance)
    obs = env.reset(scenario)
    done = False
    while not done:
        action = policy.act(env, obs)
        next_obs, _reward, done, _information = env.step(action)
        if next_obs is not None:
            obs = next_obs
    return float(env.undiscounted_cost)


def _base_stock_upper_bound(
    instance: ProblemInstance,
    scenarios: Sequence[Scenario],
    upper_quantile: float,
) -> np.ndarray:
    window = max(1, int(instance.max_replenishment_lead_time))
    path_requirements = []
    for scenario in scenarios:
        per_period = np.einsum(
            "it,itj->tj",
            np.asarray(scenario.demand, dtype=float),
            np.asarray(scenario.realized_bom, dtype=float),
        )
        cumulative = np.vstack(
            [
                np.zeros((1, instance.J), dtype=float),
                np.cumsum(per_period, axis=0),
            ]
        )
        rolling = np.zeros((instance.T, instance.J), dtype=float)
        for start in range(instance.T):
            end = min(instance.T, start + window)
            rolling[start] = cumulative[end] - cumulative[start]
        path_requirements.append(rolling.max(axis=0))
    bound = np.quantile(
        np.asarray(path_requirements, dtype=float),
        float(upper_quantile),
        axis=0,
    )
    return np.maximum(1.0, np.ceil(bound))


def _coordinate_search(
    start: np.ndarray,
    upper_bound: np.ndarray,
    step_sizes: Tuple[int, ...],
    objective,
) -> Tuple[np.ndarray, float]:
    stock = np.clip(np.rint(start), 0.0, upper_bound)
    value = float(objective(stock))
    for step in step_sizes:
        while True:
            improved = False
            for component in range(stock.size):
                selected = stock
                selected_value = value
                for direction in (1, -1):
                    candidate = stock.copy()
                    candidate[component] = np.clip(
                        candidate[component] + direction * step,
                        0.0,
                        upper_bound[component],
                    )
                    candidate = np.rint(candidate)
                    if np.array_equal(candidate, stock):
                        continue
                    candidate_value = float(objective(candidate))
                    if candidate_value < selected_value - 1e-9:
                        selected = candidate
                        selected_value = candidate_value
                if selected_value < value - 1e-9:
                    stock = selected
                    value = selected_value
                    improved = True
            if not improved:
                break
    return stock, value
