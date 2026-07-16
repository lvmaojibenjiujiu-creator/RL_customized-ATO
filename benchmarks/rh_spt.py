from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from rl_ato.env import ATOEnv, ControlAction, Observation
from rl_ato.scenario import (
    ProblemInstance,
    _period_demand_means,
    _sample_demand_path,
    _sample_order_specific_bom,
)

from .base import BasePolicy


@dataclass(frozen=True)
class ConditionalScenario:
    demand: np.ndarray
    bom: np.ndarray
    lead_times: np.ndarray
    existing_arrivals: np.ndarray


class RHSPTPolicy(BasePolicy):
    name = "RH-SPT"

    def __init__(
        self,
        instance: ProblemInstance,
        horizon: int,
        n_scenarios: int,
        discount_factor: float,
        terminal_backlog_weight: float,
        terminal_inventory_weight: float,
        time_limit: float,
        mip_gap: float,
        threads: int,
        seed: int,
    ):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive")
        if not 0.0 < discount_factor <= 1.0:
            raise ValueError("discount_factor must be in (0, 1]")
        if terminal_backlog_weight < 0.0 or terminal_inventory_weight < 0.0:
            raise ValueError("terminal weights must be nonnegative")
        if time_limit < 0.0:
            raise ValueError("time_limit must be nonnegative")
        if mip_gap < 0.0:
            raise ValueError("mip_gap must be nonnegative")
        if threads < 0:
            raise ValueError("threads must be nonnegative")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        self.instance = instance
        self.horizon = int(horizon)
        self.n_scenarios = int(n_scenarios)
        self.discount_factor = float(discount_factor)
        self.terminal_backlog_weight = float(terminal_backlog_weight)
        self.terminal_inventory_weight = float(terminal_inventory_weight)
        self.time_limit = float(time_limit)
        self.mip_gap = float(mip_gap)
        self.threads = int(threads)
        self.seed = int(seed)
        self.last_solver_status = ""
        self.last_solver_gap = np.nan
        self.last_runtime_seconds = np.nan

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        if env.instance is not self.instance:
            raise ValueError("policy and environment must use the same instance")
        if env.scenario is None:
            raise RuntimeError("environment is not initialized")
        started = time.perf_counter()
        scenarios = _sample_conditional_scenarios(
            self.instance,
            env,
            obs,
            self.horizon,
            self.n_scenarios,
            self.seed,
        )
        allocations, orders, status, gap = self._solve(env, obs, scenarios)
        self.last_solver_status = status
        self.last_solver_gap = gap
        self.last_runtime_seconds = float(time.perf_counter() - started)
        return ControlAction(allocations=allocations, orders=orders)

    def _solve(
        self,
        env: ATOEnv,
        obs: Observation,
        scenarios: List[ConditionalScenario],
    ) -> Tuple[List[Tuple[int, int, float]], np.ndarray, str, float]:
        os.environ["LC_ALL"] = "C"
        import gurobipy as gp
        from gurobipy import GRB

        inst = self.instance
        t0 = int(obs.t)
        length = min(self.horizon, inst.T - t0)
        if length <= 0:
            raise RuntimeError("rolling horizon is empty")
        current_revealed = {
            (int(product), int(cohort_period)): float(remaining)
            for product, cohort_period, remaining, _bom in obs.revealed
            if float(remaining) > 1e-9
        }
        active = {
            (product, cohort_period): float(env.remaining[product, cohort_period])
            for product in range(inst.I)
            for cohort_period in range(t0 + 1)
            if float(env.remaining[product, cohort_period]) > 1e-9
        }

        model = gp.Model("rh_spt")
        model.Params.OutputFlag = 0
        model.Params.MIPGap = self.mip_gap
        if self.time_limit > 0.0:
            model.Params.TimeLimit = self.time_limit
        if self.threads > 0:
            model.Params.Threads = self.threads

        current_orders = {
            component: model.addVar(
                lb=0.0,
                vtype=GRB.INTEGER,
                name=f"q_0_{component}",
            )
            for component in range(inst.J)
        }
        current_backlogs = {
            cohort: model.addVar(
                lb=0.0,
                ub=float(remaining),
                vtype=GRB.INTEGER,
                name=f"B_0_{cohort[0]}_{cohort[1]}",
            )
            for cohort, remaining in current_revealed.items()
        }
        orders: Dict[Tuple[int, int, int], Any] = {}
        backlogs: Dict[Tuple[int, int, int, int], Any] = {}
        inventories: Dict[Tuple[int, int, int], Any] = {}
        cohorts_by_scenario: Dict[int, List[Tuple[int, int]]] = {}

        for scenario_index, scenario in enumerate(scenarios):
            cohorts = list(active)
            for cohort_period in range(t0 + 1, t0 + length):
                for product in range(inst.I):
                    if float(scenario.demand[product, cohort_period]) > 1e-9:
                        cohorts.append((product, cohort_period))
            cohorts_by_scenario[scenario_index] = cohorts

            for relative_period in range(length):
                for component in range(inst.J):
                    if relative_period == 0:
                        orders[(scenario_index, component, relative_period)] = current_orders[component]
                    else:
                        orders[(scenario_index, component, relative_period)] = model.addVar(
                            lb=0.0,
                            vtype=GRB.CONTINUOUS,
                            name=f"q_{scenario_index}_{relative_period}_{component}",
                        )
                    inventories[(scenario_index, component, relative_period)] = model.addVar(
                        lb=0.0,
                        vtype=GRB.CONTINUOUS,
                        name=f"Y_{scenario_index}_{relative_period}_{component}",
                    )

            for product, cohort_period in cohorts:
                first_period = 0 if cohort_period <= t0 else cohort_period - t0
                for relative_period in range(first_period, length):
                    key = (scenario_index, product, cohort_period, relative_period)
                    if relative_period == 0 and (product, cohort_period) in current_backlogs:
                        backlogs[key] = current_backlogs[(product, cohort_period)]
                    else:
                        backlogs[key] = model.addVar(
                            lb=0.0,
                            vtype=GRB.CONTINUOUS,
                            name=f"B_{scenario_index}_{product}_{cohort_period}_{relative_period}",
                        )
                    previous = (
                        float(active[(product, cohort_period)])
                        if relative_period == first_period and cohort_period <= t0
                        else 0.0
                        if relative_period == first_period
                        else backlogs[
                            (
                                scenario_index,
                                product,
                                cohort_period,
                                relative_period - 1,
                            )
                        ]
                    )
                    added = (
                        float(scenario.demand[product, cohort_period])
                        if cohort_period > t0 and relative_period == first_period
                        else 0.0
                    )
                    model.addConstr(
                        backlogs[key] <= previous + added,
                        name=f"backlog_bound_{scenario_index}_{product}_{cohort_period}_{relative_period}",
                    )
                    calendar_period = t0 + relative_period
                    if calendar_period < cohort_period + int(inst.design_lead_times[product]):
                        model.addConstr(
                            backlogs[key] == previous + added,
                            name=f"release_{scenario_index}_{product}_{cohort_period}_{relative_period}",
                        )

            for relative_period in range(length):
                for component in range(inst.J):
                    previous_inventory = (
                        float(obs.inventory[component])
                        if relative_period == 0
                        else inventories[(scenario_index, component, relative_period - 1)]
                    )
                    received_orders = [
                        orders[(scenario_index, component, order_period)]
                        for order_period in range(relative_period)
                        if order_period
                        + int(scenario.lead_times[order_period, component])
                        == relative_period
                    ]
                    consumption = []
                    for product, cohort_period in cohorts:
                        first_period = 0 if cohort_period <= t0 else cohort_period - t0
                        if relative_period < first_period:
                            continue
                        previous_backlog = (
                            float(active[(product, cohort_period)])
                            if relative_period == first_period and cohort_period <= t0
                            else 0.0
                            if relative_period == first_period
                            else backlogs[
                                (
                                    scenario_index,
                                    product,
                                    cohort_period,
                                    relative_period - 1,
                                )
                            ]
                        )
                        added = (
                            float(scenario.demand[product, cohort_period])
                            if cohort_period > t0 and relative_period == first_period
                            else 0.0
                        )
                        assembled = (
                            previous_backlog
                            + added
                            - backlogs[
                                (
                                    scenario_index,
                                    product,
                                    cohort_period,
                                    relative_period,
                                )
                            ]
                        )
                        coefficient = float(scenario.bom[product, cohort_period, component])
                        if coefficient > 1e-12:
                            consumption.append(coefficient * assembled)
                    model.addConstr(
                        inventories[(scenario_index, component, relative_period)]
                        == previous_inventory
                        + float(scenario.existing_arrivals[relative_period, component])
                        + gp.quicksum(received_orders)
                        - gp.quicksum(consumption),
                        name=f"inventory_{scenario_index}_{relative_period}_{component}",
                    )

        objective_terms = []
        for scenario_index, scenario in enumerate(scenarios):
            cohorts = cohorts_by_scenario[scenario_index]
            scenario_terms = []
            for relative_period in range(length):
                discount = self.discount_factor ** relative_period
                stage_terms = [
                    gp.quicksum(
                        float(inst.ordering_costs[component])
                        * orders[(scenario_index, component, relative_period)]
                        for component in range(inst.J)
                    ),
                    gp.quicksum(
                        float(inst.holding_costs[component])
                        * inventories[(scenario_index, component, relative_period)]
                        for component in range(inst.J)
                    ),
                ]
                backlog_terms = []
                calendar_period = t0 + relative_period
                for product, cohort_period in cohorts:
                    first_period = 0 if cohort_period <= t0 else cohort_period - t0
                    due = (
                        cohort_period
                        + int(inst.design_lead_times[product])
                        + int(inst.delivery_window)
                    )
                    if relative_period >= first_period and calendar_period > due:
                        backlog_terms.append(
                            float(inst.backlog_costs[product])
                            * backlogs[
                                (
                                    scenario_index,
                                    product,
                                    cohort_period,
                                    relative_period,
                                )
                            ]
                        )
                stage_terms.append(gp.quicksum(backlog_terms))
                scenario_terms.append(discount * gp.quicksum(stage_terms))

            terminal_period = length - 1
            terminal_backlogs = [
                float(inst.backlog_costs[product])
                * backlogs[(scenario_index, product, cohort_period, terminal_period)]
                for product, cohort_period in cohorts
            ]
            terminal_inventories = [
                float(inst.holding_costs[component])
                * inventories[(scenario_index, component, terminal_period)]
                for component in range(inst.J)
            ]
            scenario_terms.append(
                self.terminal_backlog_weight * gp.quicksum(terminal_backlogs)
                + self.terminal_inventory_weight * gp.quicksum(terminal_inventories)
            )
            objective_terms.append(gp.quicksum(scenario_terms))

        model.setObjective(
            gp.quicksum(objective_terms) / float(self.n_scenarios),
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.Status != GRB.OPTIMAL or model.SolCount <= 0:
            raise RuntimeError(
                f"RH-SPT optimization failed with status {model.Status} and {model.SolCount} solutions"
            )

        action_orders = np.asarray(
            [
                float(max(0, int(round(current_orders[component].X))))
                for component in range(inst.J)
            ],
            dtype=float,
        )
        allocations = []
        for (product, cohort_period), remaining in current_revealed.items():
            target = float(max(0, int(round(current_backlogs[(product, cohort_period)].X))))
            quantity = max(0.0, float(remaining) - target)
            if quantity > 1e-9:
                allocations.append((product, cohort_period, quantity))
        status = _solver_status(model.Status)
        gap = float(model.MIPGap) if model.IsMIP else 0.0
        return allocations, action_orders, status, gap


def _sample_conditional_scenarios(
    instance: ProblemInstance,
    env: ATOEnv,
    obs: Observation,
    horizon: int,
    n_scenarios: int,
    seed: int,
) -> List[ConditionalScenario]:
    if env.scenario is None:
        raise RuntimeError("environment is not initialized")
    t0 = int(obs.t)
    length = min(int(horizon), instance.T - t0)
    episode = int(getattr(env.scenario, "episode", 0))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), episode, t0]))
    means = _period_demand_means(instance)
    scenarios = []
    for _ in range(int(n_scenarios)):
        sampled_demand = _sample_demand_path(instance, means, rng).astype(float)
        sampled_bom, _ = _sample_order_specific_bom(instance, rng)
        demand = np.zeros((instance.I, instance.T), dtype=float)
        bom = np.zeros((instance.I, instance.T, instance.J), dtype=float)
        for product in range(instance.I):
            for cohort_period in range(t0 + 1):
                if env.revealed[product, cohort_period]:
                    bom[product, cohort_period] = env.scenario.realized_bom[
                        product,
                        cohort_period,
                    ]
                else:
                    bom[product, cohort_period] = sampled_bom[product, cohort_period]
            for cohort_period in range(t0 + 1, t0 + length):
                demand[product, cohort_period] = sampled_demand[product, cohort_period]
                bom[product, cohort_period] = sampled_bom[product, cohort_period]
        lead_times = rng.integers(
            low=instance.min_replenishment_lead_time,
            high=instance.max_replenishment_lead_time + 1,
            size=(length, instance.J),
        ).astype(int)
        existing_arrivals = _sample_existing_arrivals(
            instance,
            obs,
            length,
            rng,
        )
        scenarios.append(
            ConditionalScenario(
                demand=demand,
                bom=bom,
                lead_times=lead_times,
                existing_arrivals=existing_arrivals,
            )
        )
    return scenarios


def _sample_existing_arrivals(
    instance: ProblemInstance,
    obs: Observation,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    arrivals = np.zeros((horizon, instance.J), dtype=float)
    for component in range(instance.J):
        for age_index in range(obs.pipeline_by_age.shape[1]):
            quantity = float(obs.pipeline_by_age[component, age_index])
            if quantity <= 1e-9:
                continue
            age = age_index + 1
            minimum_lead = max(instance.min_replenishment_lead_time, age + 1)
            if minimum_lead > instance.max_replenishment_lead_time:
                raise RuntimeError("observed pipeline age is inconsistent with lead-time support")
            lead = int(
                rng.integers(
                    minimum_lead,
                    instance.max_replenishment_lead_time + 1,
                )
            )
            relative_arrival = lead - age
            if relative_arrival < horizon:
                arrivals[relative_arrival, component] += quantity
    return arrivals


def _solver_status(status: int) -> str:
    names = {
        2: "OPTIMAL",
    }
    return names.get(int(status), f"STATUS_{status}")
