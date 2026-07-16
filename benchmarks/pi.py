from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from rl_ato.scenario import ProblemInstance, Scenario


@dataclass(frozen=True)
class PICostBreakdown:
    total: float
    ordering: float
    holding: float
    backlog: float
    initial_inventory: float
    ordered_components: float
    fulfilled_units: float


def _integer_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    rounded = np.rint(array)
    if np.any(array < -1e-9) or not np.allclose(array, rounded, atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must contain nonnegative integers")
    return rounded.astype(np.int64)


def perfect_information_cost(instance: ProblemInstance, scenario: Scenario) -> float:
    return perfect_information_breakdown(instance, scenario).total


def perfect_information_breakdown(
    instance: ProblemInstance,
    scenario: Scenario,
) -> PICostBreakdown:
    os.environ["LC_ALL"] = "C"
    import gurobipy as gp
    from gurobipy import GRB

    I, J, T = int(instance.I), int(instance.J), int(instance.T)
    gamma = float(instance.discount_factor)
    if not 0.0 < gamma <= 1.0:
        raise ValueError("instance.discount_factor must be in (0, 1]")

    demand = _integer_array(scenario.demand, "scenario.demand")
    bom = _integer_array(scenario.realized_bom, "scenario.realized_bom")
    lead_times = _integer_array(scenario.lead_times, "scenario.lead_times")
    initial_inventory = _integer_array(instance.initial_inventory, "instance.initial_inventory")
    if demand.shape != (I, T):
        raise ValueError("scenario.demand has an invalid shape")
    if bom.shape != (I, T, J):
        raise ValueError("scenario.realized_bom has an invalid shape")
    if lead_times.shape != (T, J):
        raise ValueError("scenario.lead_times has an invalid shape")
    if np.any(lead_times < 1):
        raise ValueError("scenario.lead_times must be positive")
    if initial_inventory.shape != (J,):
        raise ValueError("instance.initial_inventory has an invalid shape")

    x_keys = [(i, s, t) for i in range(I) for s in range(T) for t in range(s, T)]
    r_keys = [(i, s, t) for i in range(I) for s in range(T) for t in range(s, T + 1)]

    model = gp.Model("pi")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.0
    q = model.addVars(J, T, lb=0.0, vtype=GRB.INTEGER, name="q")
    x = model.addVars(x_keys, lb=0.0, vtype=GRB.INTEGER, name="x")
    r = model.addVars(r_keys, lb=0.0, vtype=GRB.INTEGER, name="r")
    y = model.addVars(J, T, lb=0.0, vtype=GRB.INTEGER, name="y")

    for i in range(I):
        release = int(instance.design_lead_times[i])
        for s in range(T):
            model.addConstr(r[i, s, s] == int(demand[i, s]))
            for t in range(s, T):
                model.addConstr(r[i, s, t + 1] == r[i, s, t] - x[i, s, t])
                model.addConstr(x[i, s, t] <= r[i, s, t])
                if t < s + release:
                    model.addConstr(x[i, s, t] == 0)

    for j in range(J):
        for t in range(T):
            previous_inventory = int(initial_inventory[j]) if t == 0 else y[j, t - 1]
            arrivals = gp.quicksum(
                q[j, tau]
                for tau in range(t)
                if tau + int(lead_times[tau, j]) == t
            )
            consumption = gp.quicksum(
                int(bom[i, s, j]) * x[i, s, t]
                for i in range(I)
                for s in range(t + 1)
                if int(bom[i, s, j]) > 0
            )
            model.addConstr(y[j, t] == previous_inventory + arrivals - consumption)

    ordering_terms: Dict[Tuple[int, int], object] = {}
    holding_terms: Dict[Tuple[int, int], object] = {}
    backlog_terms: Dict[Tuple[int, int, int], object] = {}
    for t in range(T):
        discount = gamma**t
        for j in range(J):
            ordering_terms[j, t] = discount * float(instance.ordering_costs[j]) * q[j, t]
            holding_terms[j, t] = discount * float(instance.holding_costs[j]) * y[j, t]
        for i in range(I):
            due_offset = int(instance.design_lead_times[i]) + int(instance.delivery_window)
            for s in range(t + 1):
                if t >= s + due_offset:
                    backlog_terms[i, s, t] = discount * float(instance.backlog_costs[i]) * r[i, s, t + 1]

    model.setObjective(
        gp.quicksum(ordering_terms.values())
        + gp.quicksum(holding_terms.values())
        + gp.quicksum(backlog_terms.values()),
        GRB.MINIMIZE,
    )
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"PI solve failed with status {model.Status}")

    ordering = float(sum(term.getValue() for term in ordering_terms.values()))
    holding = float(sum(term.getValue() for term in holding_terms.values()))
    backlog = float(sum(term.getValue() for term in backlog_terms.values()))
    total = ordering + holding + backlog
    return PICostBreakdown(
        total=total,
        ordering=ordering,
        holding=holding,
        backlog=backlog,
        initial_inventory=float(initial_inventory.sum()),
        ordered_components=float(sum(q[j, t].X for j in range(J) for t in range(T))),
        fulfilled_units=float(sum(x[key].X for key in x_keys)),
    )
