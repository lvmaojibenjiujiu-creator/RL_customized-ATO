from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .scenario import ProblemInstance, Scenario


@dataclass(frozen=True)
class PICostBreakdown:
    total: float
    ordering: float
    holding: float
    backlog: float
    raw_ordering: float
    initial_kept: float
    ordered_components: float
    fulfilled_units: float


def perfect_information_cost(
    instance: ProblemInstance,
    scenario: Scenario,
    ordering_cost_weight: float = 1.0,
) -> float:
    return perfect_information_breakdown(
        instance,
        scenario,
        ordering_cost_weight=ordering_cost_weight,
    ).total


def perfect_information_breakdown(
    instance: ProblemInstance,
    scenario: Scenario,
    ordering_cost_weight: float = 1.0,
) -> PICostBreakdown:
    os.environ["LC_ALL"] = "C"
    import gurobipy as gp
    from gurobipy import GRB

    inst = instance
    I, J, T = inst.I, inst.J, inst.T
    order_weight = max(0.0, float(ordering_cost_weight))

    x_vars: List[Tuple[int, int, int]] = []
    x_index: Dict[Tuple[int, int, int], int] = {}
    for i in range(I):
        ld = int(inst.design_lead_times[i])
        for s in range(T):
            if scenario.demand[i, s] <= 1e-9:
                continue
            for t in range(s + ld, T):
                x_index[(i, s, t)] = len(x_vars)
                x_vars.append((i, s, t))

    model = gp.Model("perfect_information_oracle")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.0
    q = model.addVars(J, T, lb=0.0, vtype=GRB.INTEGER, name="q")
    y0 = model.addVars(
        J,
        lb=0.0,
        ub=[float(v) for v in inst.initial_inventory],
        vtype=GRB.INTEGER,
        name="initial_kept",
    )
    inv = model.addVars(J, T, lb=0.0, vtype=GRB.INTEGER, name="end_inventory")
    x_ub = [float(np.floor(max(0.0, scenario.demand[i, s]))) for i, s, _t in x_vars]
    x = model.addVars(len(x_vars), lb=0.0, ub=x_ub, vtype=GRB.INTEGER, name="x")

    for i in range(I):
        ld = int(inst.design_lead_times[i])
        for s in range(T):
            if scenario.demand[i, s] <= 1e-9:
                continue
            keys = [x_index[(i, s, t)] for t in range(s + ld, T) if (i, s, t) in x_index]
            if keys:
                model.addConstr(
                    gp.quicksum(x[k] for k in keys) <= float(scenario.demand[i, s]),
                    name=f"demand_{i}_{s}",
                )

    consumption: List[List[List[object]]] = [[[] for _ in range(T)] for _ in range(J)]
    for k, (i, s, t) in enumerate(x_vars):
        bom = scenario.realized_bom[i, s]
        for j in np.flatnonzero(bom > 1e-12):
            consumption[int(j)][t].append(float(bom[j]) * x[k])

    arrivals: List[List[List[object]]] = [[[] for _ in range(T)] for _ in range(J)]
    for j in range(J):
        for u in range(T):
            arrival = u + int(scenario.lead_times[u, j])
            if arrival < T:
                arrivals[j][arrival].append(q[j, u])

    for j in range(J):
        for t in range(T):
            start_inv = y0[j] if t == 0 else inv[j, t - 1]
            model.addConstr(
                inv[j, t]
                == start_inv
                + gp.quicksum(arrivals[j][t])
                - gp.quicksum(consumption[j][t]),
                name=f"balance_{j}_{t}",
            )

    const_backlog = 0.0
    backlog_savings = []
    for i in range(I):
        due_offset = int(inst.design_lead_times[i] + inst.delivery_window)
        for s in range(T):
            const_backlog += float(
                inst.backlog_costs[i]
                * scenario.demand[i, s]
                * max(0, T - (s + due_offset + 1))
            )
    for k, (i, s, t) in enumerate(x_vars):
        due = s + int(inst.design_lead_times[i]) + inst.delivery_window
        saved_periods = max(0, T - max(t, due + 1))
        if saved_periods > 0:
            backlog_savings.append(float(inst.backlog_costs[i] * saved_periods) * x[k])

    objective = (
        order_weight
        * gp.quicksum(float(inst.ordering_costs[j]) * q[j, u] for j in range(J) for u in range(T))
        + gp.quicksum(float(inst.holding_costs[j]) * inv[j, t] for j in range(J) for t in range(T))
        + const_backlog
        - gp.quicksum(backlog_savings)
    )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi PI oracle failed with status {model.Status}")

    raw_ordering = sum(
        float(inst.ordering_costs[j]) * float(q[j, u].X)
        for j in range(J)
        for u in range(T)
    )
    ordering = order_weight * raw_ordering
    holding = sum(
        float(inst.holding_costs[j]) * float(inv[j, t].X)
        for j in range(J)
        for t in range(T)
    )
    saved_backlog = 0.0
    fulfilled = 0.0
    for k, (i, s, t) in enumerate(x_vars):
        qty = float(x[k].X)
        if qty <= 1e-9:
            continue
        fulfilled += qty
        due = s + int(inst.design_lead_times[i]) + inst.delivery_window
        saved_backlog += float(inst.backlog_costs[i] * max(0, T - max(t, due + 1)) * qty)
    backlog = max(0.0, const_backlog - saved_backlog)
    total = ordering + holding + backlog
    return PICostBreakdown(
        total=float(total),
        ordering=float(ordering),
        holding=float(holding),
        backlog=float(backlog),
        raw_ordering=float(raw_ordering),
        initial_kept=float(sum(y0[j].X for j in range(J))),
        ordered_components=float(sum(q[j, u].X for j in range(J) for u in range(T))),
        fulfilled_units=float(fulfilled),
    )
