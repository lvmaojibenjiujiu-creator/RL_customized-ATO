from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .scenario import ProblemInstance, Scenario


def throughput_normalized_inventory_ratio(
    sum_end_inventory: float,
    sum_replenishment: float,
    horizon: int,
) -> float:
    if float(sum_replenishment) <= 1e-9:
        return float("nan")
    return float(sum_end_inventory) / float(max(1, horizon)) / float(sum_replenishment)


@dataclass
class Observation:
    t: int
    comp_features: np.ndarray
    prod_features: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    history: np.ndarray
    inventory: np.ndarray
    outstanding: np.ndarray
    pipeline_by_age: np.ndarray
    urgency: np.ndarray
    revealed: List[Tuple[int, int, float, np.ndarray]]
    scale: float


@dataclass
class ControlAction:
    allocations: List[Tuple[int, int, float]]
    orders: np.ndarray


class ATOEnv:
    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        self.scenario: Scenario | None = None
        self.t = 0
        self.inventory = np.zeros(instance.J, dtype=float)
        self.remaining = np.zeros((instance.I, instance.T), dtype=float)
        self.revealed = np.zeros((instance.I, instance.T), dtype=bool)
        self.pipeline: List[Dict[str, float]] = []
        self.orders_history = np.zeros((instance.T, instance.J), dtype=float)
        self.arrivals_history = np.zeros((instance.T, instance.J), dtype=float)
        self.fulfilled = np.zeros((instance.I, instance.T, instance.T), dtype=float)
        self.total_cost = 0.0
        self.total_order_cost = 0.0
        self.total_holding_cost = 0.0
        self.total_backlog_cost = 0.0
        self.undiscounted_cost = 0.0
        self.undiscounted_order_cost = 0.0
        self.undiscounted_holding_cost = 0.0
        self.undiscounted_backlog_cost = 0.0
        self.holding_inventory_sum = 0.0
        self.total_ordered_components = 0.0

    def reset(self, scenario: Scenario) -> Observation:
        self.scenario = scenario
        inst = self.instance
        self.t = 0
        self.inventory = inst.initial_inventory.astype(float).copy()
        self.remaining = np.zeros((inst.I, inst.T), dtype=float)
        self.revealed = np.zeros((inst.I, inst.T), dtype=bool)
        self.pipeline = []
        self.orders_history = np.zeros((inst.T, inst.J), dtype=float)
        self.arrivals_history = np.zeros((inst.T, inst.J), dtype=float)
        self.fulfilled = np.zeros((inst.I, inst.T, inst.T), dtype=float)
        self.total_cost = 0.0
        self.total_order_cost = 0.0
        self.total_holding_cost = 0.0
        self.total_backlog_cost = 0.0
        self.undiscounted_cost = 0.0
        self.undiscounted_order_cost = 0.0
        self.undiscounted_holding_cost = 0.0
        self.undiscounted_backlog_cost = 0.0
        self.holding_inventory_sum = 0.0
        self.total_ordered_components = 0.0
        self._process_period_start()
        return self.observe()

    def step(self, action: ControlAction) -> tuple[Observation | None, float, bool, Dict[str, float]]:
        assert self.scenario is not None, "Call reset() before step()."
        inst = self.instance
        t = self.t
        action_orders = np.maximum(np.asarray(action.orders, dtype=float), 0.0)
        action_orders = np.nan_to_num(action_orders, nan=0.0, posinf=0.0, neginf=0.0)
        action_orders = np.ceil(action_orders - 1e-9)

        self._apply_allocations(action.allocations)
        end_inventory = self.inventory.copy()
        self.orders_history[t] = action_orders
        self.total_ordered_components += float(action_orders.sum())
        for j, qty in enumerate(action_orders):
            if qty <= 1e-9:
                continue
            lead = int(self.scenario.lead_times[t, j])
            self.pipeline.append({"j": j, "qty": float(qty), "placed": t, "arrive": t + lead})

        order_cost = float(np.dot(inst.ordering_costs, action_orders))
        holding_cost = float(np.dot(inst.holding_costs, end_inventory))
        backlog_cost = 0.0
        for i in range(inst.I):
            due_offset = int(inst.design_lead_times[i] + inst.delivery_window)
            for s in range(t + 1):
                if t >= s + due_offset:
                    backlog_cost += float(inst.backlog_costs[i] * self.remaining[i, s])
        cost = order_cost + holding_cost + backlog_cost
        discount = float(inst.discount_factor) ** t
        self.total_cost += discount * cost
        self.total_order_cost += discount * order_cost
        self.total_holding_cost += discount * holding_cost
        self.total_backlog_cost += discount * backlog_cost
        self.undiscounted_cost += cost
        self.undiscounted_order_cost += order_cost
        self.undiscounted_holding_cost += holding_cost
        self.undiscounted_backlog_cost += backlog_cost
        self.holding_inventory_sum += float(end_inventory.sum())

        self.t += 1
        done = self.t >= inst.T
        if not done:
            self._process_period_start()
            obs: Observation | None = self.observe()
        else:
            obs = None
        return obs, -cost, done, {
            "cost": cost,
            "discounted_cost": discount * cost,
            "order_cost": order_cost,
            "holding_cost": holding_cost,
            "backlog_cost": backlog_cost,
        }

    def metrics(self) -> Dict[str, float]:
        assert self.scenario is not None
        total_demand = float(self.scenario.demand.sum())
        unfilled = float(self.remaining.sum())
        ontime = 0.0
        for i in range(self.instance.I):
            for s in range(self.instance.T):
                latest = min(self.instance.T - 1, s + int(self.instance.design_lead_times[i]) + self.instance.delivery_window)
                earliest = min(self.instance.T, s + int(self.instance.design_lead_times[i]))
                if earliest <= latest:
                    ontime += float(self.fulfilled[i, s, earliest : latest + 1].sum())
        denom = max(total_demand, 1e-9)
        residual_inventory_ratio = throughput_normalized_inventory_ratio(
            self.holding_inventory_sum,
            self.total_ordered_components,
            self.instance.T,
        )
        return {
            "cost": float(self.total_cost),
            "order_cost": float(self.total_order_cost),
            "holding_cost": float(self.total_holding_cost),
            "backlog_cost": float(self.total_backlog_cost),
            "undiscounted_cost": float(self.undiscounted_cost),
            "fill_rate": float(1.0 - unfilled / denom),
            "ontime_rate": float(ontime / denom),
            "residual_inventory_ratio": float(residual_inventory_ratio),
            "mismatch_rate": float(residual_inventory_ratio),
            "total_demand": total_demand,
        }

    def observe(self) -> Observation:
        inst = self.instance
        scale = inst.feature_scale
        outstanding = self._outstanding()
        pipeline_by_age = self._pipeline_by_age()
        avail = self._coavailability(outstanding)
        comp_features = np.column_stack(
            [
                self.inventory / scale,
                outstanding / scale,
                inst.ordering_costs,
                inst.holding_costs,
                avail / scale,
            ]
        ).astype(np.float32)
        rb, ub, ob = self._family_backlog_features()
        prod_features = np.column_stack(
            [rb / scale, ub / scale, ob / scale, inst.backlog_costs]
        ).astype(np.float32)
        edge_index, edge_attr = self._edge_features(scale)
        history = self._history_features(scale).astype(np.float32)
        urgency = self._component_urgency()
        return Observation(
            t=self.t,
            comp_features=comp_features,
            prod_features=prod_features,
            edge_index=edge_index,
            edge_attr=edge_attr.astype(np.float32),
            history=history,
            inventory=self.inventory.copy(),
            outstanding=outstanding,
            pipeline_by_age=pipeline_by_age,
            urgency=urgency,
            revealed=self._revealed_cohorts(),
            scale=scale,
        )

    def greedy_allocate(self, scores: Dict[Tuple[int, int], float]) -> List[Tuple[int, int, float]]:
        cohorts = self._revealed_cohorts()
        ordered = sorted(
            cohorts,
            key=lambda item: (
                -float(scores.get((item[0], item[1]), 0.0)),
                int(item[1]),
                int(item[0]),
            ),
        )
        temp_inventory = self.inventory.copy()
        allocations: List[Tuple[int, int, float]] = []
        for i, s, remaining, bom in ordered:
            positive = bom > 1e-12
            if not positive.any():
                continue
            feasible = np.min(temp_inventory[positive] / bom[positive])
            qty = max(0.0, min(remaining, np.floor(feasible + 1e-9)))
            if qty <= 1e-9:
                continue
            temp_inventory -= bom * qty
            allocations.append((i, s, float(qty)))
        return allocations

    def _process_period_start(self) -> None:
        assert self.scenario is not None
        t = self.t
        for order in self.pipeline:
            if int(order["arrive"]) == t:
                j = int(order["j"])
                qty = float(order["qty"])
                self.inventory[j] += qty
                self.arrivals_history[t, j] += qty
        self.pipeline = [o for o in self.pipeline if int(o["arrive"]) > t]

        self.remaining[:, t] = self.scenario.demand[:, t]
        for i in range(self.instance.I):
            for s in range(t + 1):
                if t >= s + int(self.instance.design_lead_times[i]):
                    self.revealed[i, s] = True

    def _apply_allocations(self, allocations: Iterable[Tuple[int, int, float]]) -> None:
        assert self.scenario is not None
        for i, s, qty in allocations:
            i, s = int(i), int(s)
            if s > self.t or not self.revealed[i, s] or self.remaining[i, s] <= 1e-9:
                continue
            bom = self.scenario.realized_bom[i, s]
            positive = bom > 1e-12
            if not positive.any():
                continue
            feasible = np.min(self.inventory[positive] / bom[positive])
            requested = np.floor(max(0.0, float(qty)) + 1e-9)
            actual = max(
                0.0,
                min(
                    requested,
                    np.floor(self.remaining[i, s] + 1e-9),
                    np.floor(feasible + 1e-9),
                ),
            )
            if actual <= 1e-9:
                continue
            self.inventory -= bom * actual
            self.inventory = np.maximum(self.inventory, 0.0)
            self.remaining[i, s] -= actual
            self.fulfilled[i, s, self.t] += actual

    def _outstanding(self) -> np.ndarray:
        out = np.zeros(self.instance.J, dtype=float)
        for order in self.pipeline:
            out[int(order["j"])] += float(order["qty"])
        return out

    def _pipeline_by_age(self) -> np.ndarray:
        inst = self.instance
        out = np.zeros((inst.J, inst.max_replenishment_lead_time), dtype=float)
        for order in self.pipeline:
            age = self.t - int(order["placed"])
            if 1 <= age <= inst.max_replenishment_lead_time:
                out[int(order["j"]), age - 1] += float(order["qty"])
        return out

    def _coavailability(self, outstanding: np.ndarray) -> np.ndarray:
        inst = self.instance
        stock = self.inventory + outstanding
        avail = np.zeros(inst.J, dtype=float)
        for j in range(inst.J):
            total = 0.0
            for i in range(inst.I):
                if not inst.support[i, j]:
                    continue
                js = np.flatnonzero(inst.support[i])
                others = js[js != j]
                if len(others) == 0:
                    continue
                ratios = stock[others] / np.maximum(inst.template_bom[i, others], 1e-9)
                total += inst.template_bom[i, j] * float(np.min(ratios))
            avail[j] = total
        return avail

    def _family_backlog_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inst = self.instance
        rb = np.zeros(inst.I, dtype=float)
        ub = np.zeros(inst.I, dtype=float)
        ob = np.zeros(inst.I, dtype=float)
        for i in range(inst.I):
            for s in range(self.t + 1):
                rem = self.remaining[i, s]
                if rem <= 1e-9:
                    continue
                if self.revealed[i, s]:
                    rb[i] += rem
                    if self.t > s + int(inst.design_lead_times[i]) + inst.delivery_window:
                        ob[i] += rem
                else:
                    ub[i] += rem
        return rb, ub, ob

    def _edge_features(self, scale: float) -> tuple[np.ndarray, np.ndarray]:
        assert self.scenario is not None
        inst = self.instance
        edges = []
        attrs = []
        for i, j in inst.edge_pairs:
            denom = 0.0
            weighted = 0.0
            total_req = 0.0
            for s in range(self.t + 1):
                rem = self.remaining[i, s]
                if rem <= 1e-9 or not self.revealed[i, s]:
                    continue
                gij = self.scenario.realized_bom[i, s, j]
                denom += rem
                weighted += gij * rem
                total_req += gij * rem
            avg_revealed = weighted / denom if denom > 1e-9 else 0.0
            edges.append([int(i), int(inst.I + j)])
            attrs.append([inst.template_bom[i, j], avg_revealed, total_req / scale])
        if not edges:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
        return np.asarray(edges, dtype=np.int64).T, np.asarray(attrs, dtype=np.float32)

    def _history_features(self, scale: float) -> np.ndarray:
        inst = self.instance
        window = inst.history_window
        hist = np.zeros((inst.J, window, 2), dtype=float)
        for offset in range(window):
            period = self.t - window + offset
            if 0 <= period < inst.T:
                hist[:, offset, 0] = self.orders_history[period] / scale
                hist[:, offset, 1] = self.arrivals_history[period] / scale
        return hist

    def _component_urgency(self) -> np.ndarray:
        inst = self.instance
        urgency = np.zeros(inst.J, dtype=float)
        assert self.scenario is not None
        for i in range(inst.I):
            due_offset = int(inst.design_lead_times[i] + inst.delivery_window)
            for s in range(self.t + 1):
                if not self.revealed[i, s] or self.remaining[i, s] <= 1e-9:
                    continue
                if self.t > s + due_offset:
                    urgency += self.scenario.realized_bom[i, s] * self.remaining[i, s]
        return urgency

    def _revealed_cohorts(self) -> List[Tuple[int, int, float, np.ndarray]]:
        assert self.scenario is not None
        cohorts: List[Tuple[int, int, float, np.ndarray]] = []
        for i in range(self.instance.I):
            for s in range(self.t + 1):
                rem = float(self.remaining[i, s])
                if rem > 1e-9 and self.revealed[i, s]:
                    cohorts.append((i, s, rem, self.scenario.realized_bom[i, s].copy()))
        return cohorts
