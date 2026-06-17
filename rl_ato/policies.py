from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from statistics import NormalDist
from typing import Dict, List, Tuple

import numpy as np

from .env import ATOEnv, ControlAction, Observation
from .scenario import ProblemInstance


class BasePolicy(ABC):
    name = "policy"

    @abstractmethod
    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        raise NotImplementedError


@dataclass
class SAAOBCAResult:
    base_stock: np.ndarray
    objective: float
    train_episodes: int
    step_sizes: Tuple[float, ...]
    beta_late: float
    known_requirement_scale: float
    starts_evaluated: int


def _priority_scores(env: ATOEnv, eta: float = 0.2) -> Dict[Tuple[int, int], float]:
    inst = env.instance
    scores: Dict[Tuple[int, int], float] = {}
    for i, s, _remaining, bom in env.observe().revealed:
        due = s + int(inst.design_lead_times[i]) + inst.delivery_window
        urgency = np.exp(eta * max(0, env.t - due))
        augmented = inst.backlog_costs[i] + float(np.dot(inst.holding_costs, bom > 1e-12))
        scores[(i, s)] = float(urgency * augmented)
    return scores


def _inventory_after_allocations(env: ATOEnv, allocations: List[Tuple[int, int, float]]) -> np.ndarray:
    assert env.scenario is not None
    inv = env.inventory.copy()
    for i, s, qty in allocations:
        inv -= env.scenario.realized_bom[int(i), int(s)] * float(qty)
    return np.maximum(inv, 0.0)


class NVDPolicy(BasePolicy):
    name = "NVD"

    def __init__(self, instance: ProblemInstance, known_requirement_scale: float = 0.5):
        self.instance = instance
        self.known_requirement_scale = float(known_requirement_scale)
        self.base_stock = self._compute_base_stock(instance)

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        allocations = env.greedy_allocate(_priority_scores(env))
        end_inventory = _inventory_after_allocations(env, allocations)
        remaining_after = env.remaining.copy()
        for i, s, qty in allocations:
            remaining_after[int(i), int(s)] = max(0.0, remaining_after[int(i), int(s)] - float(qty))
        known_requirement = remaining_after.sum(axis=1) @ self.instance.template_bom
        ip = end_inventory + obs.outstanding
        target = self.base_stock + self.known_requirement_scale * known_requirement
        orders = np.maximum(target - ip, 0.0)
        return ControlAction(allocations=allocations, orders=orders)

    @staticmethod
    def _normal_ppf(p: float, mu: float, sigma: float) -> float:
        if sigma <= 1e-9:
            return mu
        try:
            from scipy.stats import norm

            return float(norm.ppf(p, loc=mu, scale=sigma))
        except Exception:
            return float(mu + sigma * NormalDist().inv_cdf(p))

    def _compute_base_stock(self, inst: ProblemInstance) -> np.ndarray:
        s = np.zeros(inst.J, dtype=float)
        e_lead = inst.expected_lead_time
        for j in range(inst.J):
            users = [i for i in range(inst.I) if inst.support[i, j]]
            if not users:
                continue
            b_j = max(inst.backlog_costs[i] / max(inst.template_bom[i, j], 1e-9) for i in users)
            fractile = float(np.clip(b_j / (b_j + inst.holding_costs[j]), 1e-5, 1.0 - 1e-5))
            mu = 0.0
            var = 0.0
            for i in users:
                protection = max(0.0, e_lead - float(inst.design_lead_times[i]))
                gij = inst.template_bom[i, j]
                mu += inst.demand_lambdas[i] * gij * protection
                var += inst.demand_lambdas[i] * gij * gij * protection
            s[j] = max(0.0, self._normal_ppf(fractile, mu, np.sqrt(max(var, 0.0))))
        return s


class DTPPolicy(BasePolicy):
    name = "DTP"

    def __init__(self, instance: ProblemInstance, known_demand_scale: float = 1.0):
        self.instance = instance
        self.known_demand_scale = float(known_demand_scale)

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        allocations = env.greedy_allocate(_priority_scores(env))
        end_inventory = _inventory_after_allocations(env, allocations)
        remaining_after = env.remaining.copy()
        assert env.scenario is not None
        for i, s, qty in allocations:
            remaining_after[int(i), int(s)] = max(0.0, remaining_after[int(i), int(s)] - float(qty))

        target = np.zeros(self.instance.J, dtype=float)
        horizon = env.t + self.instance.max_replenishment_lead_time
        for i in range(self.instance.I):
            for s in range(env.t + 1):
                if not env.revealed[i, s] or remaining_after[i, s] <= 1e-9:
                    continue
                due = s + int(self.instance.design_lead_times[i]) + self.instance.delivery_window
                if due <= horizon:
                    target += env.scenario.realized_bom[i, s] * remaining_after[i, s]
        target *= self.known_demand_scale
        ip = end_inventory + obs.outstanding
        orders = np.maximum(target - ip, 0.0)
        return ControlAction(allocations=allocations, orders=orders)


class SAABSOBCAOptimizedPolicy(BasePolicy):
    name = "SAA-OBCA"

    def __init__(
        self,
        instance: ProblemInstance,
        base_stock: np.ndarray,
        beta_late: float = 1.0,
        allocation_solver: str = "gurobi",
        known_requirement_scale: float = 0.0,
    ):
        self.instance = instance
        self.base_stock = np.maximum(0.0, np.rint(np.asarray(base_stock, dtype=float)))
        self.beta_late = float(beta_late)
        self.allocation_solver = _normalize_obca_solver(allocation_solver)
        self.known_requirement_scale = float(known_requirement_scale)

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        allocations = solve_obca_allocation(
            env,
            obs,
            beta_late=self.beta_late,
            solver=self.allocation_solver,
        )
        end_inventory = _inventory_after_allocations(env, allocations)
        remaining_after = env.remaining.copy()
        for i, s, qty in allocations:
            remaining_after[int(i), int(s)] = max(0.0, remaining_after[int(i), int(s)] - float(qty))
        known_requirement = remaining_after.sum(axis=1) @ self.instance.template_bom
        ip = end_inventory + obs.outstanding
        orders = np.maximum(
            self.base_stock + self.known_requirement_scale * known_requirement - ip,
            0.0,
        )
        orders = np.maximum(0.0, np.ceil(orders - 1e-9))
        orders[np.abs(orders) < 1e-9] = 0.0
        return ControlAction(allocations=allocations, orders=orders)


def tune_saa_bs_obca(
    instance: ProblemInstance,
    train_scenarios,
    step_sizes: Tuple[float, ...] = (8.0, 4.0, 2.0, 1.0),
    beta_late: float = 1.0,
    allocation_solver: str = "gurobi",
    max_sweeps_per_delta: int = 3,
    known_requirement_scale: float = 0.0,
) -> SAAOBCAResult:
    allocation_solver = _normalize_obca_solver(allocation_solver)
    nvd = NVDPolicy(instance)
    s_nvd = np.maximum(0.0, np.rint(np.asarray(nvd.base_stock, dtype=float)))
    s_mean = np.maximum(0.0, np.rint(_mean_leadtime_base_stock(instance)))
    starts = [
        0.25 * s_nvd,
        0.50 * s_nvd,
        0.75 * s_nvd,
        s_nvd,
        1.25 * s_nvd,
        0.25 * s_mean,
        0.50 * s_mean,
        s_mean,
    ]
    s_ub = np.maximum(1.0, np.ceil(np.maximum.reduce(
        [
            2.5 * np.maximum(s_nvd, 1.0),
            2.5 * np.maximum(s_mean, 1.0),
            instance.initial_inventory + 2.0 * np.maximum(s_mean, 1.0),
            np.full(instance.J, 10.0),
        ]
    )))
    integer_step_sizes = tuple(
        float(max(1, int(round(delta))))
        for delta in step_sizes
        if float(delta) > 0.0
    )
    cache: Dict[Tuple[int, ...], float] = {}

    def objective(S: np.ndarray) -> float:
        S_int = np.maximum(0.0, np.rint(np.asarray(S, dtype=float)))
        key = tuple(int(x) for x in S_int)
        if key in cache:
            return cache[key]
        policy = SAABSOBCAOptimizedPolicy(
            instance,
            S_int,
            beta_late=beta_late,
            allocation_solver=allocation_solver,
            known_requirement_scale=known_requirement_scale,
        )
        costs = []
        for scenario in train_scenarios:
            env = ATOEnv(instance)
            obs = env.reset(scenario)
            done = False
            while not done:
                action = policy.act(env, obs)
                next_obs, _reward, done, _info = env.step(action)
                if next_obs is not None:
                    obs = next_obs
            costs.append(env.metrics()["cost"])
        val = float(np.mean(costs)) if costs else float("inf")
        cache[key] = val
        return val

    best_s = None
    best_val = float("inf")
    for start in starts:
        s, val = _coordinate_search(
            np.clip(np.asarray(start, dtype=float), 0.0, s_ub),
            s_ub=s_ub,
            step_sizes=integer_step_sizes,
            objective=objective,
            max_sweeps_per_delta=max_sweeps_per_delta,
        )
        if val < best_val:
            best_s, best_val = s, val
    assert best_s is not None
    return SAAOBCAResult(
        base_stock=np.maximum(0.0, np.rint(best_s)),
        objective=best_val,
        train_episodes=len(train_scenarios),
        step_sizes=integer_step_sizes,
        beta_late=float(beta_late),
        known_requirement_scale=float(known_requirement_scale),
        starts_evaluated=len(starts),
    )


def solve_obca_allocation(
    env: ATOEnv,
    obs: Observation,
    beta_late: float = 1.0,
    solver: str = "gurobi",
) -> List[Tuple[int, int, float]]:
    cohorts = [(i, s, rem, bom) for i, s, rem, bom in obs.revealed if rem > 1e-9]
    if not cohorts:
        return []
    weights = []
    for i, s, _rem, _bom in cohorts:
        overdue = env.t > s + int(env.instance.design_lead_times[i]) + env.instance.delivery_window
        weights.append(env.instance.backlog_costs[i] * (1.0 + beta_late * float(overdue)))
    return solve_weighted_allocation(cohorts, obs.inventory, weights, env.instance.J, solver)


def solve_weighted_allocation(
    cohorts: List[Tuple[int, int, float, np.ndarray]],
    inventory: np.ndarray,
    weights: List[float],
    n_components: int,
    solver: str = "gurobi",
) -> List[Tuple[int, int, float]]:
    if not cohorts:
        return []
    _normalize_obca_solver(solver)
    return _solve_obca_gurobi(cohorts, inventory, weights, n_components)


def _normalize_obca_solver(solver: str) -> str:
    solver_key = str(solver).strip().lower()
    if solver_key in {"gurobi", "gurobi-mip", "milp", "mip"}:
        return "gurobi"
    raise ValueError(f"unknown OBCA solver '{solver}'; use 'gurobi'")


def _solve_obca_gurobi(
    cohorts: List[Tuple[int, int, float, np.ndarray]],
    inventory: np.ndarray,
    weights: List[float],
    n_components: int,
) -> List[Tuple[int, int, float]]:
    os.environ["LC_ALL"] = "C"
    import gurobipy as gp
    from gurobipy import GRB

    n = len(cohorts)
    upper = np.zeros(n, dtype=float)
    for k, (_i, _s, rem, _bom) in enumerate(cohorts):
        upper[k] = np.floor(max(0.0, float(rem)))
    if np.all(upper <= 1e-9):
        return []
    model = gp.Model("obca")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.0
    x = model.addVars(n, lb=0.0, ub=upper.tolist(), vtype=GRB.INTEGER, name="x")
    model.setObjective(gp.quicksum(float(weights[k]) * x[k] for k in range(n)), GRB.MAXIMIZE)
    for j in range(n_components):
        expr_terms = []
        for k, (_i, _s, _rem, bom) in enumerate(cohorts):
            gij = float(bom[j])
            if gij > 1e-12:
                expr_terms.append(gij * x[k])
        if expr_terms:
            model.addConstr(gp.quicksum(expr_terms) <= float(inventory[j]), name=f"component_{j}")
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi OBCA failed with status {model.Status}")
    return [
        (int(cohorts[k][0]), int(cohorts[k][1]), float(max(0.0, x[k].X)))
        for k in range(n)
        if x[k].X > 1e-9
    ]


def _mean_leadtime_base_stock(instance: ProblemInstance) -> np.ndarray:
    return np.maximum(0.0, (instance.template_bom.T @ instance.demand_lambdas) * instance.expected_lead_time)


def _coordinate_search(
    start: np.ndarray,
    s_ub: np.ndarray,
    step_sizes: Tuple[float, ...],
    objective,
    max_sweeps_per_delta: int,
) -> Tuple[np.ndarray, float]:
    s_ub = np.maximum(0.0, np.rint(np.asarray(s_ub, dtype=float)))
    s = np.clip(np.rint(np.asarray(start, dtype=float)), 0.0, s_ub)
    best_cost = objective(s)
    for delta in step_sizes:
        step = float(max(1, int(round(delta))))
        for _sweep in range(max(1, int(max_sweeps_per_delta))):
            improved = False
            for j in range(len(s)):
                best_candidate = s
                best_candidate_cost = best_cost
                for sign in (+1.0, -1.0):
                    s_new = s.copy()
                    s_new[j] = np.clip(s_new[j] + sign * step, 0.0, s_ub[j])
                    s_new = np.rint(s_new)
                    if np.allclose(s_new, s):
                        continue
                    cost_new = objective(s_new)
                    if cost_new + 1e-8 < best_candidate_cost:
                        best_candidate = s_new
                        best_candidate_cost = cost_new
                if best_candidate_cost + 1e-8 < best_cost:
                    s = best_candidate
                    best_cost = best_candidate_cost
                    improved = True
            if not improved:
                break
    return s, best_cost
