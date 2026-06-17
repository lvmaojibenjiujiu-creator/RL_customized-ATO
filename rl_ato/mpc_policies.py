from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

from .env import ATOEnv, ControlAction, Observation
from .policies import BasePolicy, NVDPolicy
from .scenario import (
    ProblemInstance,
    _period_demand_means,
    _sample_demand_path,
    _sample_order_specific_bom,
)


@dataclass
class PlanningScenario:
    demand: np.ndarray
    bom: np.ndarray
    lead_times: np.ndarray
    existing_arrivals: np.ndarray


def canonical_policy_name(name: str) -> str:
    return name.strip().upper().replace("_", "-")


def add_mpc_arguments(parser: Any) -> None:
    parser.add_argument("--mpc-horizon", type=int, default=8)
    parser.add_argument("--mpc-scenarios", type=int, default=8)
    parser.add_argument("--mpc-time-limit", type=float, default=30.0)
    parser.add_argument("--mpc-mip-gap", type=float, default=0.02)
    parser.add_argument("--mpc-threads", type=int, default=0)
    parser.add_argument("--mpc-terminal-backlog-mult", type=float, default=2.0)
    parser.add_argument("--mpc-terminal-inventory-mult", type=float, default=0.0)
    parser.add_argument("--continuous-mpc", action="store_true")
    parser.add_argument("--mpc-seed", type=int, default=None)


def mpc_policies_from_request(
    requested: Iterable[str],
    args: Any,
    seed: int,
) -> List[BasePolicy]:
    tokens = {canonical_policy_name(name) for name in requested}
    threads = int(args.mpc_threads) if int(args.mpc_threads) > 0 else None
    integer = not bool(args.continuous_mpc)
    base_seed = int(seed if args.mpc_seed is None else args.mpc_seed)
    policies: List[BasePolicy] = []
    if "CE-MPC" in tokens or "CEMPC" in tokens:
        policies.append(
            CertaintyEquivalentMPCPolicy(
                horizon=args.mpc_horizon,
                time_limit=args.mpc_time_limit,
                mip_gap=args.mpc_mip_gap,
                threads=threads,
                integer=integer,
                terminal_backlog_mult=args.mpc_terminal_backlog_mult,
                terminal_inventory_mult=args.mpc_terminal_inventory_mult,
                seed=base_seed + 54321,
            )
        )
    if {"RH-SAA-MPC", "RHSAA-MPC", "RHSAAMPC"}.intersection(tokens):
        policies.append(
            RollingHorizonSAAMPCPolicy(
                horizon=args.mpc_horizon,
                scenarios=args.mpc_scenarios,
                time_limit=args.mpc_time_limit,
                mip_gap=args.mpc_mip_gap,
                threads=threads,
                integer=integer,
                terminal_backlog_mult=args.mpc_terminal_backlog_mult,
                terminal_inventory_mult=args.mpc_terminal_inventory_mult,
                seed=base_seed + 777,
            )
        )
    return policies


class RollingHorizonSAAMPCPolicy(BasePolicy):
    name = "RH-SAA-MPC"

    def __init__(
        self,
        horizon: int = 8,
        scenarios: int = 8,
        time_limit: float = 30.0,
        mip_gap: float = 0.02,
        threads: int | None = None,
        integer: bool = True,
        terminal_backlog_mult: float = 2.0,
        terminal_inventory_mult: float = 0.0,
        terminal_stock_target_mode: str = "none",
        terminal_stock_target_scale: float = 1.0,
        terminal_stock_target_penalty_mult: float = 0.0,
        planning_anchor_mode: str = "none",
        order_nonanticipativity: str = "first_stage_only",
        shared_order_prefix: int = 0,
        seed: int = 12345,
        deterministic: bool = False,
        name: str | None = None,
    ):
        self.horizon = int(horizon)
        self.scenarios = int(scenarios)
        self.time_limit = float(time_limit)
        self.mip_gap = float(mip_gap)
        self.threads = threads
        self.integer = bool(integer)
        self.terminal_backlog_mult = float(terminal_backlog_mult)
        self.terminal_inventory_mult = float(terminal_inventory_mult)
        self.terminal_stock_target_mode = str(terminal_stock_target_mode)
        self.terminal_stock_target_scale = float(max(0.0, terminal_stock_target_scale))
        self.terminal_stock_target_penalty_mult = float(max(0.0, terminal_stock_target_penalty_mult))
        self.planning_anchor_mode = str(planning_anchor_mode)
        self.order_nonanticipativity = str(order_nonanticipativity)
        self.shared_order_prefix = int(max(0, shared_order_prefix))
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.name = name or ("CE-MPC" if deterministic else "RH-SAA-MPC")
        self._fallback: NVDPolicy | None = None
        self._failures = 0
        self.last_solver_status = ""
        self.last_solver_gap = np.nan
        self.last_runtime_seconds = np.nan

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        try:
            allocations, orders = self._solve_mpc(env, obs)
            return ControlAction(allocations=allocations, orders=orders)
        except Exception as exc:
            self._failures += 1
            if self._failures <= 5:
                print(f"[WARN] {self.name} failed at episode={_episode_id(env)} t={obs.t}: {exc}; using NVD fallback.")
            if self._fallback is None:
                self._fallback = NVDPolicy(env.instance)
            return self._fallback.act(env, obs)

    def _rng_for_state(self, env: ATOEnv, obs: Observation) -> np.random.Generator:
        return np.random.default_rng(self.seed + 1_000_003 * _episode_id(env) + 97 * int(obs.t))

    def _solve_mpc(self, env: ATOEnv, obs: Observation) -> tuple[List[Tuple[int, int, float]], np.ndarray]:
        os.environ["LC_ALL"] = "C"
        import gurobipy as gp
        from gurobipy import GRB

        inst = env.instance
        t0 = int(obs.t)
        u_end = min(inst.T - 1, t0 + max(1, self.horizon) - 1)
        periods = list(range(t0, u_end + 1))
        if not periods:
            return [], np.zeros(inst.J, dtype=float)

        rng = self._rng_for_state(env, obs)
        planning = self._make_planning_set(inst, env, obs, rng)
        q_ub = self._q_upper_bound(inst, obs, planning, u_end)
        terminal_stock_target = self._terminal_stock_target(inst)

        model = gp.Model(self.name)
        _set_gurobi_params(model, self.time_limit, self.mip_gap, self.threads)
        decision_type = GRB.INTEGER if self.integer else GRB.CONTINUOUS

        q0 = {
            j: model.addVar(lb=0.0, ub=float(q_ub[j]), vtype=decision_type, name=f"q0_{j}")
            for j in range(inst.J)
        }
        current_cohorts = [(int(i), int(s), float(rem), np.asarray(bom, dtype=float)) for i, s, rem, bom in obs.revealed]
        current_cohorts = [(i, s, rem, bom) for i, s, rem, bom in current_cohorts if rem > 1e-9]
        x0 = {
            (i, s): model.addVar(lb=0.0, ub=float(rem), vtype=decision_type, name=f"x0_{i}_{s}")
            for i, s, rem, _bom in current_cohorts
        }
        for j in range(inst.J):
            terms = [
                float(bom[j]) * x0[(i, s)]
                for i, s, _rem, bom in current_cohorts
                if float(bom[j]) > 1e-12
            ]
            if terms:
                model.addConstr(gp.quicksum(terms) <= float(obs.inventory[j]), name=f"current_inventory_{j}")

        q: dict[tuple[int, int, int], Any] = {}
        x: dict[tuple[int, int, int, int], Any] = {}
        y: dict[tuple[int, int, int], Any] = {}
        x_by_cohort: dict[tuple[int, int, int], list[tuple[int, Any]]] = {}
        consumption_by_period: dict[tuple[int, int, int], list[Any]] = {}
        shared_future_q: dict[tuple[int, int], Any] = {}
        shared_orders = self.order_nonanticipativity.strip().lower() in {
            "shared_open_loop",
            "shared",
            "open_loop",
        }
        shared_prefix = self.order_nonanticipativity.strip().lower() in {
            "shared_prefix",
            "prefix",
        }

        def add_x_var(k: int, scenario: PlanningScenario, i: int, s: int, u: int, var: Any) -> None:
            x[(k, i, s, u)] = var
            x_by_cohort.setdefault((k, i, s), []).append((u, var))
            bom_vec = np.asarray(scenario.bom[i, s], dtype=float)
            for j in np.flatnonzero(bom_vec > 1e-12):
                coeff = float(bom_vec[j])
                for uu in periods:
                    if uu >= u:
                        consumption_by_period.setdefault((k, int(j), uu), []).append(coeff * var)

        for k, scenario in enumerate(planning):
            for u in periods:
                for j in range(inst.J):
                    if u == t0:
                        q[(k, j, u)] = q0[j]
                    elif shared_orders or (shared_prefix and u <= t0 + self.shared_order_prefix):
                        key = (j, u)
                        if key not in shared_future_q:
                            shared_future_q[key] = model.addVar(
                                lb=0.0,
                                ub=float(q_ub[j]),
                                vtype=decision_type,
                                name=f"q_shared_{j}_{u}",
                            )
                        q[(k, j, u)] = shared_future_q[key]
                    else:
                        q[(k, j, u)] = model.addVar(
                            lb=0.0,
                            ub=float(q_ub[j]),
                            vtype=decision_type,
                            name=f"q_{k}_{j}_{u}",
                        )
                    y[(k, j, u)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"y_{k}_{j}_{u}")

            for u in periods:
                for i in range(inst.I):
                    reveal_t = int(inst.design_lead_times[i])
                    for s in range(0, u + 1):
                        qty0 = _cohort_quantity(env, scenario, t0, i, s)
                        if qty0 <= 1e-9 or u < max(t0, s + reveal_t):
                            continue
                        if u == t0:
                            if (i, s) in x0:
                                add_x_var(k, scenario, i, s, u, x0[(i, s)])
                            continue
                        var = model.addVar(
                            lb=0.0,
                            ub=float(qty0),
                            vtype=decision_type,
                            name=f"x_{k}_{i}_{s}_{u}",
                        )
                        add_x_var(k, scenario, i, s, u, var)

        for k, scenario in enumerate(planning):
            for i in range(inst.I):
                for s in range(0, u_end + 1):
                    qty0 = _cohort_quantity(env, scenario, t0, i, s)
                    if qty0 <= 1e-9:
                        continue
                    served = [var for _uu, var in x_by_cohort.get((k, i, s), [])]
                    if served:
                        model.addConstr(gp.quicksum(served) <= float(qty0), name=f"demand_{k}_{i}_{s}")

        for k, scenario in enumerate(planning):
            for j in range(inst.J):
                for u in periods:
                    existing_arrivals = float(np.sum(scenario.existing_arrivals[t0 : u + 1, j]))
                    arrivals = [
                        q[(k, j, v)]
                        for v in periods
                        if v < u and v + int(scenario.lead_times[v, j]) <= u
                    ]
                    consumption = consumption_by_period.get((k, j, u), [])
                    model.addConstr(
                        y[(k, j, u)]
                        == float(obs.inventory[j]) + existing_arrivals + gp.quicksum(arrivals) - gp.quicksum(consumption),
                        name=f"material_{k}_{j}_{u}",
                    )

        obj_terms = []
        for k, scenario in enumerate(planning):
            for u in periods:
                obj_terms.append(gp.quicksum(float(inst.ordering_costs[j]) * q[(k, j, u)] for j in range(inst.J)))
                obj_terms.append(gp.quicksum(float(inst.holding_costs[j]) * y[(k, j, u)] for j in range(inst.J)))
                for i in range(inst.I):
                    due_offset = int(inst.design_lead_times[i] + inst.delivery_window)
                    for s in range(0, u + 1):
                        qty0 = _cohort_quantity(env, scenario, t0, i, s)
                        if qty0 <= 1e-9 or u <= s + due_offset:
                            continue
                        cohort_vars = x_by_cohort.get((k, i, s), [])
                        served_to_u = gp.quicksum(
                            var for vv, var in cohort_vars if vv <= u
                        )
                        obj_terms.append(float(inst.backlog_costs[i]) * (float(qty0) - served_to_u))
            if self.terminal_backlog_mult > 0:
                for i in range(inst.I):
                    for s in range(0, u_end + 1):
                        qty0 = _cohort_quantity(env, scenario, t0, i, s)
                        if qty0 <= 1e-9:
                            continue
                        cohort_vars = x_by_cohort.get((k, i, s), [])
                        served_all = gp.quicksum(
                            var for vv, var in cohort_vars if vv <= u_end
                        )
                        obj_terms.append(
                            self.terminal_backlog_mult * float(inst.backlog_costs[i]) * (float(qty0) - served_all)
                        )
            if self.terminal_inventory_mult > 0:
                for j in range(inst.J):
                    obj_terms.append(self.terminal_inventory_mult * float(inst.holding_costs[j]) * y[(k, j, u_end)])
            if self.terminal_stock_target_penalty_mult > 0 and np.any(terminal_stock_target > 1e-9):
                for j in range(inst.J):
                    short = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"terminal_short_{k}_{j}")
                    terminal_position = _terminal_inventory_position_expr(
                        scenario=scenario,
                        obs=obs,
                        q=q,
                        y_end=y[(k, j, u_end)],
                        k=k,
                        j=j,
                        periods=periods,
                        u_end=u_end,
                    )
                    model.addConstr(
                        short >= float(terminal_stock_target[j]) - terminal_position,
                        name=f"terminal_stock_target_{k}_{j}",
                    )
                    obj_terms.append(
                        self.terminal_stock_target_penalty_mult * float(inst.holding_costs[j]) * short
                    )

        model.setObjective((1.0 / max(1, len(planning))) * gp.quicksum(obj_terms), GRB.MINIMIZE)
        model.optimize()
        accepted = {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
        self.last_solver_status = _gurobi_status_name(model.Status)
        self.last_runtime_seconds = float(getattr(model, "Runtime", np.nan))
        try:
            self.last_solver_gap = float(model.MIPGap)
        except Exception:
            self.last_solver_gap = np.nan
        if model.Status not in accepted or model.SolCount <= 0:
            raise RuntimeError(f"Gurobi status={model.Status}, solutions={model.SolCount}")

        orders = np.zeros(inst.J, dtype=float)
        for j in range(inst.J):
            val = max(0.0, float(q0[j].X))
            orders[j] = float(round(val)) if self.integer else val
        allocations: List[Tuple[int, int, float]] = []
        for (i, s), var in x0.items():
            val = max(0.0, float(var.X))
            qty = float(round(val)) if self.integer else val
            if qty > 1e-9:
                allocations.append((int(i), int(s), qty))
        return allocations, orders

    def _q_upper_bound(
        self,
        inst: ProblemInstance,
        obs: Observation,
        planning: Sequence[PlanningScenario],
        u_end: int,
    ) -> np.ndarray:
        if self._fallback is None:
            self._fallback = NVDPolicy(inst)
        base = np.maximum(1.0, np.asarray(self._fallback.base_stock, dtype=float))
        ub = np.maximum(10.0, 3.0 * base + 10.0)
        for scenario in planning:
            req = np.zeros(inst.J, dtype=float)
            for s in range(obs.t, u_end + 1):
                for i in range(inst.I):
                    req += float(scenario.demand[i, s]) * scenario.bom[i, s]
            ub = np.maximum(ub, np.ceil(req + obs.inventory + obs.outstanding + 5.0))
        return np.maximum(10.0, ub)

    def _terminal_stock_target(self, inst: ProblemInstance) -> np.ndarray:
        mode = self.terminal_stock_target_mode.strip().lower()
        if mode in {"", "none"} or self.terminal_stock_target_penalty_mult <= 0:
            return np.zeros(inst.J, dtype=float)
        if mode == "nvd":
            target = np.asarray(NVDPolicy(inst).base_stock, dtype=float)
        elif mode in {"mean_leadtime", "mean-leadtime", "mean"}:
            target = (inst.template_bom.T @ inst.demand_lambdas) * float(inst.expected_lead_time)
        else:
            raise ValueError(f"unknown terminal stock target mode '{self.terminal_stock_target_mode}'")
        return np.maximum(0.0, self.terminal_stock_target_scale * target)

    def _make_planning_set(
        self,
        inst: ProblemInstance,
        env: ATOEnv,
        obs: Observation,
        rng: np.random.Generator,
    ) -> List[PlanningScenario]:
        horizon = max(1, self.horizon)
        if self.deterministic:
            return _make_planning_scenarios(
                inst,
                env,
                obs,
                K=1,
                H=horizon,
                rng=rng,
                deterministic=True,
            )
        k_total = max(1, self.scenarios)
        if self.planning_anchor_mode.strip().lower() in {"mean", "mean_scenario", "deterministic"}:
            anchored = _make_planning_scenarios(
                inst,
                env,
                obs,
                K=1,
                H=horizon,
                rng=rng,
                deterministic=True,
            )
            stochastic = _make_planning_scenarios(
                inst,
                env,
                obs,
                K=max(0, k_total - 1),
                H=horizon,
                rng=rng,
                deterministic=False,
            )
            return anchored + stochastic
        return _make_planning_scenarios(
            inst,
            env,
            obs,
            K=k_total,
            H=horizon,
            rng=rng,
            deterministic=False,
        )


class CertaintyEquivalentMPCPolicy(RollingHorizonSAAMPCPolicy):
    name = "CE-MPC"

    def __init__(
        self,
        horizon: int = 8,
        time_limit: float = 30.0,
        mip_gap: float = 0.02,
        threads: int | None = None,
        integer: bool = True,
        terminal_backlog_mult: float = 2.0,
        terminal_inventory_mult: float = 0.0,
        terminal_stock_target_mode: str = "none",
        terminal_stock_target_scale: float = 1.0,
        terminal_stock_target_penalty_mult: float = 0.0,
        planning_anchor_mode: str = "none",
        order_nonanticipativity: str = "first_stage_only",
        shared_order_prefix: int = 0,
        seed: int = 54321,
    ):
        super().__init__(
            horizon=horizon,
            scenarios=1,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            integer=integer,
            terminal_backlog_mult=terminal_backlog_mult,
            terminal_inventory_mult=terminal_inventory_mult,
            terminal_stock_target_mode=terminal_stock_target_mode,
            terminal_stock_target_scale=terminal_stock_target_scale,
            terminal_stock_target_penalty_mult=terminal_stock_target_penalty_mult,
            planning_anchor_mode=planning_anchor_mode,
            order_nonanticipativity=order_nonanticipativity,
            shared_order_prefix=shared_order_prefix,
            seed=seed,
            deterministic=True,
            name="CE-MPC",
        )


def _set_gurobi_params(
    model: Any,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    threads: int | None = None,
) -> None:
    model.Params.OutputFlag = 0
    if time_limit is not None and time_limit > 0:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None and mip_gap >= 0:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None and threads > 0:
        model.Params.Threads = int(threads)


def _gurobi_status_name(status: int) -> str:
    names = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT",
        16: "WORK_LIMIT",
        17: "MEM_LIMIT",
    }
    return names.get(int(status), f"STATUS_{status}")


def _episode_id(env: ATOEnv) -> int:
    if env.scenario is None:
        return 0
    return int(getattr(env.scenario, "episode", 0))


def _cohort_quantity(
    env: ATOEnv,
    scenario: PlanningScenario,
    t0: int,
    i: int,
    s: int,
) -> float:
    if s <= t0:
        return float(env.remaining[i, s])
    if s < scenario.demand.shape[1]:
        return float(scenario.demand[i, s])
    return 0.0


def _terminal_inventory_position_expr(
    scenario: PlanningScenario,
    obs: Observation,
    q: dict[tuple[int, int, int], Any],
    y_end: Any,
    k: int,
    j: int,
    periods: Sequence[int],
    u_end: int,
) -> Any:
    existing_arrivals_through_end = float(np.sum(scenario.existing_arrivals[int(obs.t) : u_end + 1, j]))
    existing_after_end = max(0.0, float(obs.outstanding[j]) - existing_arrivals_through_end)
    future_orders = [
        q[(k, j, v)]
        for v in periods
        if v + int(scenario.lead_times[v, j]) > u_end
    ]
    return y_end + existing_after_end + sum(future_orders)


def _make_planning_scenarios(
    inst: ProblemInstance,
    env: ATOEnv,
    obs: Observation,
    K: int,
    H: int,
    rng: np.random.Generator,
    deterministic: bool,
) -> List[PlanningScenario]:
    assert env.scenario is not None
    t0 = int(obs.t)
    u_end = min(inst.T - 1, t0 + H - 1)
    means = _period_demand_means(inst)
    scenarios: List[PlanningScenario] = []
    for _ in range(max(1, K)):
        if deterministic:
            demand_path = means.copy()
            bom_path = np.broadcast_to(inst.template_bom[:, None, :], (inst.I, inst.T, inst.J)).copy()
            lead_times = np.full(
                (inst.T, inst.J),
                int(round(inst.expected_lead_time)),
                dtype=int,
            )
        else:
            demand_path = _sample_demand_path(inst, means, rng).astype(float)
            bom_path, _realized_cv = _sample_order_specific_bom(inst, rng)
            lead_times = rng.integers(
                low=inst.min_replenishment_lead_time,
                high=inst.max_replenishment_lead_time + 1,
                size=(inst.T, inst.J),
            )

        demand = np.zeros((inst.I, inst.T), dtype=float)
        bom = np.zeros((inst.I, inst.T, inst.J), dtype=float)
        for i in range(inst.I):
            for s in range(0, t0 + 1):
                demand[i, s] = float(env.remaining[i, s])
                if env.revealed[i, s]:
                    bom[i, s] = env.scenario.realized_bom[i, s]
                else:
                    bom[i, s] = bom_path[i, s]
            for s in range(t0 + 1, u_end + 1):
                demand[i, s] = float(demand_path[i, s])
                bom[i, s] = bom_path[i, s]

        clipped_leads = np.clip(
            np.rint(lead_times).astype(int),
            inst.min_replenishment_lead_time,
            inst.max_replenishment_lead_time,
        )
        existing_arrivals = _sample_conditional_pipeline_arrivals(inst, obs, rng, H, deterministic)
        scenarios.append(
            PlanningScenario(
                demand=demand,
                bom=bom,
                lead_times=clipped_leads,
                existing_arrivals=existing_arrivals,
            )
        )
    return scenarios


def _sample_conditional_pipeline_arrivals(
    inst: ProblemInstance,
    obs: Observation,
    rng: np.random.Generator,
    H: int,
    deterministic: bool,
) -> np.ndarray:
    arrivals = np.zeros((inst.T, inst.J), dtype=float)
    t0 = int(obs.t)
    last_t = min(inst.T - 1, t0 + H - 1)
    for j in range(inst.J):
        for age_idx in range(obs.pipeline_by_age.shape[1]):
            age = age_idx + 1
            qty = float(obs.pipeline_by_age[j, age_idx])
            if qty <= 1e-9:
                continue
            possible_leads = np.arange(
                max(inst.min_replenishment_lead_time, age + 1),
                inst.max_replenishment_lead_time + 1,
                dtype=int,
            )
            if possible_leads.size == 0:
                at = min(last_t, t0 + 1)
                if at < inst.T:
                    arrivals[at, j] += qty
                continue
            probs = np.full(possible_leads.size, 1.0 / possible_leads.size)
            if deterministic or not math.isclose(qty, round(qty), rel_tol=0.0, abs_tol=1e-9):
                quantities = qty * probs
            else:
                quantities = rng.multinomial(int(round(qty)), probs).astype(float)
            for lead, lead_qty in zip(possible_leads, quantities):
                if lead_qty <= 1e-9:
                    continue
                at = t0 + int(lead - age)
                if t0 <= at <= last_t and at < inst.T:
                    arrivals[at, j] += float(lead_qty)
    return arrivals
