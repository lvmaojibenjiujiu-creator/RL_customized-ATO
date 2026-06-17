from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from benchmarks.dhp_sbr import DHPSBRPolicy

from .env import ATOEnv, ControlAction, Observation
from .mpc_policies import RollingHorizonSAAMPCPolicy, canonical_policy_name
from .policies import BasePolicy, NVDPolicy, solve_obca_allocation
from .scenario import ProblemInstance


@dataclass
class H3Table:
    S: np.ndarray
    R: np.ndarray
    z_max: int


@dataclass
class H3CalibrationResult:
    tables: Dict[Tuple[int, int], H3Table]
    table_path: str
    metadata: Dict[str, Any]


def add_spec_benchmark_arguments(parser: Any) -> None:
    parser.add_argument("--rh-spt-horizon", type=int, default=8, help="0 means auto from lead-time/design window.")
    parser.add_argument("--rh-spt-scenarios", type=int, default=15)
    parser.add_argument("--rh-spt-time-limit", type=float, default=2.0)
    parser.add_argument("--rh-spt-mip-gap", type=float, default=0.1)
    parser.add_argument("--rh-spt-threads", type=int, default=0)
    parser.add_argument("--rh-spt-terminal-backlog-mult", type=float, default=1.0)
    parser.add_argument("--rh-spt-terminal-inventory-mult", type=float, default=0.0)
    parser.add_argument(
        "--rh-spt-terminal-stock-target-mode",
        choices=["none", "mean_leadtime", "nvd"],
        default="none",
    )
    parser.add_argument("--rh-spt-terminal-stock-target-scale", type=float, default=1.0)
    parser.add_argument("--rh-spt-terminal-stock-target-penalty-mult", type=float, default=0.0)
    parser.add_argument("--rh-spt-planning-anchor-mode", choices=["none", "mean"], default="none")
    parser.add_argument(
        "--rh-spt-order-nonanticipativity",
        choices=["first_stage_only", "shared_prefix", "shared_open_loop"],
        default="shared_prefix",
    )
    parser.add_argument("--rh-spt-shared-order-prefix", type=int, default=1)
    parser.add_argument("--rh-spt-allocation-mode", choices=["target_tracking", "direct", "obca"], default="direct")
    parser.add_argument("--rh-spt-order-floor-mode", choices=["none", "fallback_base_stock"], default="none")
    parser.add_argument("--rh-spt-order-floor-scale", type=float, default=1.0)
    parser.add_argument("--rh-spt-large-mode", choices=["auto", "full", "compressed"], default="auto")
    parser.add_argument("--rh-spt-compressed-min-components", type=int, default=80)
    parser.add_argument("--rh-spt-compressed-load-scale", type=float, default=0.7)
    parser.add_argument("--rh-spt-compressed-future-scale", type=float, default=0.0)
    parser.add_argument("--rh-spt-continuous", action="store_true", default=True)
    parser.add_argument("--rh-spt-integer", action="store_false", dest="rh_spt_continuous")
    parser.add_argument("--rh-spt-seed", type=int, default=None)
    parser.add_argument("--h3-z-max", type=int, default=10)
    parser.add_argument("--h3-cal-paths", type=int, default=12)
    parser.add_argument("--h3-cal-horizon", type=int, default=60)
    parser.add_argument("--h3-warmup", type=int, default=10)
    parser.add_argument("--h3-table-dir", default="outputs/h3_sbr_calibration")
    parser.add_argument("--h3-seed", type=int, default=None)
    parser.add_argument("--dhp-sbr-calibration-mode", choices=["saa_grid", "formula_init"], default="saa_grid")
    parser.add_argument("--dhp-sbr-edge-prob-threshold", type=float, default=0.05)
    parser.add_argument("--dhp-sbr-z-max", type=int, default=10)
    parser.add_argument("--dhp-sbr-unrevealed-load-weight-mode", choices=["urgency", "full", "none"], default="none")
    parser.add_argument("--dhp-sbr-imminent-pipeline-window-mode", choices=["mean_lead_time", "none"], default="mean_lead_time")
    parser.add_argument("--dhp-sbr-imminent-pipeline-window", type=int, default=0)
    parser.add_argument(
        "--dhp-sbr-replenishment-target-aggregation",
        choices=["min", "demand_weighted_quantile", "demand_weighted_mean"],
        default="min",
    )
    parser.add_argument("--dhp-sbr-n-cal-paths-small", type=int, default=200)
    parser.add_argument("--dhp-sbr-n-cal-paths-large", type=int, default=80)
    parser.add_argument("--dhp-sbr-t-cal", type=int, default=200)
    parser.add_argument("--dhp-sbr-warmup", type=int, default=50)
    parser.add_argument("--dhp-sbr-safety-factor", type=float, default=0.0)
    parser.add_argument("--dhp-sbr-max-s-candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--dhp-sbr-reserve-factor", type=float, default=0.5)
    parser.add_argument("--dhp-sbr-state-sensitivity", type=float, default=0.0)
    parser.add_argument("--dhp-sbr-restrict-r-leq-s", action="store_true", default=True)
    parser.add_argument("--dhp-sbr-allow-r-gt-s", action="store_false", dest="dhp_sbr_restrict_r_leq_s")
    parser.add_argument("--dhp-sbr-candidate-step-small", type=int, default=1)
    parser.add_argument("--dhp-sbr-candidate-step-large", type=int, default=2)
    parser.add_argument("--dhp-sbr-max-grid-points", type=int, default=9)
    parser.add_argument("--dhp-sbr-cache-dir", default="outputs/dhp_sbr_calibration")
    parser.add_argument("--dhp-sbr-no-cache", action="store_true")
    parser.add_argument("--dhp-sbr-calibration-time-limit-seconds", type=float, default=3600.0)
    parser.add_argument("--dhp-sbr-seed", type=int, default=None)


def spec_benchmark_policies_from_request(
    requested: Iterable[str],
    args: Any,
    instance: ProblemInstance,
    seed: int,
    fallback_policy: BasePolicy | None = None,
) -> List[BasePolicy]:
    tokens = {canonical_policy_name(name) for name in requested}
    policies: List[BasePolicy] = []
    if {"RH-SPT", "RHSPT"}.intersection(tokens):
        threads = int(args.rh_spt_threads) if int(args.rh_spt_threads) > 0 else None
        if int(args.rh_spt_horizon) > 0:
            horizon = int(args.rh_spt_horizon)
        else:
            horizon = int(
                min(
                    12,
                    instance.T,
                    max(
                        int(instance.delivery_window),
                        int(instance.max_replenishment_lead_time + np.max(instance.design_lead_times) + 2),
                    ),
                )
            )
        policies.append(
            RHSPTPolicy(
                instance,
                horizon=horizon,
                n_scenarios=args.rh_spt_scenarios,
                time_limit=args.rh_spt_time_limit,
                mip_gap=args.rh_spt_mip_gap,
                threads=threads,
                terminal_backlog_mult=args.rh_spt_terminal_backlog_mult,
                terminal_inventory_mult=args.rh_spt_terminal_inventory_mult,
                terminal_stock_target_mode=args.rh_spt_terminal_stock_target_mode,
                terminal_stock_target_scale=args.rh_spt_terminal_stock_target_scale,
                terminal_stock_target_penalty_mult=args.rh_spt_terminal_stock_target_penalty_mult,
                planning_anchor_mode=args.rh_spt_planning_anchor_mode,
                order_nonanticipativity=args.rh_spt_order_nonanticipativity,
                shared_order_prefix=args.rh_spt_shared_order_prefix,
                allocation_mode=args.rh_spt_allocation_mode,
                order_floor_mode=args.rh_spt_order_floor_mode,
                order_floor_scale=args.rh_spt_order_floor_scale,
                large_mode=args.rh_spt_large_mode,
                compressed_min_components=args.rh_spt_compressed_min_components,
                compressed_load_scale=args.rh_spt_compressed_load_scale,
                compressed_future_scale=args.rh_spt_compressed_future_scale,
                integer_first_stage=not bool(args.rh_spt_continuous),
                fallback_policy=fallback_policy,
                seed=int(seed if args.rh_spt_seed is None else args.rh_spt_seed),
            )
        )
    if {"H3-SBR", "H3SBR"}.intersection(tokens):
        policies.append(
            H3SBRPolicy(
                instance,
                z_max=args.h3_z_max,
                n_cal_paths=args.h3_cal_paths,
                calibration_horizon=args.h3_cal_horizon,
                warmup=args.h3_warmup,
                table_dir=args.h3_table_dir,
                fallback_policy=fallback_policy,
                seed=int(seed if args.h3_seed is None else args.h3_seed),
            )
        )
    if {"DHP-SBR", "DHPSBR"}.intersection(tokens):
        calibration_config = {
            "calibration_mode": args.dhp_sbr_calibration_mode,
            "edge_prob_threshold": args.dhp_sbr_edge_prob_threshold,
            "Z_max": args.dhp_sbr_z_max,
            "n_cal_paths_small": args.dhp_sbr_n_cal_paths_small,
            "n_cal_paths_large": args.dhp_sbr_n_cal_paths_large,
            "T_cal": args.dhp_sbr_t_cal,
            "warmup": args.dhp_sbr_warmup,
            "safety_factor": args.dhp_sbr_safety_factor,
            "max_S_candidate_multiplier": args.dhp_sbr_max_s_candidate_multiplier,
            "reserve_factor": args.dhp_sbr_reserve_factor,
            "state_sensitivity": args.dhp_sbr_state_sensitivity,
            "restrict_R_leq_S": args.dhp_sbr_restrict_r_leq_s,
            "candidate_step_small": args.dhp_sbr_candidate_step_small,
            "candidate_step_large": args.dhp_sbr_candidate_step_large,
            "max_grid_points": args.dhp_sbr_max_grid_points,
            "cache_tables": not bool(args.dhp_sbr_no_cache),
            "cache_dir": args.dhp_sbr_cache_dir,
            "calibration_time_limit_seconds": args.dhp_sbr_calibration_time_limit_seconds,
        }
        online_config = {
            "unrevealed_load_weight_mode": args.dhp_sbr_unrevealed_load_weight_mode,
            "imminent_pipeline_window_mode": args.dhp_sbr_imminent_pipeline_window_mode,
            "imminent_pipeline_window": args.dhp_sbr_imminent_pipeline_window,
            "replenishment_target_aggregation": args.dhp_sbr_replenishment_target_aggregation,
        }
        policies.append(
            DHPSBRPolicy(
                instance,
                calibration_config=calibration_config,
                online_config=online_config,
                fallback_policy=fallback_policy,
                seed=int(seed if args.dhp_sbr_seed is None else args.dhp_sbr_seed),
            )
        )
    return policies


class RHSPTPolicy(RollingHorizonSAAMPCPolicy):
    name = "RH-SPT"

    def __init__(
        self,
        instance: ProblemInstance,
        horizon: int = 8,
        n_scenarios: int = 15,
        time_limit: float = 2.0,
        mip_gap: float = 0.1,
        threads: int | None = None,
        terminal_backlog_mult: float = 1.0,
        terminal_inventory_mult: float = 0.0,
        terminal_stock_target_mode: str = "none",
        terminal_stock_target_scale: float = 1.0,
        terminal_stock_target_penalty_mult: float = 0.0,
        planning_anchor_mode: str = "none",
        order_nonanticipativity: str = "shared_prefix",
        shared_order_prefix: int = 1,
        allocation_mode: str = "direct",
        order_floor_mode: str = "none",
        order_floor_scale: float = 1.0,
        large_mode: str = "auto",
        compressed_min_components: int = 80,
        compressed_load_scale: float = 0.7,
        compressed_future_scale: float = 0.0,
        integer_first_stage: bool = False,
        fallback_policy: BasePolicy | None = None,
        seed: int = 0,
    ):
        super().__init__(
            horizon=horizon,
            scenarios=n_scenarios,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            integer=integer_first_stage,
            terminal_backlog_mult=terminal_backlog_mult,
            terminal_inventory_mult=terminal_inventory_mult,
            terminal_stock_target_mode=terminal_stock_target_mode,
            terminal_stock_target_scale=terminal_stock_target_scale,
            terminal_stock_target_penalty_mult=terminal_stock_target_penalty_mult,
            planning_anchor_mode=planning_anchor_mode,
            order_nonanticipativity=order_nonanticipativity,
            shared_order_prefix=shared_order_prefix,
            seed=seed,
            deterministic=False,
            name="RH-SPT",
        )
        self.instance = instance
        self._external_fallback = fallback_policy
        self.allocation_mode = str(allocation_mode)
        self.order_floor_mode = str(order_floor_mode)
        self.order_floor_scale = float(max(0.0, order_floor_scale))
        self.large_mode = str(large_mode)
        self.compressed_min_components = int(max(1, compressed_min_components))
        self.compressed_load_scale = float(max(0.0, compressed_load_scale))
        self.compressed_future_scale = float(max(0.0, compressed_future_scale))
        self.fallback_count = 0
        self.last_solver_status = ""
        self.last_solver_gap = np.nan

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        started = time.perf_counter()
        if self._use_compressed_mode(env.instance):
            return self._act_compressed(env, obs, started)
        try:
            desired_allocations, orders = self._solve_mpc(env, obs)
            if self.allocation_mode == "direct":
                allocations = desired_allocations
            elif self.allocation_mode == "obca":
                beta_late = float(getattr(self._external_fallback, "beta_late", 1.0))
                allocations = solve_obca_allocation(env, obs, beta_late=beta_late, solver="gurobi")
            else:
                desired = {(int(i), int(s)): float(q) for i, s, q in desired_allocations}
                allocations = _target_tracking_allocation(env, obs, desired)
            orders = self._apply_order_floor(env, obs, allocations, orders)
            self.last_runtime_seconds = time.perf_counter() - started
            return ControlAction(allocations=allocations, orders=orders)
        except Exception as exc:
            self.fallback_count += 1
            self.last_solver_status = f"FALLBACK:{type(exc).__name__}"
            print(f"[RH-SPT] fallback at episode={_episode_id(env)} t={obs.t}: {exc}")
            fallback = self._external_fallback or self._fallback or NVDPolicy(env.instance)
            self.last_runtime_seconds = time.perf_counter() - started
            return fallback.act(env, obs)

    def _use_compressed_mode(self, inst: ProblemInstance) -> bool:
        mode = self.large_mode.strip().lower()
        if mode == "compressed":
            return True
        if mode == "full":
            return False
        return int(inst.J) >= self.compressed_min_components

    def _act_compressed(self, env: ATOEnv, obs: Observation, started: float) -> ControlAction:
        beta_late = float(getattr(self._external_fallback, "beta_late", 1.0))
        allocations = solve_obca_allocation(env, obs, beta_late=beta_late, solver="gurobi")
        local_inventory, remaining_after = _post_allocation_state(obs, allocations, env.remaining)
        base = self._compressed_base_stock(env.instance)
        current_load = self._current_component_load(env, obs, remaining_after)
        future_load = self._future_component_load(env, obs) if self.compressed_future_scale > 1e-12 else 0.0
        target = (
            base
            + self.compressed_load_scale * current_load
            + self.compressed_future_scale * future_load
        )
        ip = local_inventory + obs.outstanding
        orders = np.maximum(0.0, np.ceil(target - ip - 1e-9))
        orders[np.abs(orders) < 1e-9] = 0.0
        self.last_solver_status = "COMPRESSED"
        self.last_solver_gap = np.nan
        self.last_runtime_seconds = time.perf_counter() - started
        return ControlAction(allocations=allocations, orders=orders)

    def _compressed_base_stock(self, inst: ProblemInstance) -> np.ndarray:
        fallback = self._external_fallback
        if fallback is not None and hasattr(fallback, "base_stock"):
            return np.maximum(0.0, np.asarray(getattr(fallback, "base_stock"), dtype=float))
        if self._fallback is None:
            self._fallback = NVDPolicy(inst)
        return np.maximum(0.0, np.asarray(self._fallback.base_stock, dtype=float))

    def _current_component_load(self, env: ATOEnv, obs: Observation, remaining_after: np.ndarray) -> np.ndarray:
        inst = env.instance
        load = np.zeros(inst.J, dtype=float)
        scenario = env.scenario
        for i in range(inst.I):
            for s in range(min(obs.t, inst.T - 1) + 1):
                rem = float(remaining_after[i, s])
                if rem <= 1e-9:
                    continue
                if env.revealed[i, s] and scenario is not None:
                    bom = scenario.realized_bom[i, s]
                else:
                    bom = inst.template_bom[i]
                due = int(s + inst.design_lead_times[i] + inst.delivery_window)
                urgency = 1.0 + 0.25 * max(0, obs.t - due)
                load += urgency * rem * bom
        return load

    def _future_component_load(self, env: ATOEnv, obs: Observation) -> np.ndarray:
        inst = env.instance
        rng = self._rng_for_state(env, obs)
        planning = self._make_planning_set(inst, env, obs, rng)
        if not planning:
            return np.zeros(inst.J, dtype=float)
        u_end = min(inst.T - 1, int(obs.t) + max(1, self.horizon) - 1)
        loads = []
        for scenario in planning:
            load = np.zeros(inst.J, dtype=float)
            for u in range(int(obs.t) + 1, u_end + 1):
                for i in range(inst.I):
                    release = u + int(inst.design_lead_times[i])
                    if release > u_end:
                        continue
                    load += float(scenario.demand[i, u]) * scenario.bom[i, u]
            loads.append(load)
        return np.mean(loads, axis=0)

    def _apply_order_floor(
        self,
        env: ATOEnv,
        obs: Observation,
        allocations: List[Tuple[int, int, float]],
        orders: np.ndarray,
    ) -> np.ndarray:
        if self.order_floor_mode != "fallback_base_stock" or self.order_floor_scale <= 1e-12:
            return orders
        fallback = self._external_fallback
        if fallback is None or not hasattr(fallback, "base_stock"):
            return orders
        local_inventory, remaining_after = _post_allocation_state(obs, allocations, env.remaining)
        known_requirement = remaining_after.sum(axis=1) @ self.instance.template_bom
        known_scale = float(getattr(fallback, "known_requirement_scale", 0.0))
        target = self.order_floor_scale * (
            np.asarray(getattr(fallback, "base_stock"), dtype=float)
            + known_scale * known_requirement
        )
        floor_orders = np.maximum(0.0, np.ceil(target - (local_inventory + obs.outstanding) - 1e-9))
        return np.maximum(np.asarray(orders, dtype=float), floor_orders)


class H3SBRPolicy(BasePolicy):
    name = "H3-SBR"

    def __init__(
        self,
        instance: ProblemInstance,
        z_max: int = 10,
        n_cal_paths: int = 12,
        calibration_horizon: int = 60,
        warmup: int = 10,
        table_dir: str | Path = "outputs/h3_sbr_calibration",
        fallback_policy: BasePolicy | None = None,
        seed: int = 0,
    ):
        self.instance = instance
        self.z_max = int(max(1, z_max))
        self.n_cal_paths = int(max(1, n_cal_paths))
        self.calibration_horizon = int(max(10, calibration_horizon))
        self.warmup = int(max(0, warmup))
        self.table_dir = Path(table_dir)
        self.seed = int(seed)
        self.fallback_policy = fallback_policy
        self.fallback_count = 0
        self.table_path = ""
        result = self._load_or_calibrate()
        self.tables = result.tables
        self.table_path = result.table_path
        self.metadata = result.metadata

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        orders = self._replenishment(env, obs)
        allocations = self._allocation(env, obs)
        return ControlAction(allocations=allocations, orders=orders)

    def _load_or_calibrate(self) -> H3CalibrationResult:
        self.table_dir.mkdir(parents=True, exist_ok=True)
        digest = _instance_digest(self.instance, {
            "z_max": self.z_max,
            "n_cal_paths": self.n_cal_paths,
            "calibration_horizon": self.calibration_horizon,
            "warmup": self.warmup,
            "seed": self.seed,
        })
        path = self.table_dir / f"h3_sbr_tables_{digest}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            print(f"[H3-SBR] loaded table from {path}")
            return H3CalibrationResult(payload["tables"], str(path), payload["metadata"])

        print(f"[H3-SBR] calibrating tables -> {path}")
        tables = _calibrate_h3_tables(
            self.instance,
            z_max=self.z_max,
            n_cal_paths=self.n_cal_paths,
            calibration_horizon=self.calibration_horizon,
            warmup=self.warmup,
            seed=self.seed,
        )
        metadata = {
            "seed": self.seed,
            "n_cal_paths": self.n_cal_paths,
            "calibration_horizon": self.calibration_horizon,
            "warmup": self.warmup,
            "z_max": self.z_max,
            "env_config_hash": digest,
        }
        with path.open("wb") as fh:
            pickle.dump({"tables": tables, "metadata": metadata}, fh)
        return H3CalibrationResult(tables, str(path), metadata)

    def _expected_unrevealed_load(self, env: ATOEnv, obs: Observation) -> np.ndarray:
        inst = env.instance
        load = np.zeros(inst.J, dtype=float)
        horizon = obs.t + inst.expected_lead_time + inst.delivery_window
        for p in range(inst.I):
            for s in range(obs.t + 1):
                if env.revealed[p, s] or env.remaining[p, s] <= 1e-9:
                    continue
                release = s + int(inst.design_lead_times[p])
                due = release + inst.delivery_window
                if release <= horizon or due <= horizon:
                    load += inst.template_bom[p] * float(env.remaining[p, s])
        return load

    def _z(self, obs: Observation, component: int, product: int, local_on_hand: np.ndarray | None = None) -> int:
        inst = self.instance
        comps = np.flatnonzero(inst.template_bom[product] > 1e-12)
        others = [int(j) for j in comps if int(j) != int(component)]
        if not others:
            return self.z_max
        availability = np.asarray(local_on_hand if local_on_hand is not None else obs.inventory + obs.outstanding, dtype=float)
        vals = []
        for other in others:
            denom = max(1.0, float(inst.template_bom[product, other]))
            vals.append(math.floor(max(0.0, availability[other]) / denom))
        return int(np.clip(min(vals) if vals else self.z_max, 0, self.z_max))

    def _replenishment(self, env: ATOEnv, obs: Observation) -> np.ndarray:
        inst = env.instance
        expected_unrevealed = self._expected_unrevealed_load(env, obs)
        ip = obs.inventory + obs.outstanding - expected_unrevealed
        targets = np.zeros(inst.J, dtype=float)
        for j in range(inst.J):
            product_targets = []
            for p in np.flatnonzero(inst.template_bom[:, j] > 1e-12):
                table = self.tables.get((int(j), int(p)))
                if table is None:
                    continue
                z = self._z(obs, int(j), int(p))
                product_targets.append(float(table.S[z]))
            if product_targets:
                targets[j] = min(product_targets)
            elif self.fallback_policy is not None and hasattr(self.fallback_policy, "base_stock"):
                targets[j] = float(self.fallback_policy.base_stock[j])
            else:
                targets[j] = 0.0
        known_requirement = env.remaining.sum(axis=1) @ inst.template_bom
        targets += 0.35 * known_requirement
        return np.maximum(0.0, np.ceil(targets - ip - 1e-9))

    def _allocation(self, env: ATOEnv, obs: Observation) -> List[Tuple[int, int, float]]:
        inst = env.instance
        local = obs.inventory.copy()
        allocations: List[Tuple[int, int, float]] = []
        orders = []
        for p, s, remaining, bom in obs.revealed:
            due = int(s + inst.design_lead_times[p] + inst.delivery_window)
            late = obs.t > due
            orders.append((not late, -float(inst.backlog_costs[p]), due, int(s), int(p), int(s), float(remaining), bom))
        orders.sort()
        for _not_late, _neg_penalty, _due, _arrival, p, s, remaining, bom in orders:
            qty = 0
            while qty + 1 <= int(math.floor(remaining + 1e-9)):
                feasible = True
                for j in np.flatnonzero(bom > 1e-12):
                    table = self.tables.get((int(j), int(p)))
                    z = self._z(obs, int(j), int(p), local_on_hand=local)
                    reserve = float(table.R[z]) if table is not None else 0.0
                    if local[j] - float(bom[j]) < reserve - 1e-9:
                        feasible = False
                        break
                if not feasible:
                    break
                local -= bom
                qty += 1
            if qty > 0:
                allocations.append((p, s, float(qty)))
        return allocations


def _target_tracking_allocation(
    env: ATOEnv,
    obs: Observation,
    desired_allocations: Dict[Tuple[int, int], float],
) -> List[Tuple[int, int, float]]:
    inst = env.instance
    local = obs.inventory.copy()
    candidates = []
    for p, s, remaining, bom in obs.revealed:
        desired = max(0.0, desired_allocations.get((int(p), int(s)), 0.0))
        target_backlog = max(0.0, float(remaining) - desired)
        excess = float(remaining) - target_backlog
        if excess <= 1e-9:
            continue
        due = int(s + inst.design_lead_times[p] + inst.delivery_window)
        slack = max(0, due - obs.t)
        priority = float(inst.backlog_costs[p]) / (1.0 + slack)
        if obs.t > due:
            priority += 10.0 * float(inst.backlog_costs[p])
        candidates.append((-priority, int(s), int(p), int(s), float(remaining), float(excess), bom))
    candidates.sort()
    allocations: List[Tuple[int, int, float]] = []
    for _priority, _arrival, p, s, remaining, excess, bom in candidates:
        cap = min(int(math.ceil(excess - 1e-9)), int(math.floor(remaining + 1e-9)))
        qty = 0
        while qty < cap:
            if any(local[j] - float(bom[j]) < -1e-9 for j in np.flatnonzero(bom > 1e-12)):
                break
            local -= bom
            qty += 1
        if qty > 0:
            allocations.append((p, s, float(qty)))
    return allocations


def _post_allocation_state(
    obs: Observation,
    allocations: List[Tuple[int, int, float]],
    remaining: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    local = np.asarray(obs.inventory, dtype=float).copy()
    remaining_after = np.asarray(remaining, dtype=float).copy()
    boms = {(int(i), int(s)): np.asarray(bom, dtype=float) for i, s, _rem, bom in obs.revealed}
    for i, s, qty in allocations:
        i = int(i)
        s = int(s)
        if s > int(obs.t) or remaining_after[i, s] <= 1e-9:
            continue
        bom = boms.get((i, s))
        if bom is None:
            continue
        positive = bom > 1e-12
        if not positive.any():
            continue
        feasible = float(np.min(local[positive] / bom[positive]))
        actual = max(0.0, min(float(qty), float(remaining_after[i, s]), math.floor(feasible + 1e-9)))
        if actual <= 1e-9:
            continue
        local -= bom * actual
        local = np.maximum(local, 0.0)
        remaining_after[i, s] -= actual
    return local, remaining_after


def _calibrate_h3_tables(
    inst: ProblemInstance,
    z_max: int,
    n_cal_paths: int,
    calibration_horizon: int,
    warmup: int,
    seed: int,
) -> Dict[Tuple[int, int], H3Table]:
    rng = np.random.default_rng(seed)
    nvd = np.maximum(0.0, np.asarray(NVDPolicy(inst).base_stock, dtype=float))
    tables: Dict[Tuple[int, int], H3Table] = {}
    for p in range(inst.I):
        for j in np.flatnonzero(inst.template_bom[p] > 1e-12):
            mean = float(inst.demand_lambdas[p] * inst.template_bom[p, j] * (inst.expected_lead_time + 1.0))
            s_max = int(max(4, min(40, math.ceil(mean + 3.0 * math.sqrt(max(1.0, mean)) + nvd[j]))))
            s_grid = sorted({0, int(0.5 * s_max), int(0.75 * s_max), int(s_max), int(1.25 * s_max)})
            s_grid = [int(np.clip(s, 0, s_max)) for s in s_grid]
            S = np.zeros(z_max + 1, dtype=int)
            R = np.zeros(z_max + 1, dtype=int)
            competing = [q for q in range(inst.I) if q != p and inst.template_bom[q, j] > 1e-12]
            comp_lambda = float(sum(inst.demand_lambdas[q] for q in competing))
            comp_penalty = float(np.mean(inst.backlog_costs[competing])) if competing else float(inst.backlog_costs[p])
            for z in range(z_max + 1):
                best_pair = (0, 0)
                best_cost = float("inf")
                for s_val in s_grid:
                    r_grid = sorted({0, int(0.25 * s_val), int(0.5 * s_val), int(0.75 * s_val)})
                    for r_val in r_grid:
                        cost = _simulate_h3_subsystem(
                            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
                            inst=inst,
                            product=p,
                            component=int(j),
                            z=z,
                            S=s_val,
                            R=r_val,
                            comp_lambda=comp_lambda,
                            comp_penalty=comp_penalty,
                            paths=n_cal_paths,
                            horizon=calibration_horizon,
                            warmup=warmup,
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_pair = (s_val, r_val)
                S[z], R[z] = best_pair
            for z in range(1, z_max + 1):
                S[z] = max(S[z], S[z - 1])
                R[z] = min(R[z], R[z - 1])
            tables[(int(j), int(p))] = H3Table(S=S, R=np.maximum(0, R), z_max=z_max)
            if p == 0 and len(tables) <= 3:
                print(f"[H3-SBR] calibrated edge=({int(j)},{int(p)}) S0={S[0]} Smax={S[-1]} R0={R[0]}")
    return tables


def _simulate_h3_subsystem(
    rng: np.random.Generator,
    inst: ProblemInstance,
    product: int,
    component: int,
    z: int,
    S: int,
    R: int,
    comp_lambda: float,
    comp_penalty: float,
    paths: int,
    horizon: int,
    warmup: int,
) -> float:
    costs = []
    bom_a = max(1.0, float(inst.template_bom[product, component]))
    other_need = max(1.0, float(np.sum(inst.template_bom[product]) - bom_a))
    for _ in range(paths):
        on_a = int(S)
        on_g = int(max(z, other_need))
        pipe_a: List[Tuple[int, int]] = []
        back_main = 0
        back_comp = 0
        total = 0.0
        counted = 0
        for t in range(horizon):
            arrived = [item for item in pipe_a if item[0] <= t]
            if arrived:
                on_a += sum(q for _at, q in arrived)
                pipe_a = [item for item in pipe_a if item[0] > t]
            back_main += int(rng.poisson(max(0.0, inst.demand_lambdas[product])))
            if comp_lambda > 1e-9:
                back_comp += int(rng.poisson(comp_lambda))
            ip = on_a + sum(q for _at, q in pipe_a)
            q_a = max(0, int(S - ip))
            if q_a > 0:
                lead = int(rng.integers(inst.min_replenishment_lead_time, inst.max_replenishment_lead_time + 1))
                pipe_a.append((t + lead, q_a))
            while back_main > 0 and on_a - bom_a >= R and on_g >= other_need:
                on_a -= int(math.ceil(bom_a))
                on_g -= int(math.ceil(other_need))
                back_main -= 1
            while back_comp > 0 and on_a - 1 >= R:
                on_a -= 1
                back_comp -= 1
            period_cost = (
                float(inst.holding_costs[component]) * on_a
                + float(inst.backlog_costs[product]) * back_main
                + comp_penalty * back_comp
                + float(inst.ordering_costs[component]) * q_a
            )
            if t >= warmup:
                total += period_cost
                counted += 1
        costs.append(total / max(1, counted))
    return float(np.mean(costs))


def _instance_digest(inst: ProblemInstance, cfg: Dict[str, Any]) -> str:
    data = {
        "I": inst.I,
        "J": inst.J,
        "T": inst.T,
        "template_bom": inst.template_bom.tolist(),
        "support": inst.support.astype(int).tolist(),
        "design_lead_times": inst.design_lead_times.tolist(),
        "delivery_window": int(inst.delivery_window),
        "demand_lambdas": inst.demand_lambdas.tolist(),
        "demand_pattern": inst.demand_pattern,
        "demand_correlation": float(inst.demand_correlation),
        "latent_demand_correlation": float(inst.latent_demand_correlation),
        "empirical_demand_correlation": float(inst.empirical_demand_correlation),
        "seasonal_beta": float(inst.seasonal_beta),
        "seasonal_cycle": int(inst.seasonal_cycle),
        "seasonal_phases": inst.seasonal_phases.tolist(),
        "bom_cv": float(inst.bom_cv),
        "realized_commonality": float(inst.realized_commonality),
        "holding_costs": inst.holding_costs.tolist(),
        "ordering_costs": inst.ordering_costs.tolist(),
        "backlog_costs": inst.backlog_costs.tolist(),
        "lead": [inst.min_replenishment_lead_time, inst.max_replenishment_lead_time],
        "initial_inventory": inst.initial_inventory.tolist(),
        "instance_config": inst.config,
        "cfg": cfg,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _episode_id(env: ATOEnv) -> int:
    if env.scenario is None:
        return 0
    return int(getattr(env.scenario, "episode", 0))
