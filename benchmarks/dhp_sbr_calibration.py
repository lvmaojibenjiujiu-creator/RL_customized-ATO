from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from rl_ato.scenario import ProblemInstance


@dataclass
class DHPSBRCalibrationConfig:
    calibration_mode: str = "saa_grid"
    fallback_calibration_mode: str = "formula_init"
    edge_prob_threshold: float = 0.05
    Z_max: int = 20
    n_cal_paths_small: int = 200
    n_cal_paths_large: int = 80
    T_cal: int = 200
    warmup: int = 50
    safety_factor: float = 0.0
    max_S_candidate_multiplier: float = 1.5
    reserve_factor: float = 0.5
    state_sensitivity: float = 0.0
    restrict_R_leq_S: bool = True
    candidate_step_small: int = 1
    candidate_step_large: int = 2
    max_grid_points: int = 9
    parallel: bool = False
    cache_tables: bool = True
    cache_dir: str = "outputs/dhp_sbr_calibration"
    calibration_time_limit_seconds: float = 3600.0

    @classmethod
    def from_mapping(cls, values: Dict[str, Any] | None) -> "DHPSBRCalibrationConfig":
        cfg = cls()
        if not values:
            return cfg
        allowed = set(asdict(cfg))
        for key, value in values.items():
            if key in allowed:
                setattr(cfg, key, value)
        cfg.calibration_mode = str(cfg.calibration_mode)
        cfg.fallback_calibration_mode = str(cfg.fallback_calibration_mode)
        cfg.edge_prob_threshold = float(cfg.edge_prob_threshold)
        cfg.Z_max = int(max(1, cfg.Z_max))
        cfg.n_cal_paths_small = int(max(1, cfg.n_cal_paths_small))
        cfg.n_cal_paths_large = int(max(1, cfg.n_cal_paths_large))
        cfg.T_cal = int(max(5, cfg.T_cal))
        cfg.warmup = int(max(0, min(cfg.warmup, cfg.T_cal - 1)))
        cfg.safety_factor = float(max(0.0, cfg.safety_factor))
        cfg.max_S_candidate_multiplier = float(max(0.1, cfg.max_S_candidate_multiplier))
        cfg.reserve_factor = float(max(0.0, cfg.reserve_factor))
        cfg.state_sensitivity = float(np.clip(cfg.state_sensitivity, 0.0, 1.0))
        cfg.candidate_step_small = int(max(1, cfg.candidate_step_small))
        cfg.candidate_step_large = int(max(1, cfg.candidate_step_large))
        cfg.max_grid_points = int(max(3, cfg.max_grid_points))
        cfg.calibration_time_limit_seconds = float(max(0.0, cfg.calibration_time_limit_seconds))
        cfg.cache_dir = str(cfg.cache_dir)
        return cfg


@dataclass
class DHPCalibrationResult:
    S_table: Dict[Tuple[int, int], np.ndarray]
    R_table: Dict[Tuple[int, int], np.ndarray]
    components_of_product: Dict[int, Tuple[int, ...]]
    products_using_component: Dict[int, Tuple[int, ...]]
    metadata: Dict[str, Any]
    table_path: str = ""


def calibrate_dhp_sbr_tables(
    inst: ProblemInstance,
    calibration_config: DHPSBRCalibrationConfig | Dict[str, Any] | None = None,
    seed: int = 0,
) -> DHPCalibrationResult:
    cfg = (
        calibration_config
        if isinstance(calibration_config, DHPSBRCalibrationConfig)
        else DHPSBRCalibrationConfig.from_mapping(calibration_config)
    )
    cfg_dict = asdict(cfg)
    mean_bom = np.asarray(inst.template_bom, dtype=float)
    prob_bom = np.asarray(inst.support, dtype=float)
    edges, components_of_product, products_using_component = build_product_component_edges(
        mean_bom=mean_bom,
        prob_bom=prob_bom,
        edge_prob_threshold=cfg.edge_prob_threshold,
    )
    digest = _instance_digest(inst, cfg_dict, mean_bom, prob_bom)
    cache_dir = Path(cfg.cache_dir)
    path = cache_dir / f"dhp_sbr_{digest}_{cfg.calibration_mode}_{int(seed)}.pkl"
    if cfg.cache_tables and path.exists():
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return DHPCalibrationResult(
            S_table={tuple(k): np.asarray(v, dtype=int) for k, v in payload["S_table"].items()},
            R_table={tuple(k): np.asarray(v, dtype=int) for k, v in payload["R_table"].items()},
            components_of_product={int(k): tuple(v) for k, v in payload["components_of_product"].items()},
            products_using_component={int(k): tuple(v) for k, v in payload["products_using_component"].items()},
            metadata=dict(payload["metadata"]),
            table_path=str(path),
        )

    started = time.perf_counter()
    formula_S, formula_R = _formula_init_tables(inst, cfg, edges, products_using_component)
    S_table = {edge: arr.copy() for edge, arr in formula_S.items()}
    R_table = {edge: arr.copy() for edge, arr in formula_R.items()}
    fallback_edges: list[tuple[int, int]] = []
    calibrated_edges: list[tuple[int, int]] = []
    mode = cfg.calibration_mode.lower()
    n_cal_paths = _calibration_path_count(inst, cfg)

    if mode == "saa_grid":
        rng = np.random.default_rng(int(seed))
        for edge in edges:
            if _time_exceeded(started, cfg.calibration_time_limit_seconds):
                fallback_edges.extend([e for e in edges if e not in calibrated_edges and e not in fallback_edges])
                break
            try:
                S, R = _calibrate_edge_saa_grid(
                    inst=inst,
                    cfg=cfg,
                    edge=edge,
                    components_of_product=components_of_product,
                    products_using_component=products_using_component,
                    n_cal_paths=n_cal_paths,
                    rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
                    started=started,
                )
                S_table[edge] = S
                R_table[edge] = R
                calibrated_edges.append(edge)
            except Exception:
                fallback_edges.append(edge)
    elif mode != "formula_init":
        fallback_edges = list(edges)
        mode = "formula_init"

    for edge in edges:
        S_table[edge], R_table[edge] = _repair_monotonicity(S_table[edge], R_table[edge])

    runtime = time.perf_counter() - started
    metadata: Dict[str, Any] = {
        "policy": "DHP-SBR",
        "calibration_mode": cfg.calibration_mode,
        "effective_calibration_mode": mode,
        "fallback_calibration_mode": cfg.fallback_calibration_mode,
        "seed": int(seed),
        "edge_prob_threshold": float(cfg.edge_prob_threshold),
        "Z_max": int(cfg.Z_max),
        "n_edges": int(len(edges)),
        "n_cal_paths": int(n_cal_paths if mode == "saa_grid" else 0),
        "T_cal": int(cfg.T_cal),
        "fallback_edges": [(int(j), int(p)) for j, p in fallback_edges],
        "calibrated_edges": [(int(j), int(p)) for j, p in calibrated_edges],
        "runtime_seconds": float(runtime),
        "instance_hash": digest,
        "config": cfg_dict,
    }

    result = DHPCalibrationResult(
        S_table=S_table,
        R_table=R_table,
        components_of_product=components_of_product,
        products_using_component=products_using_component,
        metadata=metadata,
        table_path=str(path) if cfg.cache_tables else "",
    )
    if cfg.cache_tables:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "S_table": result.S_table,
                    "R_table": result.R_table,
                    "components_of_product": result.components_of_product,
                    "products_using_component": result.products_using_component,
                    "metadata": result.metadata,
                },
                fh,
            )
    return result


def build_product_component_edges(
    mean_bom: np.ndarray,
    prob_bom: np.ndarray,
    edge_prob_threshold: float = 0.05,
) -> tuple[list[Tuple[int, int]], Dict[int, Tuple[int, ...]], Dict[int, Tuple[int, ...]]]:
    I, J = mean_bom.shape
    edge_set: set[Tuple[int, int]] = set()
    for p in range(I):
        for j in range(J):
            if float(mean_bom[p, j]) > 1e-12 or float(prob_bom[p, j]) >= float(edge_prob_threshold):
                edge_set.add((int(j), int(p)))
        if not any(p0 == p for _j, p0 in edge_set):
            for j in np.flatnonzero(mean_bom[p] > 1e-12):
                edge_set.add((int(j), int(p)))
    edges = sorted(edge_set)
    components_of_product: Dict[int, Tuple[int, ...]] = {}
    products_using_component: Dict[int, Tuple[int, ...]] = {}
    for p in range(I):
        components_of_product[p] = tuple(sorted(j for j, p0 in edges if p0 == p))
    for j in range(J):
        products_using_component[j] = tuple(sorted(p for j0, p in edges if j0 == j))
    return edges, components_of_product, products_using_component


def _formula_init_tables(
    inst: ProblemInstance,
    cfg: DHPSBRCalibrationConfig,
    edges: Iterable[Tuple[int, int]],
    products_using_component: Dict[int, Tuple[int, ...]],
) -> tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    mu, std = _leadtime_load_moments(inst)
    base_S = np.maximum(0.0, np.ceil(mu + cfg.safety_factor * std))
    S_table: Dict[Tuple[int, int], np.ndarray] = {}
    R_table: Dict[Tuple[int, int], np.ndarray] = {}
    for j, p in edges:
        competing_pressure = 0.0
        for q in products_using_component.get(int(j), ()):
            if int(q) == int(p):
                continue
            competing_pressure += (
                float(inst.demand_lambdas[q])
                * float(inst.template_bom[q, j])
                * max(0.0, float(inst.backlog_costs[q] - inst.backlog_costs[p]))
            )
        reserve_base = int(
            math.ceil(
                cfg.reserve_factor
                * competing_pressure
                * inst.expected_lead_time
                / max(1.0, float(inst.backlog_costs[p]))
            )
        )
        S = np.zeros(cfg.Z_max + 1, dtype=int)
        R = np.zeros(cfg.Z_max + 1, dtype=int)
        for z in range(cfg.Z_max + 1):
            ratio = z / max(1, cfg.Z_max)
            S[z] = int(round(base_S[j] * (1.0 - cfg.state_sensitivity + 2.0 * cfg.state_sensitivity * ratio)))
            R[z] = int(round(reserve_base * (1.0 - ratio)))
        S_table[(int(j), int(p))], R_table[(int(j), int(p))] = _repair_monotonicity(S, R)
    return S_table, R_table


def _calibrate_edge_saa_grid(
    inst: ProblemInstance,
    cfg: DHPSBRCalibrationConfig,
    edge: Tuple[int, int],
    components_of_product: Dict[int, Tuple[int, ...]],
    products_using_component: Dict[int, Tuple[int, ...]],
    n_cal_paths: int,
    rng: np.random.Generator,
    started: float,
) -> tuple[np.ndarray, np.ndarray]:
    j, p = int(edge[0]), int(edge[1])
    mu, std = _leadtime_load_moments(inst)
    base_upper = int(math.ceil(mu[j] + cfg.safety_factor * max(1.0, std[j])))
    s_upper = max(2, int(math.ceil(base_upper * cfg.max_S_candidate_multiplier)))
    step = cfg.candidate_step_small if inst.I * inst.J <= 100 else cfg.candidate_step_large
    S_candidates = _candidate_values(0, s_upper, step, cfg.max_grid_points)
    R_candidates_all = _candidate_values(0, s_upper, step, cfg.max_grid_points)
    S = np.zeros(cfg.Z_max + 1, dtype=int)
    R = np.zeros(cfg.Z_max + 1, dtype=int)
    others = tuple(k for k in components_of_product.get(p, ()) if int(k) != j)
    comp_products = tuple(q for q in products_using_component.get(j, ()) if int(q) != p)
    comp_lambda = float(
        sum(float(inst.demand_lambdas[q]) * (1.0 if inst.support[q, j] else 0.0) for q in comp_products)
    )
    if comp_products:
        weights = np.asarray([inst.demand_lambdas[q] for q in comp_products], dtype=float)
        weights = weights / max(1e-12, float(weights.sum()))
        comp_consume = float(sum(weights[idx] * max(1.0, inst.template_bom[q, j]) for idx, q in enumerate(comp_products)))
        comp_penalty = float(sum(weights[idx] * inst.backlog_costs[q] for idx, q in enumerate(comp_products)))
    else:
        comp_consume = 1.0
        comp_penalty = float(inst.backlog_costs[p])

    for z in range(cfg.Z_max + 1):
        if _time_exceeded(started, cfg.calibration_time_limit_seconds):
            raise TimeoutError("DHP-SBR calibration time budget exceeded")
        streams = _make_crn_streams(
            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
            inst=inst,
            product=p,
            comp_lambda=comp_lambda,
            n_paths=n_cal_paths,
            horizon=cfg.T_cal,
        )
        best_cost = float("inf")
        best_pair = (int(S_candidates[0]), 0)
        for s_val in S_candidates:
            r_candidates = [r for r in R_candidates_all if (not cfg.restrict_R_leq_S or r <= s_val)]
            for r_val in r_candidates:
                cost = _simulate_dhp_subsystem(
                    inst=inst,
                    product=p,
                    component=j,
                    others=others,
                    z=int(z),
                    S=int(s_val),
                    R=int(r_val),
                    comp_consume=comp_consume,
                    comp_penalty=comp_penalty,
                    streams=streams,
                    warmup=cfg.warmup,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (int(s_val), int(r_val))
        S[z], R[z] = best_pair
    return _repair_monotonicity(S, R)


def _simulate_dhp_subsystem(
    inst: ProblemInstance,
    product: int,
    component: int,
    others: Tuple[int, ...],
    z: int,
    S: int,
    R: int,
    comp_consume: float,
    comp_penalty: float,
    streams: Dict[str, np.ndarray],
    warmup: int,
) -> float:
    bom_a = max(1.0, float(inst.template_bom[product, component]))
    if others:
        a_g = max(1.0, float(max(inst.template_bom[product, k] for k in others)))
        holding_g = float(np.mean(inst.holding_costs[list(others)]))
        ordering_g = float(np.mean(inst.ordering_costs[list(others)]))
    else:
        a_g = 0.0
        holding_g = 0.0
        ordering_g = 0.0
    costs = []
    for path in range(streams["main"].shape[0]):
        on_a = int(max(S, 0))
        on_g = int(max(z, math.ceil(a_g)))
        pipe_a: list[tuple[int, int]] = []
        pipe_g: list[tuple[int, int]] = []
        back_main = 0
        back_comp = 0
        total = 0.0
        counted = 0
        for t in range(streams["main"].shape[1]):
            if pipe_a:
                arrived = [q for at, q in pipe_a if at <= t]
                if arrived:
                    on_a += int(sum(arrived))
                    pipe_a = [(at, q) for at, q in pipe_a if at > t]
            if pipe_g:
                arrived_g = [q for at, q in pipe_g if at <= t]
                if arrived_g:
                    on_g += int(sum(arrived_g))
                    pipe_g = [(at, q) for at, q in pipe_g if at > t]
            back_main += int(streams["main"][path, t])
            back_comp += int(streams["comp"][path, t])
            ip_a = on_a + sum(q for _at, q in pipe_a)
            q_a = max(0, int(math.ceil(S - ip_a)))
            if q_a > 0:
                pipe_a.append((t + int(streams["lead_a"][path, t]), q_a))
            q_g = 0
            if others:
                ip_g = on_g + sum(q for _at, q in pipe_g)
                q_g = max(0, int(math.ceil(z - ip_g)))
                if q_g > 0:
                    pipe_g.append((t + int(streams["lead_g"][path, t]), q_g))
            while back_main > 0 and on_a - bom_a >= R and (not others or on_g >= a_g):
                on_a -= int(math.ceil(bom_a))
                if others:
                    on_g -= int(math.ceil(a_g))
                back_main -= 1
            while back_comp > 0 and on_a >= comp_consume:
                if float(comp_penalty) < float(inst.backlog_costs[product]) and on_a - comp_consume < R:
                    break
                on_a -= int(math.ceil(comp_consume))
                back_comp -= 1
            if t >= warmup:
                total += (
                    float(inst.ordering_costs[component]) * q_a
                    + ordering_g * q_g
                    + float(inst.holding_costs[component]) * max(0, on_a)
                    + holding_g * max(0, on_g)
                    + float(inst.backlog_costs[product]) * back_main
                    + float(comp_penalty) * back_comp
                )
                counted += 1
        costs.append(total / max(1, counted))
    return float(np.mean(costs))


def _make_crn_streams(
    rng: np.random.Generator,
    inst: ProblemInstance,
    product: int,
    comp_lambda: float,
    n_paths: int,
    horizon: int,
) -> Dict[str, np.ndarray]:
    main_lambda = max(0.0, float(inst.demand_lambdas[product]))
    return {
        "main": rng.poisson(main_lambda, size=(n_paths, horizon)),
        "comp": rng.poisson(max(0.0, comp_lambda), size=(n_paths, horizon)),
        "lead_a": rng.integers(
            inst.min_replenishment_lead_time,
            inst.max_replenishment_lead_time + 1,
            size=(n_paths, horizon),
        ),
        "lead_g": rng.integers(
            inst.min_replenishment_lead_time,
            inst.max_replenishment_lead_time + 1,
            size=(n_paths, horizon),
        ),
    }


def _leadtime_load_moments(inst: ProblemInstance) -> tuple[np.ndarray, np.ndarray]:
    mean_lead = inst.expected_lead_time
    mu = np.zeros(inst.J, dtype=float)
    var = np.zeros(inst.J, dtype=float)
    for j in range(inst.J):
        for p in range(inst.I):
            if not inst.support[p, j]:
                continue
            mean = float(inst.template_bom[p, j])
            bom_var = (float(inst.bom_cv) * mean) ** 2
            lam = float(inst.demand_lambdas[p])
            mu[j] += lam * mean * mean_lead
            var[j] += lam * (mean * mean + bom_var) * mean_lead
    return mu, np.sqrt(np.maximum(var, 0.0))


def _candidate_values(low: int, high: int, step: int, max_points: int) -> list[int]:
    vals = list(range(int(low), int(high) + 1, max(1, int(step))))
    if not vals or vals[-1] != int(high):
        vals.append(int(high))
    vals = sorted(set(max(0, int(v)) for v in vals))
    if len(vals) <= max_points:
        return vals
    idx = np.linspace(0, len(vals) - 1, num=max_points)
    return sorted(set(vals[int(round(i))] for i in idx))


def _repair_monotonicity(S: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    S = np.maximum(0, np.rint(np.asarray(S, dtype=float))).astype(int)
    R = np.maximum(0, np.rint(np.asarray(R, dtype=float))).astype(int)
    for z in range(1, len(S)):
        S[z] = max(S[z], S[z - 1])
        R[z] = min(R[z], R[z - 1])
    return S, R


def _calibration_path_count(inst: ProblemInstance, cfg: DHPSBRCalibrationConfig) -> int:
    return int(cfg.n_cal_paths_small if inst.I * inst.J <= 100 else cfg.n_cal_paths_large)


def _time_exceeded(started: float, budget: float) -> bool:
    return bool(budget > 0.0 and time.perf_counter() - started >= budget)


def _instance_digest(
    inst: ProblemInstance,
    cfg: Dict[str, Any],
    mean_bom: np.ndarray,
    prob_bom: np.ndarray,
) -> str:
    data = {
        "product_count": int(inst.I),
        "component_count": int(inst.J),
        "horizon": int(inst.T),
        "mean_bom": mean_bom.tolist(),
        "prob_bom": prob_bom.tolist(),
        "demand_params": {
            "demand_lambdas": inst.demand_lambdas.tolist(),
            "demand_pattern": inst.demand_pattern,
            "demand_correlation": float(inst.demand_correlation),
            "seasonal_beta": float(inst.seasonal_beta),
            "seasonal_cycle": int(inst.seasonal_cycle),
            "seasonal_phases": inst.seasonal_phases.tolist(),
        },
        "lead_time_params": {
            "min_replenishment_lead_time": int(inst.min_replenishment_lead_time),
            "max_replenishment_lead_time": int(inst.max_replenishment_lead_time),
            "design_lead_times": inst.design_lead_times.tolist(),
            "delivery_window": int(inst.delivery_window),
        },
        "cost_params": {
            "holding_costs": inst.holding_costs.tolist(),
            "ordering_costs": inst.ordering_costs.tolist(),
            "backlog_costs": inst.backlog_costs.tolist(),
        },
        "bom_cv": float(inst.bom_cv),
        "Z_max": int(cfg["Z_max"]),
        "calibration_mode": str(cfg["calibration_mode"]),
        "cfg": cfg,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
