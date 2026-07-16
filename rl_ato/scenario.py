from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .config import ExperimentConfig


@dataclass
class ProblemInstance:
    I: int
    J: int
    T: int
    template_bom: np.ndarray
    support: np.ndarray
    design_lead_times: np.ndarray
    delivery_window: int
    min_replenishment_lead_time: int
    max_replenishment_lead_time: int
    holding_costs: np.ndarray
    ordering_costs: np.ndarray
    backlog_costs: np.ndarray
    demand_lambdas: np.ndarray
    seasonal_phases: np.ndarray
    demand_pattern: str
    demand_correlation: float
    latent_demand_correlation: float
    empirical_demand_correlation: float
    seasonal_beta: float
    seasonal_cycle: int
    bom_cv: float
    realized_commonality: float
    target_common_components: np.ndarray
    history_window: int
    initial_inventory: np.ndarray
    feature_scale: float
    discount_factor: float
    seeds: Dict[str, int]
    config: Dict[str, Any]

    @property
    def edge_pairs(self) -> np.ndarray:
        return np.argwhere(self.support > 0)

    @property
    def expected_lead_time(self) -> float:
        return (float(self.min_replenishment_lead_time) + float(self.max_replenishment_lead_time)) / 2.0

    @property
    def complement_matrix(self) -> np.ndarray:
        comp = np.zeros((self.J, self.J), dtype=bool)
        for i in range(self.I):
            js = np.flatnonzero(self.support[i])
            for j in js:
                comp[j, js[js != j]] = True
        return comp


@dataclass
class Scenario:
    demand: np.ndarray
    realized_bom: np.ndarray
    lead_times: np.ndarray
    realized_bom_cv: float = np.nan
    episode: int = 0


def substream_seeds(master_seed: int) -> Dict[str, int]:
    return {
        "master": int(master_seed),
        "graph": int(master_seed + 101),
        "template_bom": int(master_seed + 201),
        "cost": int(master_seed + 251),
        "demand_rates": int(master_seed + 301),
        "correlation_calibration": int(master_seed + 351),
        "seasonal_phases": int(master_seed + 401),
        "demand_episode_base": int(master_seed + 501),
        "order_bom_episode_base": int(master_seed + 601),
        "lead_time_episode_base": int(master_seed + 701),
        "training_environment": int(master_seed + 801),
        "network_initialization": int(master_seed + 901),
    }


def make_instance(config: ExperimentConfig, seed: int | None = None) -> ProblemInstance:
    master_seed = int(config.seed if seed is None else seed)
    seeds = substream_seeds(master_seed)
    I, J, T = config.products, config.components, config.horizon

    support, common_components, realized_commonality = _generate_product_component_graph(
        I=I,
        J=J,
        rho=float(config.component_commonality),
        seed=seeds["graph"],
    )
    template_bom = _generate_template_bom(
        support,
        seeds["template_bom"],
        int(config.bom_min),
        int(config.bom_max),
    )
    demand_lambdas = _generate_demand_rates(
        rng=np.random.default_rng(seeds["demand_rates"]),
        n=I,
        mean=float(config.mean_demand),
        cv=float(config.demand_cv),
    )
    realized_demand_cv = float(
        demand_lambdas.std(ddof=0) / max(float(demand_lambdas.mean()), 1e-12)
    )
    seasonal_phases = np.random.default_rng(seeds["seasonal_phases"]).uniform(0.0, 2.0 * np.pi, size=I)
    latent_corr, empirical_corr = _calibrate_latent_correlation(
        target=float(config.demand_correlation),
        lambdas=demand_lambdas,
        beta=float(config.seasonal_beta),
        phases=seasonal_phases,
        cycle=int(config.seasonal_cycle),
        pattern=str(config.demand_pattern),
        pilot_samples=int(config.correlation_pilot_samples),
        tolerance=float(config.correlation_tolerance),
        seed=seeds["correlation_calibration"],
    )

    if config.random_costs:
        cost_rng = np.random.default_rng(seeds["cost"])
        holding_costs = cost_rng.uniform(config.holding_cost_low, config.holding_cost_high, size=J)
        ordering_costs = cost_rng.uniform(config.ordering_cost_low, config.ordering_cost_high, size=J)
        cost_rule = "uniform"
    else:
        holding_costs = np.full(J, config.holding_cost, dtype=float)
        ordering_costs = np.full(J, config.ordering_cost, dtype=float)
        cost_rule = "fixed"
    backlog_costs = np.zeros(I, dtype=float)
    for i in range(I):
        backlog_costs[i] = config.backorder_to_holding * float(np.dot(holding_costs, template_bom[i]))

    design_lead_times = np.full(I, config.design_lead_time, dtype=int)
    template_component_demand = template_bom.T @ demand_lambdas
    initial_inventory = np.ceil(
        _average_replenishment_lead_time(config) * template_component_demand
    )
    feature_scale = max(
        1.0,
        float(np.mean(demand_lambdas) * max(1, config.max_replenishment_lead_time) * max(1.0, template_bom.mean())),
    )
    return ProblemInstance(
        I=I,
        J=J,
        T=T,
        template_bom=template_bom,
        support=support,
        design_lead_times=design_lead_times,
        delivery_window=config.delivery_window,
        min_replenishment_lead_time=config.min_replenishment_lead_time,
        max_replenishment_lead_time=config.max_replenishment_lead_time,
        holding_costs=holding_costs,
        ordering_costs=ordering_costs,
        backlog_costs=backlog_costs,
        demand_lambdas=demand_lambdas,
        seasonal_phases=seasonal_phases,
        demand_pattern=config.demand_pattern,
        demand_correlation=config.demand_correlation,
        latent_demand_correlation=latent_corr,
        empirical_demand_correlation=empirical_corr,
        seasonal_beta=config.seasonal_beta,
        seasonal_cycle=config.seasonal_cycle,
        bom_cv=config.bom_cv,
        realized_commonality=realized_commonality,
        target_common_components=common_components,
        history_window=config.history_window,
        initial_inventory=initial_inventory,
        feature_scale=feature_scale,
        discount_factor=float(config.discount_factor),
        seeds=seeds,
        config={
            "component_commonality": float(config.component_commonality),
            "realized_commonality": float(realized_commonality),
            "product_degrees": support.sum(axis=1).astype(int).tolist(),
            "active_components": int(np.sum(support.any(axis=0))),
            "bom_cv": float(config.bom_cv),
            "backorder_to_holding": float(config.backorder_to_holding),
            "mean_demand": float(config.mean_demand),
            "demand_cv": float(config.demand_cv),
            "realized_demand_cv": realized_demand_cv,
            "min_replenishment_lead_time": float(config.min_replenishment_lead_time),
            "max_replenishment_lead_time": float(config.max_replenishment_lead_time),
            "discount_factor": float(config.discount_factor),
            "latent_demand_correlation": float(latent_corr),
            "empirical_demand_correlation": float(empirical_corr),
            "cost_rule": cost_rule,
            "initial_inventory_rule": "expected replenishment lead-time demand",
            "feature_scale": float(feature_scale),
        },
    )


class ScenarioGenerator:
    def __init__(self, instance: ProblemInstance, seed: int | None = None, start_episode: int = 0):
        self.instance = instance
        self.seed_base = int(instance.seeds["master"] if seed is None else seed)
        self.episode_counter = int(start_episode)

    def sample(self, episode: int | None = None) -> Scenario:
        ep = self.episode_counter if episode is None else int(episode)
        if episode is None:
            self.episode_counter += 1
        inst = self.instance
        means = _period_demand_means(inst)
        demand_rng = np.random.default_rng(self.seed_base + 501 + ep)
        bom_rng = np.random.default_rng(self.seed_base + 601 + ep)
        lead_rng = np.random.default_rng(self.seed_base + 701 + ep)
        demand = _sample_demand_path(inst, means, demand_rng)
        bom, realized_cv = _sample_order_specific_bom(inst, bom_rng)
        lead_times = lead_rng.integers(
            low=inst.min_replenishment_lead_time,
            high=inst.max_replenishment_lead_time + 1,
            size=(inst.T, inst.J),
        )
        return Scenario(
            demand=demand.astype(float),
            realized_bom=bom.astype(float),
            lead_times=lead_times.astype(int),
            realized_bom_cv=realized_cv,
            episode=ep,
        )


def _average_replenishment_lead_time(config: ExperimentConfig) -> float:
    return (float(config.min_replenishment_lead_time) + float(config.max_replenishment_lead_time)) / 2.0


def _generate_product_component_graph(
    I: int,
    J: int,
    rho: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if I < 1 or J < 1:
        raise ValueError("products and components must be positive")
    rng = np.random.default_rng(seed)
    jc = min(J, max(0, int(round(float(rho) * J)))) if I > 1 else 0
    common_components = rng.choice(np.arange(J), size=jc, replace=False)
    common_set = set(int(value) for value in common_components)
    specific_components = np.asarray(
        [component for component in range(J) if component not in common_set],
        dtype=int,
    )
    support = np.zeros((I, J), dtype=bool)
    total_edges = J + jc
    target_degrees = np.full(I, total_edges // I, dtype=int)
    remainder = total_edges % I
    if remainder:
        target_degrees[rng.choice(np.arange(I), size=remainder, replace=False)] += 1
    remaining_degrees = target_degrees.copy()
    for component in rng.permutation(common_components):
        candidates = np.flatnonzero(remaining_degrees > 0)
        if candidates.size < 2:
            raise RuntimeError("product degree sequence cannot support common components")
        ranking = remaining_degrees[candidates] + rng.random(candidates.size)
        users = candidates[np.argsort(ranking)[-2:]]
        support[users, int(component)] = True
        remaining_degrees[users] -= 1
    for component in rng.permutation(specific_components):
        candidates = np.flatnonzero(remaining_degrees > 0)
        if candidates.size == 0:
            raise RuntimeError("product degree sequence is exhausted")
        ranking = remaining_degrees[candidates] + rng.random(candidates.size)
        user = int(candidates[int(np.argmax(ranking))])
        support[user, int(component)] = True
        remaining_degrees[user] -= 1
    if np.any(remaining_degrees != 0):
        raise RuntimeError("product degree sequence was not satisfied")
    realized = _realized_commonality(support)
    return support, np.sort(common_components), realized


def _realized_commonality(support: np.ndarray) -> float:
    degrees = support.sum(axis=0)
    return float(np.mean(degrees >= 2))


def _generate_template_bom(
    support: np.ndarray,
    seed: int,
    coefficient_min: int,
    coefficient_max: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bom = np.zeros(support.shape, dtype=float)
    low = max(1, int(coefficient_min))
    high = max(low, int(coefficient_max))
    bom[support] = rng.integers(low, high + 1, size=int(support.sum()))
    return bom


def _generate_demand_rates(
    rng: np.random.Generator,
    n: int,
    mean: float,
    cv: float,
) -> np.ndarray:
    if cv <= 0.0:
        return np.full(n, mean, dtype=float)
    sigma2 = float(np.log1p(float(cv) ** 2))
    z = rng.lognormal(mean=-0.5 * sigma2, sigma=np.sqrt(sigma2), size=n)
    return float(mean) * z / max(float(z.mean()), 1e-12)


def _period_demand_means(inst: ProblemInstance) -> np.ndarray:
    means = np.zeros((inst.I, inst.T), dtype=float)
    for t in range(inst.T):
        if inst.demand_pattern.lower().startswith("season"):
            wave = 1.0 + inst.seasonal_beta * np.sin(
                2.0 * np.pi * t / max(1, inst.seasonal_cycle) + inst.seasonal_phases
            )
            means[:, t] = np.maximum(1e-9, inst.demand_lambdas * wave)
        else:
            means[:, t] = inst.demand_lambdas
    return means


def _sample_order_specific_bom(inst: ProblemInstance, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    bom = np.zeros((inst.I, inst.T, inst.J), dtype=float)
    ratios = []
    sigma2_log = np.log1p(float(inst.bom_cv) ** 2)
    sigma_log = np.sqrt(sigma2_log)
    for i in range(inst.I):
        active = np.flatnonzero(inst.support[i])
        for s in range(inst.T):
            for j in active:
                g0 = inst.template_bom[i, j]
                mu_log = np.log(g0) - 0.5 * sigma2_log
                raw = rng.lognormal(mean=mu_log, sigma=sigma_log)
                gij = max(1.0, float(np.rint(raw)))
                bom[i, s, j] = gij
                ratios.append(gij / g0)
    ratio_arr = np.asarray(ratios, dtype=float)
    realized_cv = float(ratio_arr.std(ddof=0) / max(ratio_arr.mean(), 1e-12)) if len(ratio_arr) else 0.0
    return bom, realized_cv


def _sample_demand_path(inst: ProblemInstance, means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    target = float(np.clip(inst.demand_correlation, 0.0, 0.99))
    if target <= 1e-9 or inst.I == 1:
        return rng.poisson(means)
    from scipy.stats import norm, poisson

    corr = np.full((inst.I, inst.I), float(np.clip(inst.latent_demand_correlation, 0.0, 0.999)), dtype=float)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    out = np.zeros_like(means)
    for t in range(inst.T):
        z = chol @ rng.normal(size=inst.I)
        u = np.clip(norm.cdf(z), 1e-12, 1.0 - 1e-12)
        out[:, t] = poisson.ppf(u, means[:, t])
    return out


def _calibrate_latent_correlation(
    target: float,
    lambdas: np.ndarray,
    beta: float,
    phases: np.ndarray,
    cycle: int,
    pattern: str,
    pilot_samples: int,
    tolerance: float,
    seed: int,
) -> tuple[float, float]:
    target = float(np.clip(target, 0.0, 0.99))
    if target <= 1e-12 or len(lambdas) <= 1:
        return 0.0, 0.0
    from scipy.stats import norm, poisson

    pilot_samples = max(1000, int(pilot_samples))
    rng_normals = np.random.default_rng(seed)
    normals = rng_normals.normal(size=(pilot_samples, len(lambdas)))
    periods = np.arange(pilot_samples) % max(1, cycle)
    if pattern.lower().startswith("season"):
        means = np.vstack(
            [
                np.maximum(
                    1e-9,
                    lambdas
                    * (1.0 + beta * np.sin(2.0 * np.pi * int(t) / max(1, cycle) + phases)),
                )
                for t in periods
            ]
        )
    else:
        means = np.broadcast_to(lambdas, (pilot_samples, len(lambdas))).copy()

    def count_corr(latent: float) -> float:
        corr = np.full((len(lambdas), len(lambdas)), latent, dtype=float)
        np.fill_diagonal(corr, 1.0)
        chol = np.linalg.cholesky(corr)
        z = normals @ chol.T
        u = np.clip(norm.cdf(z), 1e-12, 1.0 - 1e-12)
        counts = poisson.ppf(u, means)
        cm = np.corrcoef(counts, rowvar=False)
        vals = cm[np.triu_indices(len(lambdas), k=1)]
        empirical = float(np.nanmean(vals))
        return 0.0 if not np.isfinite(empirical) else empirical

    lo, hi = 0.0, 0.999
    best_latent, best_empirical = 0.0, 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        empirical = count_corr(mid)
        if abs(empirical - target) < abs(best_empirical - target):
            best_latent, best_empirical = mid, empirical
        if abs(empirical - target) <= tolerance:
            return mid, empirical
        if empirical < target:
            lo = mid
        else:
            hi = mid
    return best_latent, best_empirical
