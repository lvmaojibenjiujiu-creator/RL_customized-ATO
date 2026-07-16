from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


@dataclass
class ExperimentConfig:
    seed: int = 7
    products: int = 5
    components: int = 15
    horizon: int = 40
    demand_pattern: str = "poisson"
    mean_demand: float = 3.0
    demand_cv: float = 0.3
    demand_correlation: float = 0.0
    seasonal_beta: float = 0.3
    seasonal_cycle: int = 6
    design_lead_time: int = 3
    delivery_window: int = 3
    min_replenishment_lead_time: int = 1
    max_replenishment_lead_time: int = 6
    component_commonality: float = 0.3
    bom_min: int = 1
    bom_max: int = 3
    bom_cv: float = 0.2
    random_costs: bool = True
    holding_cost: float = 1.0
    ordering_cost: float = 1.0
    holding_cost_low: float = 0.8
    holding_cost_high: float = 1.2
    ordering_cost_low: float = 0.8
    ordering_cost_high: float = 1.2
    backorder_to_holding: float = 2.0
    correlation_pilot_samples: int = 8000
    correlation_tolerance: float = 1e-3
    history_window: int = 8
    discount_factor: float = 0.99
    train_episodes: int = 1000
    evaluation_episodes: int = 100
    pi_episodes: int = 100
    holdout_episode_offset: int = 2_000_000
    comparison_scales: Iterable[Iterable[int]] = field(
        default_factory=lambda: [[5, 15], [10, 20], [20, 100]]
    )
    comparison_demand_patterns: Iterable[str] = field(
        default_factory=lambda: ["poisson", "seasonal"]
    )

    @property
    def scale(self) -> Tuple[int, int]:
        return self.products, self.components


@dataclass
class PPOConfig:
    network_seed: int = -1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_start: float = 0.2
    clip_end: float = 0.1
    entropy_start: float = 1e-2
    entropy_end: float = 0.0
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    batch_size: int = 256
    epochs: int = 10
    rollout_episodes: int = 8
    max_grad_norm: float = 0.7
    gat_layers: int = 2
    gat_hidden_width: int = 128
    attention_heads: int = 4
    gru_hidden_size: int = 64
    context_dim: int = 64
    eta_max: float = 1.0
    alpha_mon_start: float = 1e-4
    alpha_lip_start: float = 1e-4
    alpha_rat_start: float = 1e-4
    alpha_mon_end: float = 1e-2
    alpha_lip_end: float = 1e-2
    alpha_rat_end: float = 1e-2
    device: str = "cpu"


@dataclass
class BenchmarkConfig:
    saa_training_paths: int = 20
    saa_calibration_episode_offset: int = 1_500_000
    saa_step_sizes: Iterable[int] = field(default_factory=lambda: [8, 4, 2, 1])
    saa_beta_late: float = 1.0
    saa_upper_quantile: float = 0.99
    dhp_z_max: int = 10
    dhp_calibration_paths: int = 200
    dhp_calibration_episode_offset: int = 1_000_000
    rh_horizon: int = 8
    rh_scenarios: int = 15
    rh_terminal_backlog_weight: float = 1.0
    rh_terminal_inventory_weight: float = 0.0
    rh_mip_gap: float = 0.0
    rh_time_limit: float = 0.0
    rh_threads: int = 0
    rh_seed_offset: int = 2_500_000


@dataclass
class SensitivityConfig:
    bom_cv: Iterable[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    component_commonality: Iterable[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7, 0.8])
    delivery_window: Iterable[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    max_replenishment_lead_time: Iterable[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    design_lead_time: Iterable[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    demand_cv: Iterable[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
    demand_correlation: Iterable[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
    seasonal_beta: Iterable[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    backorder_to_holding: Iterable[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])


def _update_dataclass(obj: Any, values: Dict[str, Any]) -> Any:
    allowed = {item.name for item in fields(obj)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown configuration fields for {type(obj).__name__}: {', '.join(unknown)}")
    for key, value in values.items():
        setattr(obj, key, value)
    return obj


def load_config(
    path: str | Path | None = None,
) -> tuple[ExperimentConfig, PPOConfig, BenchmarkConfig, SensitivityConfig]:
    exp = ExperimentConfig()
    ppo = PPOConfig()
    bench = BenchmarkConfig()
    sens = SensitivityConfig()
    if path is None:
        return exp, ppo, bench, sens
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed_sections = {"experiment", "ppo", "benchmarks", "sensitivity"}
    unknown_sections = sorted(set(data) - allowed_sections)
    if unknown_sections:
        raise ValueError(f"Unknown configuration sections: {', '.join(unknown_sections)}")
    _update_dataclass(exp, data.get("experiment", {}))
    _update_dataclass(ppo, data.get("ppo", {}))
    _update_dataclass(bench, data.get("benchmarks", {}))
    _update_dataclass(sens, data.get("sensitivity", {}))
    if abs(float(exp.discount_factor) - float(ppo.gamma)) > 1e-12:
        raise ValueError("experiment.discount_factor and ppo.gamma must be identical")
    return exp, ppo, bench, sens


def to_nested_dict(
    exp: ExperimentConfig,
    ppo: PPOConfig,
    bench: BenchmarkConfig,
    sens: SensitivityConfig | None = None,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "experiment": asdict(exp),
        "ppo": asdict(ppo),
        "benchmarks": asdict(bench),
    }
    if sens is not None:
        data["sensitivity"] = asdict(sens)
    return data
