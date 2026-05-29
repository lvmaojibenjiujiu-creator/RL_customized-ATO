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
    min_replenishment_lead_time: int = 2
    max_replenishment_lead_time: int = 6
    component_commonality: float = 0.3
    commonality_tolerance: float = 0.02
    family_degree_min: int = 2
    family_degree_max: int = 6
    bom_cv: float = 0.2
    random_costs: bool = True
    holding_cost: float = 1.0
    ordering_cost: float = 1.0
    holding_cost_low: float = 0.8
    holding_cost_high: float = 1.2
    ordering_cost_low: float = 0.8
    ordering_cost_high: float = 1.2
    backorder_to_holding: float = 2.0
    initial_inventory_factor: float = 1.0
    correlation_pilot_samples: int = 8000
    correlation_tolerance: float = 1e-3
    history_window: int = 8
    train_episodes: int = 1000
    eval_episodes: int = 1000
    pi_episodes: int = 50
    pi_ordering_cost_weight: float = 1.0

    @property
    def scale(self) -> Tuple[int, int]:
        return (self.products, self.components)


@dataclass
class PPOConfig:
    network_seed: int = -1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 1e-2
    value_coef: float = 0.5
    reward_scale: float = 1e-3
    learning_rate: float = 3e-4
    batch_size: int = 256
    epochs: int = 10
    rollout_episodes: int = 4
    max_grad_norm: float = 0.7
    gat_layers: int = 2
    gat_hidden_width: int = 128
    attention_heads: int = 4
    gru_hidden_size: int = 64
    context_dim: int = 64
    eta_max: float = 1.0
    allocation_solver: str = "greedy"
    allocation_weight_mode: str = "rlbr"
    allocation_due_weight: float = 0.0
    allocation_late_weight: float = 1.0
    allocation_holding_augmented: bool = False
    base_stock_scale: float = 1.0
    base_stock_floor: float = 0.0
    base_stock_safety: float = 0.0
    nvd_floor_scale: float = 0.0
    actual_revealed_bom_replenishment: bool = False
    adaptive_dtp_blend: float = 0.0
    short_design_scale: float = 1.0
    shadow_price_scale: float = 1.0
    alpha_mon_start: float = 1e-4
    alpha_lip_start: float = 1e-4
    alpha_rat_start: float = 1e-4
    alpha_mon_end: float = 1e-2
    alpha_lip_end: float = 1e-2
    alpha_rat_end: float = 1e-2
    regularization_samples: int = 16
    device: str = "cpu"


@dataclass
class SensitivityConfig:
    bom_cv: Iterable[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    component_commonality: Iterable[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
    )
    delivery_window: Iterable[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    max_replenishment_lead_time: Iterable[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    design_lead_time: Iterable[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    demand_correlation: Iterable[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
    seasonal_beta: Iterable[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    backorder_to_holding: Iterable[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])


def _update_dataclass(obj: Any, values: Dict[str, Any]) -> Any:
    allowed = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key in allowed:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = None) -> tuple[ExperimentConfig, PPOConfig, SensitivityConfig]:
    exp = ExperimentConfig()
    ppo = PPOConfig()
    sens = SensitivityConfig()
    if path is None:
        return exp, ppo, sens

    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    _update_dataclass(exp, data.get("experiment", {}))
    _update_dataclass(ppo, data.get("ppo", {}))
    _update_dataclass(sens, data.get("sensitivity", {}))
    return exp, ppo, sens


def to_nested_dict(exp: ExperimentConfig, ppo: PPOConfig, sens: SensitivityConfig | None = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {"experiment": asdict(exp), "ppo": asdict(ppo)}
    if sens is not None:
        data["sensitivity"] = asdict(sens)
    return data
