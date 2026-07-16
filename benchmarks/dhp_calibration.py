from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np

from rl_ato.scenario import ProblemInstance, Scenario, ScenarioGenerator

from .nvd import NVDPolicy


DHP_PARAMETER_GRIDS: dict[str, tuple[float, ...]] = {
    "rho_R": (0.0, 0.1, 0.25, 0.5, 0.75),
    "rho_delta": (0.0, 0.25, 0.5, 0.75, 1.0),
    "beta_late": (0.0, 0.5, 1.0, 2.0),
    "beta_slack": (0.0, 0.5, 1.0, 2.0),
    "beta_age": (0.0, 0.25, 0.5, 1.0),
    "beta_R": (0.0, 0.5, 1.0, 2.0),
}
DHP_CACHE_SCHEMA = "dhp-appendix-c-v2"


@dataclass(frozen=True)
class DHPParameters:
    rho_R: float = 0.25
    rho_delta: float = 0.5
    beta_late: float = 1.0
    beta_slack: float = 0.5
    beta_age: float = 0.25
    beta_R: float = 0.5

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "DHPParameters":
        if values is None:
            parameters = cls()
        else:
            allowed = {field.name for field in fields(cls)}
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise ValueError(
                    "Unknown DHP parameters: " + ", ".join(unknown)
                )
            parameters = cls(
                **{key: float(value) for key, value in values.items()}
            )
        parameters.validate()
        return parameters

    def validate(self) -> None:
        for parameter_name, grid in DHP_PARAMETER_GRIDS.items():
            value = float(getattr(self, parameter_name))
            if value not in grid:
                raise ValueError(
                    f"{parameter_name}={value} is outside the Appendix C grid"
                )

    def key(self) -> tuple[float, ...]:
        return tuple(
            float(getattr(self, field.name))
            for field in fields(self)
        )


@dataclass
class DHPCalibrationConfig:
    Z_max: int = 10
    n_cal_paths: int = 200
    calibration_episode_offset: int = 1_000_000
    cache_results: bool = True
    cache_dir: str = "outputs/dhp_calibration"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "DHPCalibrationConfig":
        if values is None:
            config = cls()
        else:
            allowed = {field.name for field in fields(cls)}
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise ValueError(
                    "Unknown DHP calibration fields: " + ", ".join(unknown)
                )
            config = cls(**dict(values))
        config.Z_max = int(config.Z_max)
        config.n_cal_paths = int(config.n_cal_paths)
        config.calibration_episode_offset = int(
            config.calibration_episode_offset
        )
        config.cache_results = bool(config.cache_results)
        config.cache_dir = str(config.cache_dir)
        if config.Z_max <= 0:
            raise ValueError("Z_max must be positive")
        if config.n_cal_paths <= 0:
            raise ValueError("n_cal_paths must be positive")
        if config.calibration_episode_offset < 0:
            raise ValueError("calibration_episode_offset must be nonnegative")
        return config


@dataclass
class DHPCalibrationResult:
    parameters: DHPParameters
    base_stock: np.ndarray
    objective: float
    metadata: dict[str, Any]
    cache_path: str = ""


def calibrate_dhp_parameters(
    instance: ProblemInstance,
    calibration_config: DHPCalibrationConfig | Mapping[str, Any] | None = None,
    seed: int = 0,
) -> DHPCalibrationResult:
    config = (
        calibration_config
        if isinstance(calibration_config, DHPCalibrationConfig)
        else DHPCalibrationConfig.from_mapping(calibration_config)
    )
    base_stock = np.maximum(
        0.0,
        np.asarray(NVDPolicy(instance).base_stock, dtype=float),
    )
    digest = _instance_digest(instance, config, int(seed), base_stock)
    cache_path = Path(config.cache_dir) / f"dhp_{digest}.pkl"
    if config.cache_results and cache_path.exists():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("schema") == DHP_CACHE_SCHEMA:
            cached_parameters = DHPParameters.from_mapping(
                payload.get("parameters")
            )
            cached_stock = np.asarray(payload.get("base_stock"), dtype=float)
            if (
                cached_stock.shape == base_stock.shape
                and np.allclose(cached_stock, base_stock, rtol=0.0, atol=1e-12)
            ):
                return DHPCalibrationResult(
                    parameters=cached_parameters,
                    base_stock=cached_stock.copy(),
                    objective=float(payload["objective"]),
                    metadata=dict(payload["metadata"]),
                    cache_path=str(cache_path),
                )
    generator = ScenarioGenerator(
        instance,
        seed=int(seed),
        start_episode=config.calibration_episode_offset,
    )
    scenarios = tuple(
        generator.sample()
        for _ in range(config.n_cal_paths)
    )
    objective_cache: dict[tuple[float, ...], float] = {}

    def objective(parameters: DHPParameters) -> float:
        key = parameters.key()
        if key not in objective_cache:
            objective_cache[key] = _mean_policy_cost(
                instance,
                scenarios,
                parameters,
                config.Z_max,
            )
        return objective_cache[key]

    started = time.perf_counter()
    current = DHPParameters()
    current.validate()
    current_objective = objective(current)
    history: list[dict[str, Any]] = []
    sweeps_completed = 0
    while True:
        improved = False
        sweeps_completed += 1
        for parameter_name, grid in DHP_PARAMETER_GRIDS.items():
            selected = current
            selected_objective = current_objective
            for value in grid:
                candidate = replace(
                    current,
                    **{parameter_name: float(value)},
                )
                candidate_objective = objective(candidate)
                if candidate_objective < selected_objective - 1e-12:
                    selected = candidate
                    selected_objective = candidate_objective
            if selected != current:
                current = selected
                current_objective = selected_objective
                improved = True
            history.append(
                {
                    "sweep": sweeps_completed,
                    "coordinate": parameter_name,
                    "selected_value": float(
                        getattr(current, parameter_name)
                    ),
                    "objective": float(current_objective),
                }
            )
        if not improved:
            break
    metadata: dict[str, Any] = {
        "policy": "DHP",
        "cache_schema": DHP_CACHE_SCHEMA,
        "method": "Appendix C CRN coordinate search",
        "seed": int(seed),
        "Z_max": int(config.Z_max),
        "n_cal_paths": int(config.n_cal_paths),
        "calibration_episode_offset": int(config.calibration_episode_offset),
        "episode_ids": [int(scenario.episode) for scenario in scenarios],
        "parameter_grids": {
            name: list(values)
            for name, values in DHP_PARAMETER_GRIDS.items()
        },
        "initial_parameters": asdict(DHPParameters()),
        "sweeps_completed": int(sweeps_completed),
        "evaluated_parameter_vectors": int(len(objective_cache)),
        "history": history,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    result = DHPCalibrationResult(
        parameters=current,
        base_stock=base_stock.copy(),
        objective=float(current_objective),
        metadata=metadata,
        cache_path=str(cache_path) if config.cache_results else "",
    )
    if config.cache_results:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(
                {
                    "schema": DHP_CACHE_SCHEMA,
                    "parameters": asdict(result.parameters),
                    "base_stock": result.base_stock,
                    "objective": result.objective,
                    "metadata": result.metadata,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    return result


def _mean_policy_cost(
    instance: ProblemInstance,
    scenarios: Sequence[Scenario],
    parameters: DHPParameters,
    z_max: int,
) -> float:
    from rl_ato.env import ATOEnv

    from .dhp import DHPPolicy

    policy = DHPPolicy(
        instance,
        parameters=parameters,
        calibration_config={
            "Z_max": int(z_max),
            "n_cal_paths": 200,
            "cache_results": False,
        },
    )
    costs = []
    for scenario in scenarios:
        env = ATOEnv(instance)
        observation = env.reset(scenario)
        done = False
        while not done:
            action = policy.act(env, observation)
            next_observation, _reward, done, _info = env.step(action)
            if next_observation is not None:
                observation = next_observation
        costs.append(float(env.total_cost))
    return float(np.mean(np.asarray(costs, dtype=float)))


def _instance_digest(
    instance: ProblemInstance,
    config: DHPCalibrationConfig,
    seed: int,
    base_stock: np.ndarray,
) -> str:
    payload = {
        "cache_schema": DHP_CACHE_SCHEMA,
        "I": int(instance.I),
        "J": int(instance.J),
        "T": int(instance.T),
        "template_bom": np.asarray(
            instance.template_bom,
            dtype=float,
        ).tolist(),
        "design_lead_times": np.asarray(
            instance.design_lead_times,
            dtype=int,
        ).tolist(),
        "delivery_window": int(instance.delivery_window),
        "initial_inventory": np.asarray(
            instance.initial_inventory,
            dtype=float,
        ).tolist(),
        "holding_costs": np.asarray(
            instance.holding_costs,
            dtype=float,
        ).tolist(),
        "ordering_costs": np.asarray(
            instance.ordering_costs,
            dtype=float,
        ).tolist(),
        "backlog_costs": np.asarray(
            instance.backlog_costs,
            dtype=float,
        ).tolist(),
        "demand_lambdas": np.asarray(
            instance.demand_lambdas,
            dtype=float,
        ).tolist(),
        "demand_pattern": str(instance.demand_pattern),
        "demand_correlation": float(instance.demand_correlation),
        "latent_demand_correlation": float(
            instance.latent_demand_correlation
        ),
        "seasonal_beta": float(instance.seasonal_beta),
        "seasonal_cycle": int(instance.seasonal_cycle),
        "seasonal_phases": np.asarray(
            instance.seasonal_phases,
            dtype=float,
        ).tolist(),
        "bom_cv": float(instance.bom_cv),
        "lead_time_range": [
            int(instance.min_replenishment_lead_time),
            int(instance.max_replenishment_lead_time),
        ],
        "discount_factor": float(instance.discount_factor),
        "base_stock": np.asarray(base_stock, dtype=float).tolist(),
        "Z_max": int(config.Z_max),
        "n_cal_paths": int(config.n_cal_paths),
        "calibration_episode_offset": int(
            config.calibration_episode_offset
        ),
        "seed": int(seed),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
