from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from rl_ato.env import ATOEnv, ControlAction, Observation
from rl_ato.policies import BasePolicy
from rl_ato.scenario import ProblemInstance

from .dhp_sbr_calibration import (
    DHPCalibrationResult,
    DHPSBRCalibrationConfig,
    calibrate_dhp_sbr_tables,
)


@dataclass
class DHPSBROnlineConfig:
    unrevealed_load_weight_mode: str = "none"
    imminent_pipeline_window_mode: str = "mean_lead_time"
    imminent_pipeline_window: int = 0
    delivery_slack: int = 0
    replenishment_target_aggregation: str = "min"
    weighted_quantile_q: float = 0.25

    @classmethod
    def from_mapping(cls, values: Dict[str, Any] | None) -> "DHPSBROnlineConfig":
        cfg = cls()
        if not values:
            return cfg
        allowed = set(asdict(cfg))
        for key, value in values.items():
            if key in allowed:
                setattr(cfg, key, value)
        cfg.unrevealed_load_weight_mode = str(cfg.unrevealed_load_weight_mode)
        cfg.imminent_pipeline_window_mode = str(cfg.imminent_pipeline_window_mode)
        cfg.imminent_pipeline_window = int(max(0, cfg.imminent_pipeline_window))
        cfg.delivery_slack = int(max(0, cfg.delivery_slack))
        cfg.replenishment_target_aggregation = str(cfg.replenishment_target_aggregation)
        cfg.weighted_quantile_q = float(np.clip(cfg.weighted_quantile_q, 0.0, 1.0))
        return cfg


class DHPSBRPolicy(BasePolicy):
    name = "DHP-SBR"

    def __init__(
        self,
        instance: ProblemInstance,
        calibration_config: DHPSBRCalibrationConfig | Dict[str, Any] | None = None,
        online_config: DHPSBROnlineConfig | Dict[str, Any] | None = None,
        fallback_policy: BasePolicy | None = None,
        seed: int = 0,
    ):
        self.instance = instance
        self.seed = int(seed)
        self.calibration_config = (
            calibration_config
            if isinstance(calibration_config, DHPSBRCalibrationConfig)
            else DHPSBRCalibrationConfig.from_mapping(calibration_config)
        )
        self.online_config = (
            online_config
            if isinstance(online_config, DHPSBROnlineConfig)
            else DHPSBROnlineConfig.from_mapping(online_config)
        )
        self.fallback_policy = fallback_policy
        self.fallback_count = 0
        self.last_solver_status = "CALIBRATED"
        self.last_solver_gap = np.nan
        self._warned_missing_tables: set[Tuple[int, int]] = set()

        result = calibrate_dhp_sbr_tables(instance, self.calibration_config, seed=self.seed)
        self.calibration_result: DHPCalibrationResult = result
        self.S_table = result.S_table
        self.R_table = result.R_table
        self.components_of_product = result.components_of_product
        self.products_using_component = result.products_using_component
        self.metadata = result.metadata
        self.table_path = result.table_path
        self.dhp_table_path = result.table_path
        self.calibration_runtime_seconds = float(result.metadata.get("runtime_seconds", np.nan))
        self.calibration_mode = str(result.metadata.get("effective_calibration_mode", result.metadata.get("calibration_mode", "")))
        self.n_calibration_edges = int(result.metadata.get("n_edges", len(self.S_table)))

    def act(self, env: ATOEnv, obs: Observation | None = None) -> ControlAction:
        if obs is None:
            obs = env.observe()
        try:
            revealed_load = self._revealed_load(obs)
            expected_unrevealed = self._expected_unrevealed_load(env, obs)
            imminent = self._imminent_pipeline(env, obs)
            ip = obs.inventory + obs.outstanding - revealed_load - expected_unrevealed
            a_eff = obs.inventory + imminent - revealed_load - expected_unrevealed
            orders = self._replenishment_orders(ip, a_eff)
            allocations = self._rationed_allocations(env, obs, expected_unrevealed, imminent)
            return ControlAction(allocations=allocations, orders=orders)
        except Exception:
            self.fallback_count += 1
            self.last_solver_status = "FALLBACK"
            if self.fallback_policy is not None:
                return self.fallback_policy.act(env, obs)
            return ControlAction(allocations=[], orders=np.zeros(self.instance.J, dtype=float))

    def _revealed_load(self, obs: Observation) -> np.ndarray:
        load = np.zeros(self.instance.J, dtype=float)
        for p, _s, remaining, bom in obs.revealed:
            load += np.asarray(bom, dtype=float) * float(remaining)
        return load

    def _expected_unrevealed_load(self, env: ATOEnv, obs: Observation) -> np.ndarray:
        inst = self.instance
        mode = self.online_config.unrevealed_load_weight_mode.lower()
        load = np.zeros(inst.J, dtype=float)
        if mode == "none":
            return load
        for p in range(inst.I):
            for s in range(obs.t + 1):
                rem = float(env.remaining[p, s])
                if rem <= 1e-9 or env.revealed[p, s]:
                    continue
                if mode == "full":
                    weight = 1.0
                else:
                    release = int(s + inst.design_lead_times[p])
                    due = int(release + inst.delivery_window)
                    slack = self.online_config.delivery_slack or inst.delivery_window
                    if release <= obs.t + inst.expected_lead_time or due <= obs.t + inst.expected_lead_time + slack:
                        weight = 1.0
                    else:
                        weight = 0.5
                load += rem * float(weight) * inst.template_bom[p]
        return load

    def _imminent_pipeline(self, env: ATOEnv, obs: Observation) -> np.ndarray:
        inst = self.instance
        if self.online_config.imminent_pipeline_window > 0:
            window = int(self.online_config.imminent_pipeline_window)
        elif self.online_config.imminent_pipeline_window_mode.lower() == "none":
            window = 0
        else:
            window = max(1, int(round(inst.expected_lead_time)))
        imminent = np.zeros(inst.J, dtype=float)
        for order in env.pipeline:
            if int(order["arrive"]) <= int(obs.t) + window:
                imminent[int(order["j"])] += float(order["qty"])
        return imminent

    def _z_from_availability(self, component: int, product: int, availability: np.ndarray) -> int:
        others = [j for j in self.components_of_product.get(int(product), ()) if int(j) != int(component)]
        if not others:
            return int(self.calibration_config.Z_max)
        vals = []
        for other in others:
            denom = max(1.0, float(self.instance.template_bom[product, other]))
            vals.append(math.floor(max(0.0, float(availability[other])) / denom))
        return int(np.clip(min(vals) if vals else self.calibration_config.Z_max, 0, self.calibration_config.Z_max))

    def _replenishment_orders(self, ip: np.ndarray, a_eff: np.ndarray) -> np.ndarray:
        inst = self.instance
        orders = np.zeros(inst.J, dtype=float)
        for j in range(inst.J):
            products = list(self.products_using_component.get(j, ()))
            targets = []
            weights = []
            for p in products:
                table = self.S_table.get((int(j), int(p)))
                if table is None:
                    continue
                z = self._z_from_availability(j, p, a_eff)
                targets.append(float(table[z]))
                weights.append(float(inst.demand_lambdas[p]))
            target = self._aggregate_targets(targets, weights)
            orders[j] = max(0.0, math.ceil(target - float(ip[j]) - 1e-9))
        return orders

    def _aggregate_targets(self, targets: List[float], weights: List[float]) -> float:
        if not targets:
            return 0.0
        mode = self.online_config.replenishment_target_aggregation.lower()
        arr = np.asarray(targets, dtype=float)
        if mode == "demand_weighted_mean":
            w = np.asarray(weights, dtype=float)
            return float(np.average(arr, weights=w)) if float(w.sum()) > 1e-12 else float(arr.mean())
        if mode == "demand_weighted_quantile":
            return _weighted_quantile(arr, np.asarray(weights, dtype=float), self.online_config.weighted_quantile_q)
        return float(np.min(arr))

    def _rationed_allocations(
        self,
        env: ATOEnv,
        obs: Observation,
        expected_unrevealed: np.ndarray,
        imminent: np.ndarray,
    ) -> List[Tuple[int, int, float]]:
        inst = self.instance
        local = obs.inventory.copy()
        allocations: List[Tuple[int, int, float]] = []
        cohorts = []
        for p, s, remaining, bom in obs.revealed:
            due = int(s + inst.design_lead_times[p] + inst.delivery_window)
            late = obs.t > due
            slack = max(0, due - obs.t)
            cohorts.append((not late, slack, -float(inst.backlog_costs[p]), int(s), int(p), float(remaining), np.asarray(bom, dtype=float)))
        cohorts.sort()
        for _not_late, _slack, _neg_penalty, s, p, remaining, bom in cohorts:
            qty = 0
            while qty + 1 <= int(math.floor(remaining + 1e-9)):
                if not self._can_fulfill_one(p, bom, local, expected_unrevealed, imminent):
                    break
                local -= bom
                local = np.maximum(local, 0.0)
                qty += 1
            if qty > 0:
                allocations.append((int(p), int(s), float(qty)))
        return allocations

    def _can_fulfill_one(
        self,
        product: int,
        bom: np.ndarray,
        local_on_hand: np.ndarray,
        expected_unrevealed: np.ndarray,
        imminent: np.ndarray,
    ) -> bool:
        availability = local_on_hand + imminent - expected_unrevealed
        for j in np.flatnonzero(bom > 1e-12):
            table = self.R_table.get((int(j), int(product)))
            if table is None:
                reserve = 0.0
                key = (int(j), int(product))
                if key not in self._warned_missing_tables:
                    self._warned_missing_tables.add(key)
                    print(f"[DHP-SBR] missing S/R table for edge={key}; using reserve=0")
            else:
                z = self._z_from_availability(int(j), int(product), availability)
                reserve = float(table[z])
            if float(local_on_hand[j]) - float(bom[j]) < reserve - 1e-9:
                return False
        return True


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    weights = np.maximum(0.0, np.asarray(weights, dtype=float))
    if float(weights.sum()) <= 1e-12:
        return float(np.quantile(values, q))
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / float(weights.sum())
    return float(values[int(np.searchsorted(cdf, q, side="left"))])
