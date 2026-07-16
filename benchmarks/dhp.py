from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping

import numpy as np

from rl_ato.env import ATOEnv, ControlAction, Observation
from rl_ato.scenario import ProblemInstance

from .base import BasePolicy
from .dhp_calibration import (
    DHPCalibrationConfig,
    DHPCalibrationResult,
    DHPParameters,
    calibrate_dhp_parameters,
)
from .nvd import NVDPolicy


class DHPPolicy(BasePolicy):
    name = "DHP"

    def __init__(
        self,
        instance: ProblemInstance,
        calibration_config: DHPCalibrationConfig | Mapping[str, Any] | None = None,
        parameters: DHPParameters | Mapping[str, Any] | None = None,
        seed: int = 0,
    ):
        self.instance = instance
        self.seed = int(seed)
        self.calibration_config = (
            calibration_config
            if isinstance(calibration_config, DHPCalibrationConfig)
            else DHPCalibrationConfig.from_mapping(calibration_config)
        )
        if parameters is None:
            result = calibrate_dhp_parameters(
                instance,
                self.calibration_config,
                seed=self.seed,
            )
            self.calibration_result: DHPCalibrationResult | None = result
            self.parameters = result.parameters
            self.base_stock = np.asarray(result.base_stock, dtype=float).copy()
            self.cache_path = result.cache_path
            self.metadata = result.metadata
        else:
            self.calibration_result = None
            self.parameters = (
                parameters
                if isinstance(parameters, DHPParameters)
                else DHPParameters.from_mapping(parameters)
            )
            self.parameters.validate()
            self.base_stock = np.asarray(NVDPolicy(instance).base_stock, dtype=float).copy()
            self.cache_path = ""
            self.metadata = {
                "policy": self.name,
                "parameters": asdict(self.parameters),
                "Z_max": int(self.calibration_config.Z_max),
                "method": "provided Appendix C parameters",
            }
        if self.base_stock.shape != (instance.J,):
            raise ValueError(f"NVD base_stock must have shape ({instance.J},)")
        self.base_stock = np.maximum(self.base_stock, 0.0)
        self.calibration_runtime_seconds = float(
            self.metadata.get("runtime_seconds", 0.0)
        )
        self.calibration_mode = str(self.metadata.get("method", ""))
        self.components_of_product = tuple(
            tuple(int(j) for j in np.flatnonzero(instance.support[i]))
            for i in range(instance.I)
        )
        self.products_using_component = tuple(
            tuple(int(i) for i in np.flatnonzero(instance.support[:, j]))
            for j in range(instance.J)
        )

    def act(self, env: ATOEnv, obs: Observation | None = None) -> ControlAction:
        if obs is None:
            obs = env.observe()
        if env.scenario is None:
            raise RuntimeError("DHP requires an initialized environment")
        local_inventory = np.asarray(obs.inventory, dtype=float).copy()
        local_remaining = np.asarray(env.remaining, dtype=float).copy()
        revealed_load, unrevealed_load = self._remaining_loads(
            env,
            obs.t,
            local_remaining,
        )
        cohorts = [
            (
                int(product),
                int(cohort_period),
                np.asarray(realized_bom, dtype=float),
            )
            for product, cohort_period, remaining, realized_bom in obs.revealed
            if float(remaining) > 0.0
        ]
        cohorts.sort(
            key=lambda cohort: (
                -self._priority_c2(cohort[0], cohort[1], obs.t),
                cohort[1],
                float(np.sum(cohort[2])),
                cohort[0],
            )
        )
        allocations: list[tuple[int, int, float]] = []
        for product, cohort_period, realized_bom in cohorts:
            quantity = 0
            while local_remaining[product, cohort_period] >= 1.0:
                if not self._can_allocate_c3(
                    product,
                    cohort_period,
                    obs.t,
                    realized_bom,
                    local_inventory,
                    revealed_load,
                    unrevealed_load,
                ):
                    break
                local_inventory -= realized_bom
                local_remaining[product, cohort_period] -= 1.0
                revealed_load -= realized_bom
                quantity += 1
            if quantity:
                allocations.append((product, cohort_period, float(quantity)))
        orders = self._orders_c4(
            obs,
            local_inventory,
            revealed_load,
            unrevealed_load,
        )
        return ControlAction(allocations=allocations, orders=orders)

    def _remaining_loads(
        self,
        env: ATOEnv,
        current_period: int,
        local_remaining: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if env.scenario is None:
            raise RuntimeError("DHP requires an initialized environment")
        remaining = np.asarray(local_remaining[:, : current_period + 1], dtype=float)
        revealed_mask = np.asarray(
            env.revealed[:, : current_period + 1],
            dtype=bool,
        )
        realized_bom = np.asarray(
            env.scenario.realized_bom[:, : current_period + 1, :],
            dtype=float,
        )
        revealed_load = np.einsum(
            "is,isj->j",
            remaining * revealed_mask,
            realized_bom,
            optimize=True,
        )
        unrevealed_by_product = np.sum(
            remaining * np.logical_not(revealed_mask),
            axis=1,
        )
        unrevealed_load = unrevealed_by_product @ np.asarray(
            self.instance.template_bom,
            dtype=float,
        )
        return (
            np.asarray(revealed_load, dtype=float),
            np.asarray(unrevealed_load, dtype=float),
        )

    def _z_c1(
        self,
        product: int,
        component: int,
        local_inventory: np.ndarray,
        revealed_load: np.ndarray,
        unrevealed_load: np.ndarray,
    ) -> int:
        other_components = tuple(
            candidate
            for candidate in self.components_of_product[product]
            if candidate != component
        )
        if not other_components:
            return int(self.calibration_config.Z_max)
        supported_units = []
        for other in other_components:
            residual = max(
                0.0,
                float(
                    local_inventory[other]
                    - revealed_load[other]
                    - unrevealed_load[other]
                ),
            )
            denominator = max(
                1.0,
                float(self.instance.template_bom[product, other]),
            )
            supported_units.append(math.floor(residual / denominator))
        return int(
            min(
                int(self.calibration_config.Z_max),
                max(0, min(supported_units)),
            )
        )

    def _urgency_c2(
        self,
        product: int,
        cohort_period: int,
        current_period: int,
    ) -> float:
        due_period = (
            cohort_period
            + int(self.instance.design_lead_times[product])
            + int(self.instance.delivery_window)
        )
        late = 1.0 if current_period > due_period else 0.0
        slack = max(0.0, float(due_period - current_period))
        age = float(current_period - cohort_period)
        return float(
            self.parameters.beta_late * late
            + self.parameters.beta_slack / (1.0 + slack)
            + self.parameters.beta_age * age / (1.0 + age)
        )

    def _priority_c2(
        self,
        product: int,
        cohort_period: int,
        current_period: int,
    ) -> float:
        return float(
            self.instance.backlog_costs[product]
            * (
                1.0
                + self._urgency_c2(product, cohort_period, current_period)
            )
        )

    def _can_allocate_c3(
        self,
        product: int,
        cohort_period: int,
        current_period: int,
        realized_bom: np.ndarray,
        local_inventory: np.ndarray,
        revealed_load: np.ndarray,
        unrevealed_load: np.ndarray,
    ) -> bool:
        urgency = self._urgency_c2(product, cohort_period, current_period)
        denominator = 1.0 + self.parameters.beta_R * urgency
        z_max = float(self.calibration_config.Z_max)
        for component in self.components_of_product[product]:
            z_value = self._z_c1(
                product,
                component,
                local_inventory,
                revealed_load,
                unrevealed_load,
            )
            reserve = math.floor(
                self.parameters.rho_R
                * float(self.base_stock[component])
                * (1.0 - float(z_value) / z_max)
                / denominator
            )
            if (
                float(local_inventory[component] - realized_bom[component])
                < float(reserve)
            ):
                return False
        return True

    def _orders_c4(
        self,
        obs: Observation,
        local_inventory: np.ndarray,
        revealed_load: np.ndarray,
        unrevealed_load: np.ndarray,
    ) -> np.ndarray:
        orders = np.zeros(self.instance.J, dtype=float)
        for component in range(self.instance.J):
            delta = max(
                (
                    float(self.instance.template_bom[product, component])
                    * float(
                        self._z_c1(
                            product,
                            component,
                            local_inventory,
                            revealed_load,
                            unrevealed_load,
                        )
                    )
                    for product in self.products_using_component[component]
                ),
                default=0.0,
            )
            target = (
                float(self.base_stock[component])
                + self.parameters.rho_delta * delta
                + float(revealed_load[component])
                + float(unrevealed_load[component])
            )
            inventory_position = float(
                local_inventory[component] + obs.outstanding[component]
            )
            orders[component] = float(
                max(0, math.ceil(target - inventory_position))
            )
        return orders
