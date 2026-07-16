from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from benchmarks import (
    DHPPolicy,
    DTPPolicy,
    NVDPolicy,
    RHSPTPolicy,
    SAAOBCAPolicy,
    calibrate_saa_obca,
)
from rl_ato.config import BenchmarkConfig, ExperimentConfig, PPOConfig, load_config
from rl_ato.evaluate import benchmark_policies, compute_pi_breakdowns
from rl_ato.rlbr import RLBRPolicy
from rl_ato.scenario import ScenarioGenerator, make_instance


def _policies(
    instance,
    experiment: ExperimentConfig,
    ppo: PPOConfig,
    benchmark: BenchmarkConfig,
    model_path: Path,
    output_dir: Path,
):
    rlbr = RLBRPolicy(instance, ppo)
    rlbr.load(str(model_path))
    calibration_generator = ScenarioGenerator(
        instance,
        seed=experiment.seed,
        start_episode=benchmark.saa_calibration_episode_offset,
    )
    calibration_scenarios = [
        calibration_generator.sample()
        for _ in range(int(benchmark.saa_training_paths))
    ]
    saa_result = calibrate_saa_obca(
        instance,
        calibration_scenarios,
        step_sizes=benchmark.saa_step_sizes,
        beta_late=benchmark.saa_beta_late,
        upper_quantile=benchmark.saa_upper_quantile,
    )
    return [
        rlbr,
        NVDPolicy(instance),
        DTPPolicy(instance),
        RHSPTPolicy(
            instance,
            horizon=benchmark.rh_horizon,
            n_scenarios=benchmark.rh_scenarios,
            discount_factor=instance.discount_factor,
            terminal_backlog_weight=benchmark.rh_terminal_backlog_weight,
            terminal_inventory_weight=benchmark.rh_terminal_inventory_weight,
            time_limit=benchmark.rh_time_limit,
            mip_gap=benchmark.rh_mip_gap,
            threads=benchmark.rh_threads,
            seed=experiment.seed + benchmark.rh_seed_offset,
        ),
        SAAOBCAPolicy(
            instance,
            saa_result.base_stock,
            beta_late=benchmark.saa_beta_late,
        ),
        DHPPolicy(
            instance,
            calibration_config={
                "Z_max": benchmark.dhp_z_max,
                "n_cal_paths": benchmark.dhp_calibration_paths,
                "calibration_episode_offset": benchmark.dhp_calibration_episode_offset,
                "cache_results": True,
                "cache_dir": str(output_dir / "dhp_calibration"),
            },
            seed=experiment.seed,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manuscript.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/sensitivity")
    parser.add_argument("--parameter")
    arguments = parser.parse_args()
    baseline, ppo, benchmark, sensitivity = load_config(arguments.config)
    if int(baseline.pi_episodes) != int(baseline.evaluation_episodes):
        raise ValueError("PI must use all common held-out paths")
    parameter_values = {
        name: list(getattr(sensitivity, name))
        for name in sensitivity.__dataclass_fields__
    }
    if arguments.parameter is not None:
        if arguments.parameter not in parameter_values:
            raise ValueError(f"Unknown sensitivity parameter: {arguments.parameter}")
        parameter_values = {arguments.parameter: parameter_values[arguments.parameter]}
    model_path = Path(arguments.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_frames = []
    summary_frames = []
    for parameter, values in parameter_values.items():
        for value in values:
            experiment = replace(baseline)
            setattr(experiment, parameter, value)
            if parameter == "max_replenishment_lead_time":
                experiment.min_replenishment_lead_time = 1
            if parameter == "seasonal_beta":
                experiment.demand_pattern = "seasonal"
            instance = make_instance(experiment)
            realized_parameter_value = float(value)
            if parameter == "component_commonality":
                realized_parameter_value = float(instance.realized_commonality)
            elif parameter == "demand_cv":
                realized_parameter_value = float(
                    instance.config["realized_demand_cv"]
                )
            elif parameter == "demand_correlation":
                realized_parameter_value = float(
                    instance.empirical_demand_correlation
                )
            policies = _policies(
                instance,
                experiment,
                ppo,
                benchmark,
                model_path,
                output_dir,
            )
            holdout_generator = ScenarioGenerator(
                instance,
                seed=experiment.seed,
                start_episode=experiment.holdout_episode_offset,
            )
            scenarios = [
                holdout_generator.sample()
                for _ in range(int(experiment.evaluation_episodes))
            ]
            pi_breakdowns = compute_pi_breakdowns(instance, scenarios)
            episodes, summary = benchmark_policies(
                policies,
                instance,
                scenarios,
                pi_breakdowns,
            )
            episodes.insert(0, "parameter", parameter)
            episodes.insert(1, "parameter_value", value)
            episodes.insert(2, "realized_parameter_value", realized_parameter_value)
            summary.insert(0, "parameter", parameter)
            summary.insert(1, "parameter_value", value)
            summary.insert(2, "realized_parameter_value", realized_parameter_value)
            episode_frames.append(episodes)
            summary_frames.append(summary)
    pd.concat(episode_frames, ignore_index=True).to_csv(output_dir / "episodes.csv", index=False)
    pd.concat(summary_frames, ignore_index=True).to_csv(output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
