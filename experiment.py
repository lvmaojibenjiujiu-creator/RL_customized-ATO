from __future__ import annotations

import argparse
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
from rl_ato.config import load_config
from rl_ato.evaluate import benchmark_policies, compute_pi_breakdowns
from rl_ato.rlbr import RLBRPolicy
from rl_ato.scenario import ScenarioGenerator, make_instance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manuscript.yaml")
    parser.add_argument("--models-dir", default="outputs/models")
    parser.add_argument("--output-dir", default="outputs/comparison")
    parser.add_argument("--products", type=int)
    parser.add_argument("--components", type=int)
    parser.add_argument("--demand-pattern", choices=["poisson", "seasonal"])
    arguments = parser.parse_args()
    experiment, ppo, benchmark, _sensitivity = load_config(arguments.config)
    if int(experiment.pi_episodes) != int(experiment.evaluation_episodes):
        raise ValueError("PI must use all common held-out paths")
    if (arguments.products is None) != (arguments.components is None):
        raise ValueError("products and components must be provided together")
    scales = (
        [(int(arguments.products), int(arguments.components))]
        if arguments.products is not None
        else [(int(scale[0]), int(scale[1])) for scale in experiment.comparison_scales]
    )
    patterns = (
        [str(arguments.demand_pattern)]
        if arguments.demand_pattern is not None
        else [str(pattern) for pattern in experiment.comparison_demand_patterns]
    )
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_frames = []
    summary_frames = []
    for pattern in patterns:
        for products, components in scales:
            experiment.products = products
            experiment.components = components
            experiment.demand_pattern = pattern
            instance = make_instance(experiment)
            model_path = Path(arguments.models_dir) / f"rlbr_{pattern}_{products}x{components}.pt"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
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
            saa = SAAOBCAPolicy(
                instance,
                saa_result.base_stock,
                beta_late=benchmark.saa_beta_late,
            )
            dhp = DHPPolicy(
                instance,
                calibration_config={
                    "Z_max": benchmark.dhp_z_max,
                    "n_cal_paths": benchmark.dhp_calibration_paths,
                    "calibration_episode_offset": benchmark.dhp_calibration_episode_offset,
                    "cache_results": True,
                    "cache_dir": str(output_dir / "dhp_calibration"),
                },
                seed=experiment.seed,
            )
            rh_spt = RHSPTPolicy(
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
            policies = [
                rlbr,
                NVDPolicy(instance),
                DTPPolicy(instance),
                rh_spt,
                saa,
                dhp,
            ]
            episodes, summary = benchmark_policies(
                policies,
                instance,
                scenarios,
                pi_breakdowns,
            )
            episodes.insert(0, "demand_pattern", pattern)
            episodes.insert(1, "products", products)
            episodes.insert(2, "components", components)
            summary.insert(0, "demand_pattern", pattern)
            summary.insert(1, "products", products)
            summary.insert(2, "components", components)
            episode_frames.append(episodes)
            summary_frames.append(summary)
            stem = f"{pattern}_{products}x{components}"
            episodes.to_csv(output_dir / f"episodes_{stem}.csv", index=False)
            summary.to_csv(output_dir / f"summary_{stem}.csv", index=False)
    pd.concat(episode_frames, ignore_index=True).to_csv(output_dir / "episodes.csv", index=False)
    pd.concat(summary_frames, ignore_index=True).to_csv(output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
