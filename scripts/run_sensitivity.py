#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rl_ato_mpl")

import pandas as pd

from rl_ato.config import load_config
from rl_ato.evaluate import benchmark_policies, compute_pi_costs
from rl_ato.plotting import plot_sensitivity_figures
from rl_ato.policies import DTPPolicy, NVDPolicy, SAABSOBCAOptimizedPolicy, tune_saa_bs_obca
from rl_ato.rlbr import RLBRPolicy
from rl_ato.scenario import ScenarioGenerator, make_instance
from rl_ato.train import train_rlbr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-at-a-time sensitivity analyses.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="outputs/rlbr.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--pi-episodes", type=int, default=20)
    parser.add_argument("--pi-ordering-weight", type=float, default=None)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--parameters", default="", help="Comma-separated subset of sensitivity parameters.")
    parser.add_argument("--max-values", type=int, default=None, help="Limit values per parameter.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--fig-dir", default=None, help="Directory for manuscript-style figures.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated figure formats.")
    parser.add_argument("--dtp-known-scale", type=float, default=1.0)
    parser.add_argument("--include-saa-bs-obca", action="store_true")
    parser.add_argument("--saa-train-episodes", type=int, default=20)
    parser.add_argument("--saa-step-sizes", default="8,4,2,1")
    parser.add_argument("--saa-beta-late", type=float, default=1.0)
    parser.add_argument("--saa-known-scale", type=float, default=0.0)
    parser.add_argument("--saa-max-sweeps", type=int, default=3)
    parser.add_argument(
        "--saa-allocation-solver",
        choices=["gurobi"],
        default="gurobi",
        help="SAA OBCA solver. Uses integer Gurobi.",
    )
    parser.add_argument("--out", default="outputs/sensitivity_summary.csv")
    args = parser.parse_args()

    exp, ppo, sens = load_config(args.config)
    rows = []
    episode_rows = []
    variables = {
        "bom_cv": sens.bom_cv,
        "component_commonality": sens.component_commonality,
        "delivery_window": sens.delivery_window,
        "max_replenishment_lead_time": sens.max_replenishment_lead_time,
        "design_lead_time": sens.design_lead_time,
        "demand_correlation": sens.demand_correlation,
        "seasonal_beta": sens.seasonal_beta,
        "backorder_to_holding": sens.backorder_to_holding,
    }
    if args.parameters:
        requested = {x.strip() for x in args.parameters.split(",") if x.strip()}
        variables = {k: v for k, v in variables.items() if k in requested}

    for var, values in variables.items():
        value_list = list(values)
        if args.max_values is not None:
            value_list = value_list[: args.max_values]
        for pos, value in enumerate(value_list):
            cfg = replace(exp)
            setattr(cfg, var, value)
            cfg.eval_episodes = args.episodes
            cfg.pi_episodes = args.pi_episodes
            if args.pi_ordering_weight is not None:
                cfg.pi_ordering_cost_weight = args.pi_ordering_weight
            if var == "seasonal_beta":
                cfg.demand_pattern = "seasonal"
            if args.train_episodes is not None:
                cfg.train_episodes = args.train_episodes
            instance = make_instance(cfg, seed=cfg.seed)
            if args.retrain:
                policy, _history = train_rlbr(instance, cfg, ppo, progress=False)
            else:
                policy = RLBRPolicy(instance, ppo)
                policy.load(args.model)
            policies = [policy, NVDPolicy(instance), DTPPolicy(instance, known_demand_scale=args.dtp_known_scale)]
            if args.include_saa_bs_obca:
                step_sizes = tuple(float(x) for x in args.saa_step_sizes.split(",") if x.strip())
                saa_solver = "gurobi"
                train_generator = ScenarioGenerator(instance, start_episode=1_000_000)
                train_scenarios = [train_generator.sample() for _ in range(args.saa_train_episodes)]
                saa_result = tune_saa_bs_obca(
                    instance,
                    train_scenarios,
                    step_sizes=step_sizes,
                    beta_late=args.saa_beta_late,
                    allocation_solver=saa_solver,
                    max_sweeps_per_delta=args.saa_max_sweeps,
                    known_requirement_scale=args.saa_known_scale,
                )
                policies.append(
                    SAABSOBCAOptimizedPolicy(
                        instance,
                        saa_result.base_stock,
                        beta_late=args.saa_beta_late,
                        allocation_solver=saa_solver,
                        known_requirement_scale=args.saa_known_scale,
                    )
                )
            generator = ScenarioGenerator(instance)
            scenarios = [generator.sample() for _ in range(cfg.eval_episodes)]
            pi_costs = (
                compute_pi_costs(
                    instance,
                    scenarios,
                    limit=cfg.pi_episodes,
                    ordering_cost_weight=cfg.pi_ordering_cost_weight,
                )
                if cfg.pi_episodes > 0
                else None
            )
            episodes, summary = benchmark_policies(policies, instance, scenarios, pi_costs=pi_costs)
            episodes["parameter"] = var
            episodes["value"] = value
            if pi_costs is not None:
                pi_lookup = {idx: cost for idx, cost in enumerate(pi_costs) if pd.notna(cost)}
                episodes["pi_cost"] = episodes["episode"].map(pi_lookup)
                episodes["cost_to_pi_ratio"] = episodes["cost"] / episodes["pi_cost"]
            else:
                episodes["pi_cost"] = float("nan")
                episodes["cost_to_pi_ratio"] = float("nan")
            summary["parameter"] = var
            summary["value"] = value
            summary["pi_ordering_cost_weight"] = cfg.pi_ordering_cost_weight
            summary["cost_to_pi_ratio"] = 1.0 + summary["gap_to_pi_pct"] / 100.0
            ratio_std = (
                episodes.dropna(subset=["cost_to_pi_ratio"])
                .groupby("policy")["cost_to_pi_ratio"]
                .std(ddof=0)
            )
            summary["cost_to_pi_ratio_std"] = summary["policy"].map(ratio_std).fillna(0.0)
            episode_rows.append(episodes)
            rows.append(summary)
            print(f"{var}={value} done", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.concat(rows, ignore_index=True)
    episode_result = pd.concat(episode_rows, ignore_index=True)
    result.to_csv(out, index=False)
    episode_out = out.with_name(f"{out.stem}_episodes.csv")
    episode_result.to_csv(episode_out, index=False)
    if not args.no_plot:
        fig_dir = Path(args.fig_dir) if args.fig_dir else out.with_name("sensitivity_figures")
        formats = tuple(fmt.strip() for fmt in args.formats.split(",") if fmt.strip())
        saved = plot_sensitivity_figures(result, episode_result, fig_dir, formats=formats)
        print(f"saved figures: {fig_dir} ({len(saved)} files)")
    print(f"saved sensitivity summary: {out}")


if __name__ == "__main__":
    main()
