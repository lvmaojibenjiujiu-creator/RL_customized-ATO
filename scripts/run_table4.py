from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from rl_ato.config import load_config
from rl_ato.evaluate import benchmark_policies, compute_pi_costs
from rl_ato.mpc_policies import add_mpc_arguments, mpc_policies_from_request
from rl_ato.policies import DTPPolicy, NVDPolicy, SAABSOBCAOptimizedPolicy, tune_saa_bs_obca
from rl_ato.rlbr import RLBRPolicy
from rl_ato.scenario import ScenarioGenerator, make_instance
from rl_ato.spec_benchmarks import add_spec_benchmark_arguments, spec_benchmark_policies_from_request
from rl_ato.train import train_rlbr


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the Table 4 scale/demand comparison.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--pi-episodes", type=int, default=None)
    parser.add_argument("--pi-ordering-weight", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--regularization-samples", type=int, default=None)
    parser.add_argument("--rollout-episodes", type=int, default=None)
    parser.add_argument("--scales", default="5x15,10x20,20x100")
    parser.add_argument("--patterns", default="poisson,seasonal")
    parser.add_argument("--dtp-known-scale", type=float, default=1.0)
    parser.add_argument("--include-saa-bs-obca", action="store_true")
    parser.add_argument("--include-ce-mpc", action="store_true")
    parser.add_argument("--include-rh-saa-mpc", action="store_true")
    parser.add_argument("--include-mpc-benchmarks", action="store_true")
    parser.add_argument("--include-rh-spt", action="store_true")
    parser.add_argument("--include-h3-sbr", action="store_true")
    parser.add_argument("--include-dhp-sbr", action="store_true")
    parser.add_argument("--include-spec-benchmarks", action="store_true")
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
    parser.add_argument("--out-dir", default="outputs/table4")
    add_mpc_arguments(parser)
    add_spec_benchmark_arguments(parser)
    args = parser.parse_args()

    base_exp, ppo, _sens = load_config(args.config)
    if args.epochs is not None:
        ppo.epochs = args.epochs
    if args.regularization_samples is not None:
        ppo.regularization_samples = args.regularization_samples
    if args.rollout_episodes is not None:
        ppo.rollout_episodes = args.rollout_episodes
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    scales = []
    for item in args.scales.split(","):
        left, right = item.lower().split("x")
        scales.append((int(left), int(right)))
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]

    for pattern in patterns:
        for I, J in scales:
            cfg = replace(base_exp, demand_pattern=pattern, products=I, components=J)
            if args.train_episodes is not None:
                cfg.train_episodes = args.train_episodes
            if args.eval_episodes is not None:
                cfg.eval_episodes = args.eval_episodes
            if args.pi_episodes is not None:
                cfg.pi_episodes = args.pi_episodes
            if args.pi_ordering_weight is not None:
                cfg.pi_ordering_cost_weight = args.pi_ordering_weight
            seed_offset = 100 * I + J + (0 if pattern == "poisson" else 5000)
            instance = make_instance(cfg, seed=cfg.seed + seed_offset)
            print(f"training RLBR for {pattern} {I}/{J}", flush=True)
            policy, history = train_rlbr(instance, cfg, ppo, progress=True)
            model_path = out_dir / f"rlbr_{pattern}_{I}x{J}.pt"
            policy.save(str(model_path))
            pd.DataFrame(history).to_csv(out_dir / f"history_{pattern}_{I}x{J}.csv", index=False)

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
            policies = [policy, NVDPolicy(instance), DTPPolicy(instance, known_demand_scale=args.dtp_known_scale)]
            fallback_for_spec = None
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
                npy_path = out_dir / f"saa_base_stock_{pattern}_{I}x{J}.npy"
                import numpy as np

                np.save(npy_path, saa_result.base_stock)
                fallback_for_spec = SAABSOBCAOptimizedPolicy(
                    instance,
                    saa_result.base_stock,
                    beta_late=args.saa_beta_late,
                    allocation_solver=saa_solver,
                    known_requirement_scale=args.saa_known_scale,
                )
                policies.append(fallback_for_spec)
            mpc_requested = []
            if args.include_ce_mpc or args.include_mpc_benchmarks:
                mpc_requested.append("CE-MPC")
            if args.include_rh_saa_mpc or args.include_mpc_benchmarks:
                mpc_requested.append("RH-SAA-MPC")
            policies.extend(mpc_policies_from_request(mpc_requested, args, seed=instance.seeds["master"]))
            spec_requested = []
            if args.include_rh_spt or args.include_spec_benchmarks:
                spec_requested.append("RH-SPT")
            if args.include_h3_sbr or args.include_spec_benchmarks:
                spec_requested.append("H3-SBR")
            if args.include_dhp_sbr or args.include_spec_benchmarks:
                spec_requested.append("DHP-SBR")
            policies.extend(
                spec_benchmark_policies_from_request(
                    spec_requested,
                    args,
                    instance,
                    seed=instance.seeds["master"],
                    fallback_policy=fallback_for_spec,
                )
            )
            episodes, summary = benchmark_policies(
                policies,
                instance,
                scenarios,
                pi_costs=pi_costs,
            )
            summary["demand_pattern"] = pattern
            summary["scale"] = f"{I}/{J}"
            summary["pi_ordering_cost_weight"] = cfg.pi_ordering_cost_weight
            episodes["demand_pattern"] = pattern
            episodes["scale"] = f"{I}/{J}"
            episodes.to_csv(out_dir / f"episodes_{pattern}_{I}x{J}.csv", index=False)
            summaries.append(summary)

    table = pd.concat(summaries, ignore_index=True)
    table.to_csv(out_dir / "table4_summary.csv", index=False)
    print(table.to_string(index=False))
    print(f"saved: {out_dir / 'table4_summary.csv'}")


if __name__ == "__main__":
    main()
