#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from rl_ato.config import load_config
from rl_ato.evaluate import benchmark_policies, compute_pi_breakdowns
from rl_ato.policies import DTPPolicy, NVDPolicy, SAABSOBCAOptimizedPolicy, tune_saa_bs_obca
from rl_ato.rlbr import RLBRPolicy
from rl_ato.scenario import ScenarioGenerator, make_instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RLBR against benchmark policies.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="outputs/rlbr.pt")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--pi-episodes", type=int, default=None)
    parser.add_argument("--pi-ordering-weight", type=float, default=None)
    parser.add_argument("--policies", default="RLBR,NVD,DTP")
    parser.add_argument("--dtp-known-scale", type=float, default=1.0)
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
    parser.add_argument("--episode-out", default="outputs/benchmark_episodes.csv")
    parser.add_argument("--summary-out", default="outputs/benchmark_summary.csv")
    args = parser.parse_args()

    exp, ppo, _sens = load_config(args.config)
    if args.episodes is not None:
        exp.eval_episodes = args.episodes
    if args.pi_episodes is not None:
        exp.pi_episodes = args.pi_episodes
    if args.pi_ordering_weight is not None:
        exp.pi_ordering_cost_weight = args.pi_ordering_weight
    instance = make_instance(exp, seed=exp.seed)
    generator = ScenarioGenerator(instance)
    scenarios = [generator.sample() for _ in range(exp.eval_episodes)]

    requested = {p.strip().upper() for p in args.policies.split(",") if p.strip()}
    policies = []
    if "RLBR" in requested:
        policy = RLBRPolicy(instance, ppo)
        policy.load(args.model)
        policies.append(policy)
    if "NVD" in requested:
        policies.append(NVDPolicy(instance))
    if "DTP" in requested:
        policies.append(DTPPolicy(instance, known_demand_scale=args.dtp_known_scale))
    if "SAA" in requested or "SAA-BS-OBCA" in requested or "SAABSOBCA" in requested:
        step_sizes = tuple(float(x) for x in args.saa_step_sizes.split(",") if x.strip())
        saa_solver = "gurobi"
        print(f"tuning SAA-BS-OBCA on {args.saa_train_episodes} training episodes...")
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
        np.save(Path(args.summary_out).with_suffix(".saa_base_stock.npy"), saa_result.base_stock)
        Path(args.summary_out).with_suffix(".saa_metadata.json").write_text(
            json.dumps(
                {
                    "policy": "SAA-BS-OBCA",
                    "train_episodes": saa_result.train_episodes,
                    "step_sizes": list(saa_result.step_sizes),
                    "beta_late": saa_result.beta_late,
                    "known_requirement_scale": saa_result.known_requirement_scale,
                    "allocation_solver": saa_solver,
                    "max_sweeps_per_delta": args.saa_max_sweeps,
                    "objective": saa_result.objective,
                    "starts_evaluated": saa_result.starts_evaluated,
                    "base_stock": saa_result.base_stock.tolist(),
                    "training_episode_offset": 1_000_000,
                    "master_seed": instance.seeds["master"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"SAA-BS-OBCA train objective={saa_result.objective:.3f}")

    pi_costs = None
    if exp.pi_episodes > 0:
        print(f"solving PI oracle for {min(exp.pi_episodes, len(scenarios))} episodes...")
        pi_breakdowns = compute_pi_breakdowns(
            instance,
            scenarios,
            limit=exp.pi_episodes,
            ordering_cost_weight=exp.pi_ordering_cost_weight,
        )
        pi_costs = np.full(len(scenarios), np.nan, dtype=float)
        pi_rows = []
        for idx, breakdown in enumerate(pi_breakdowns):
            if breakdown is None:
                continue
            pi_costs[idx] = breakdown.total
            pi_rows.append(
                {
                    "episode": idx,
                    "pi_cost": breakdown.total,
                    "pi_ordering_cost": breakdown.ordering,
                    "pi_holding_cost": breakdown.holding,
                    "pi_backlog_cost": breakdown.backlog,
                    "pi_raw_ordering_cost": breakdown.raw_ordering,
                    "pi_initial_kept": breakdown.initial_kept,
                    "pi_ordered_components": breakdown.ordered_components,
                    "pi_fulfilled_units": breakdown.fulfilled_units,
                    "pi_ordering_cost_weight": exp.pi_ordering_cost_weight,
                }
            )
        np.save(Path(args.summary_out).with_suffix(".pi.npy"), pi_costs)
        if pi_rows:
            import pandas as pd

            pd.DataFrame(pi_rows).to_csv(
                Path(args.summary_out).with_suffix(".pi_breakdown.csv"),
                index=False,
            )

    episodes, summary = benchmark_policies(policies, instance, scenarios, pi_costs=pi_costs)
    summary["pi_ordering_cost_weight"] = exp.pi_ordering_cost_weight
    Path(args.episode_out).parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.episode_out, index=False)
    summary.to_csv(args.summary_out, index=False)
    print(summary.to_string(index=False))
    print(f"saved episodes: {args.episode_out}")
    print(f"saved summary: {args.summary_out}")


if __name__ == "__main__":
    main()
