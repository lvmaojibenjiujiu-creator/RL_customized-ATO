from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from rl_ato.config import load_config
from rl_ato.evaluate import evaluate_policy
from rl_ato.policies import SAABSOBCAOptimizedPolicy
from rl_ato.scenario import ScenarioGenerator, make_instance
from rl_ato.spec_benchmarks import add_spec_benchmark_arguments, spec_benchmark_policies_from_request


def _summary(df: pd.DataFrame, pi_costs: np.ndarray | None = None) -> pd.DataFrame:
    valid_pi = np.zeros(len(df), dtype=bool)
    pi_mean = np.nan
    gap = np.nan
    if pi_costs is not None:
        pi_vals = df["episode"].map(
            {idx: float(cost) for idx, cost in enumerate(pi_costs) if np.isfinite(cost)}
        ).to_numpy(dtype=float)
        valid_pi = np.isfinite(pi_vals)
        if valid_pi.any():
            pi_mean = float(np.nanmean(pi_vals[valid_pi]))
            gap = (float(df.loc[valid_pi, "cost"].mean()) - pi_mean) / max(1e-9, pi_mean) * 100.0
    cost_se = float(df["cost"].std(ddof=1) / np.sqrt(len(df))) if len(df) > 1 else 0.0
    row = {
        "policy": str(df["policy"].iloc[0]) if len(df) else "RH-SPT",
        "episodes": int(len(df)),
        "cost_mean": float(df["cost"].mean()),
        "cost_std": float(df["cost"].std(ddof=0)),
        "cost_se": cost_se,
        "cost_ci95_low": float(df["cost"].mean()) - 1.96 * cost_se,
        "cost_ci95_high": float(df["cost"].mean()) + 1.96 * cost_se,
        "order_cost_mean": float(df["order_cost"].mean()),
        "holding_cost_mean": float(df["holding_cost"].mean()),
        "backlog_cost_mean": float(df["backlog_cost"].mean()),
        "gap_to_pi_pct": gap,
        "fill_rate": float(df["fill_rate"].mean()),
        "ontime_rate": float(df["ontime_rate"].mean()),
        "residual_inventory_ratio": float(
            df["residual_inventory_ratio"].mean()
            if "residual_inventory_ratio" in df
            else df["mismatch_rate"].mean()
        ),
        "mismatch_rate": float(df["mismatch_rate"].mean()),
        "pi_cost_mean": pi_mean,
        "runtime_seconds_mean": float(df["runtime_seconds"].mean()),
        "runtime_seconds_total": float(df["runtime_seconds"].sum()),
        "fallback_count": int(df["fallback_count"].sum()),
        "fallback_rate": float(df["fallback_count"].sum() / max(1, len(df))),
        "solver_status": str(df["solver_status"].iloc[-1]) if "solver_status" in df else "",
        "pi_ordering_cost_weight": np.nan,
    }
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed RH-SPT evaluation.")
    parser.add_argument("--config", default="configs/formal_tuned.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--episode-out", default="outputs/rh_spt_checkpoint_episodes.csv")
    parser.add_argument("--summary-out", default="outputs/rh_spt_checkpoint_summary.csv")
    parser.add_argument("--pi-costs-in", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--saa-base-stock-in", default=None)
    parser.add_argument("--saa-known-scale", type=float, default=0.0)
    parser.add_argument("--saa-beta-late", type=float, default=1.0)
    add_spec_benchmark_arguments(parser)
    args = parser.parse_args()

    exp, _ppo, _sens = load_config(args.config)
    if args.episodes is not None:
        exp.eval_episodes = args.episodes
    instance = make_instance(exp, seed=exp.seed)

    fallback = None
    if args.saa_base_stock_in:
        fallback = SAABSOBCAOptimizedPolicy(
            instance,
            np.load(args.saa_base_stock_in),
            beta_late=args.saa_beta_late,
            allocation_solver="gurobi",
            known_requirement_scale=args.saa_known_scale,
        )

    policies = spec_benchmark_policies_from_request(
        ["RH-SPT"],
        args,
        instance,
        seed=instance.seeds["master"],
        fallback_policy=fallback,
    )
    policy = policies[0]

    episode_out = Path(args.episode_out)
    summary_out = Path(args.summary_out)
    episode_out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    completed: set[int] = set()
    if args.resume and episode_out.exists():
        existing = pd.read_csv(episode_out)
        rows.append(existing)
        completed = {int(x) for x in existing["episode"].unique()}
        print(f"resuming from {episode_out}, completed={len(completed)}", flush=True)

    generator = ScenarioGenerator(instance)
    pi_costs = np.load(args.pi_costs_in) if args.pi_costs_in else None
    for episode in range(exp.eval_episodes):
        if episode in completed:
            continue
        scenario = generator.sample(episode=episode)
        df = evaluate_policy(policy, instance, [scenario])
        df["episode"] = episode
        df["episode_id"] = int(getattr(scenario, "episode", episode))
        rows.append(df)
        all_rows = pd.concat(rows, ignore_index=True)
        all_rows.to_csv(episode_out, index=False)
        summary = _summary(all_rows, pi_costs=pi_costs)
        summary["pi_ordering_cost_weight"] = exp.pi_ordering_cost_weight
        summary.to_csv(summary_out, index=False)
        latest = df.iloc[0]
        print(
            f"episode {episode + 1}/{exp.eval_episodes} "
            f"cost={latest['cost']:.3f} runtime={latest['runtime_seconds']:.2f}s "
            f"fallback={int(latest['fallback_count'])}",
            flush=True,
        )

    result = pd.concat(rows, ignore_index=True)
    summary = _summary(result, pi_costs=pi_costs)
    summary["pi_ordering_cost_weight"] = exp.pi_ordering_cost_weight
    summary.to_csv(summary_out, index=False)
    print(summary.to_string(index=False))
    print(f"saved episodes: {episode_out}")
    print(f"saved summary: {summary_out}")


if __name__ == "__main__":
    main()
