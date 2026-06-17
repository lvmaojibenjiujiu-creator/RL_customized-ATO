from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from rl_ato.config import load_config
from rl_ato.evaluate import evaluate_policy
from rl_ato.policies import DTPPolicy, NVDPolicy, SAABSOBCAOptimizedPolicy
from rl_ato.rlbr import RLBRPolicy
from rl_ato.scenario import ScenarioGenerator, make_instance


TABLE4_COMBOS = [
    ("poisson", 5, 15),
    ("poisson", 10, 20),
    ("poisson", 20, 100),
    ("seasonal", 5, 15),
    ("seasonal", 10, 20),
    ("seasonal", 20, 100),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect same-scenario runtime for Table 4 policies without replacing performance metrics."
    )
    parser.add_argument("--config", default="configs/formal_tuned.yaml")
    parser.add_argument("--artifact-dir", default="outputs/paper_official_table4_preview")
    parser.add_argument("--out-dir", default="outputs/revision_v6")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--dtp-known-scale", type=float, default=1.0)
    parser.add_argument("--saa-beta-late", type=float, default=1.0)
    parser.add_argument("--saa-known-scale", type=float, default=0.0)
    parser.add_argument("--saa-allocation-solver", default="gurobi")
    args = parser.parse_args()

    exp, ppo, _sens = load_config(args.config)
    artifact_dir = Path(args.artifact_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    episode_rows = []
    for pattern, n_products, n_components in TABLE4_COMBOS:
        cfg = replace(
            exp,
            demand_pattern=pattern,
            products=n_products,
            components=n_components,
            eval_episodes=args.episodes,
            pi_episodes=args.episodes,
        )
        seed_offset = 100 * n_products + n_components + (0 if pattern == "poisson" else 5000)
        instance = make_instance(cfg, seed=cfg.seed + seed_offset)
        generator = ScenarioGenerator(instance)
        scenarios = [generator.sample() for _ in range(cfg.eval_episodes)]

        rlbr = RLBRPolicy(instance, ppo)
        rlbr.load(str(artifact_dir / f"rlbr_{pattern}_{n_products}x{n_components}.pt"))
        base_stock = np.load(artifact_dir / f"saa_base_stock_{pattern}_{n_products}x{n_components}.npy")
        policies = [
            rlbr,
            NVDPolicy(instance),
            DTPPolicy(instance, known_demand_scale=args.dtp_known_scale),
            SAABSOBCAOptimizedPolicy(
                instance,
                base_stock,
                beta_late=args.saa_beta_late,
                allocation_solver=args.saa_allocation_solver,
                known_requirement_scale=args.saa_known_scale,
            ),
        ]

        for policy in policies:
            df = evaluate_policy(policy, instance, scenarios)
            display_policy = "SAA-OBCA" if policy.name in {"SAA-BS-OBCA", "SAA-OBCA"} else policy.name
            df = df.copy()
            df["demand_pattern"] = pattern
            df["scale"] = f"{n_products}/{n_components}"
            df["policy"] = display_policy
            df["policy_name"] = display_policy
            episode_rows.append(df)
            row = {
                "demand_pattern": pattern,
                "scale": f"{n_products}/{n_components}",
                "policy": display_policy,
                "episodes": len(df),
                "runtime_seconds_mean": float(df["runtime_seconds"].mean()),
                "runtime_seconds_total": float(df["runtime_seconds"].sum()),
            }
            rows.append(row)
            print(
                f"{pattern} {n_products}/{n_components} {display_policy}: "
                f"{row['runtime_seconds_mean']:.4f}s/episode",
                flush=True,
            )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table4_runtime_existing_policies.csv", index=False)
    episodes = pd.concat(episode_rows, ignore_index=True) if episode_rows else pd.DataFrame()
    episodes.to_csv(out_dir / "table4_existing_policy_episodes.csv", index=False)
    print(f"saved: {out_dir / 'table4_existing_policy_episodes.csv'}")
    print(f"saved: {out_dir / 'table4_runtime_existing_policies.csv'}")


if __name__ == "__main__":
    main()
