from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .env import ATOEnv
from .pi_oracle import PICostBreakdown, perfect_information_breakdown, perfect_information_cost
from .policies import BasePolicy
from .scenario import ProblemInstance, Scenario


def evaluate_policy(policy: BasePolicy, instance: ProblemInstance, scenarios: Iterable[Scenario]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for episode, scenario in enumerate(scenarios):
        env = ATOEnv(instance)
        obs = env.reset(scenario)
        done = False
        while not done:
            action = policy.act(env, obs)
            next_obs, _reward, done, _info = env.step(action)
            if next_obs is not None:
                obs = next_obs
        metrics = env.metrics()
        metrics["episode"] = episode
        metrics["policy"] = policy.name
        rows.append(metrics)
    return pd.DataFrame(rows)


def compute_pi_costs(
    instance: ProblemInstance,
    scenarios: List[Scenario],
    limit: int | None = None,
    ordering_cost_weight: float = 1.0,
) -> np.ndarray:
    n = len(scenarios) if limit is None else min(limit, len(scenarios))
    costs = np.full(len(scenarios), np.nan, dtype=float)
    for idx in range(n):
        costs[idx] = perfect_information_cost(
            instance,
            scenarios[idx],
            ordering_cost_weight=ordering_cost_weight,
        )
    return costs


def compute_pi_breakdowns(
    instance: ProblemInstance,
    scenarios: List[Scenario],
    limit: int | None = None,
    ordering_cost_weight: float = 1.0,
) -> List[PICostBreakdown | None]:
    n = len(scenarios) if limit is None else min(limit, len(scenarios))
    out: List[PICostBreakdown | None] = [None for _ in scenarios]
    for idx in range(n):
        out[idx] = perfect_information_breakdown(
            instance,
            scenarios[idx],
            ordering_cost_weight=ordering_cost_weight,
        )
    return out


def benchmark_policies(
    policies: Iterable[BasePolicy],
    instance: ProblemInstance,
    scenarios: List[Scenario],
    pi_costs: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows = []
    summary_rows = []
    valid_pi = np.isfinite(pi_costs) if pi_costs is not None else np.zeros(len(scenarios), dtype=bool)
    pi_mean = float(np.nanmean(pi_costs[valid_pi])) if valid_pi.any() and pi_costs is not None else np.nan
    for policy in policies:
        df = evaluate_policy(policy, instance, scenarios)
        episode_rows.append(df)
        costs = df["cost"].to_numpy()
        gap = np.nan
        if valid_pi.any() and pi_costs is not None and pi_mean > 1e-9:
            gap = (float(costs[valid_pi].mean()) - pi_mean) / pi_mean * 100.0
        summary_rows.append(
            {
                "policy": policy.name,
                "episodes": len(df),
                "cost_mean": float(df["cost"].mean()),
                "cost_std": float(df["cost"].std(ddof=0)),
                "order_cost_mean": float(df["order_cost"].mean()) if "order_cost" in df else np.nan,
                "holding_cost_mean": float(df["holding_cost"].mean()) if "holding_cost" in df else np.nan,
                "backlog_cost_mean": float(df["backlog_cost"].mean()) if "backlog_cost" in df else np.nan,
                "gap_to_pi_pct": gap,
                "fill_rate": float(df["fill_rate"].mean()),
                "ontime_rate": float(df["ontime_rate"].mean()),
                "mismatch_rate": float(df["mismatch_rate"].mean()),
                "pi_cost_mean": pi_mean,
            }
        )
    return pd.concat(episode_rows, ignore_index=True), pd.DataFrame(summary_rows)
