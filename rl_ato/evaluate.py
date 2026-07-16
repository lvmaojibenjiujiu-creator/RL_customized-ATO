from __future__ import annotations

import time
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from benchmarks.base import BasePolicy
from benchmarks.pi import PICostBreakdown, perfect_information_breakdown

from .env import ATOEnv
from .scenario import ProblemInstance, Scenario


def evaluate_policy(
    policy: BasePolicy,
    instance: ProblemInstance,
    scenarios: Iterable[Scenario],
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for index, scenario in enumerate(scenarios):
        started = time.perf_counter()
        env = ATOEnv(instance)
        obs = env.reset(scenario)
        done = False
        while not done:
            action = policy.act(env, obs)
            next_obs, _reward, done, _information = env.step(action)
            if next_obs is not None:
                obs = next_obs
        row: Dict[str, float | int | str] = dict(env.metrics())
        row["episode"] = int(index)
        row["episode_id"] = int(getattr(scenario, "episode", index))
        row["policy"] = str(policy.name)
        row["runtime_seconds"] = float(time.perf_counter() - started)
        row["solver_status"] = str(getattr(policy, "last_solver_status", ""))
        row["solver_gap"] = float(getattr(policy, "last_solver_gap", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_pi_breakdowns(
    instance: ProblemInstance,
    scenarios: Sequence[Scenario],
) -> List[PICostBreakdown]:
    return [perfect_information_breakdown(instance, scenario) for scenario in scenarios]


def benchmark_policies(
    policies: Iterable[BasePolicy],
    instance: ProblemInstance,
    scenarios: Sequence[Scenario],
    pi_breakdowns: Sequence[PICostBreakdown],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(pi_breakdowns) != len(scenarios):
        raise ValueError("PI and policy evaluations must use the same scenario paths")
    pi_costs = np.asarray([item.total for item in pi_breakdowns], dtype=float)
    pi_mean = float(pi_costs.mean())
    episode_frames = []
    summaries = []
    for policy in policies:
        frame = evaluate_policy(policy, instance, scenarios)
        episode_frames.append(frame)
        costs = frame["cost"].to_numpy(dtype=float)
        standard_error = float(costs.std(ddof=1) / np.sqrt(len(costs))) if len(costs) > 1 else 0.0
        summaries.append(
            {
                "policy": policy.name,
                "episodes": int(len(frame)),
                "cost_mean": float(costs.mean()),
                "cost_std": float(costs.std(ddof=0)),
                "cost_se": standard_error,
                "cost_ci95_low": float(costs.mean() - 1.96 * standard_error),
                "cost_ci95_high": float(costs.mean() + 1.96 * standard_error),
                "order_cost_mean": float(frame["order_cost"].mean()),
                "holding_cost_mean": float(frame["holding_cost"].mean()),
                "backlog_cost_mean": float(frame["backlog_cost"].mean()),
                "pi_cost_mean": pi_mean,
                "gap_to_pi": float((costs.mean() - pi_mean) / pi_mean),
                "fill_rate": float(frame["fill_rate"].mean()),
                "ontime_rate": float(frame["ontime_rate"].mean()),
                "mismatch_rate": float(frame["mismatch_rate"].mean()),
                "runtime_seconds_mean": float(frame["runtime_seconds"].mean()),
            }
        )
    return pd.concat(episode_frames, ignore_index=True), pd.DataFrame(summaries)
