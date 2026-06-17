from __future__ import annotations

import time
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
        started = time.perf_counter()
        fallback_before = int(getattr(policy, "fallback_count", 0))
        env = ATOEnv(instance)
        obs = env.reset(scenario)
        done = False
        while not done:
            action = policy.act(env, obs)
            next_obs, _reward, done, _info = env.step(action)
            if next_obs is not None:
                obs = next_obs
        metrics = env.metrics()
        fallback_after = int(getattr(policy, "fallback_count", fallback_before))
        runtime_seconds = float(time.perf_counter() - started)
        metrics["episode"] = episode
        metrics["policy"] = policy.name
        metrics["policy_name"] = policy.name
        metrics["episode_id"] = int(getattr(scenario, "episode", episode))
        metrics["total_cost"] = metrics["cost"]
        metrics["ordering_cost"] = metrics.get("order_cost", np.nan)
        metrics["backlog_or_tardiness_cost"] = metrics.get("backlog_cost", np.nan)
        metrics["on_time_rate"] = metrics.get("ontime_rate", np.nan)
        metrics["runtime_seconds"] = runtime_seconds
        metrics["fallback_count"] = max(0, fallback_after - fallback_before)
        metrics["solver_status"] = str(getattr(policy, "last_solver_status", ""))
        metrics["solver_gap"] = float(getattr(policy, "last_solver_gap", np.nan))
        table_path = str(getattr(policy, "table_path", ""))
        metrics["h3_table_path"] = table_path if policy.name == "H3-SBR" else ""
        metrics["dhp_table_path"] = table_path if policy.name == "DHP-SBR" else ""
        metrics["table_path"] = table_path
        metrics["calibration_mode"] = str(getattr(policy, "calibration_mode", ""))
        metrics["calibration_runtime_seconds"] = float(getattr(policy, "calibration_runtime_seconds", np.nan))
        metrics["n_calibration_edges"] = float(getattr(policy, "n_calibration_edges", np.nan))
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
        cost_std_sample = float(df["cost"].std(ddof=1)) if len(df) > 1 else 0.0
        cost_se = cost_std_sample / float(np.sqrt(len(df))) if len(df) > 1 else 0.0
        gap = np.nan
        if valid_pi.any() and pi_costs is not None and pi_mean > 1e-9:
            gap = (float(costs[valid_pi].mean()) - pi_mean) / pi_mean * 100.0
        row = {
                "policy": policy.name,
                "episodes": len(df),
                "cost_mean": float(df["cost"].mean()),
                "cost_std": float(df["cost"].std(ddof=0)),
                "cost_se": cost_se,
                "cost_ci95_low": float(df["cost"].mean()) - 1.96 * cost_se,
                "cost_ci95_high": float(df["cost"].mean()) + 1.96 * cost_se,
                "order_cost_mean": float(df["order_cost"].mean()) if "order_cost" in df else np.nan,
                "holding_cost_mean": float(df["holding_cost"].mean()) if "holding_cost" in df else np.nan,
                "backlog_cost_mean": float(df["backlog_cost"].mean()) if "backlog_cost" in df else np.nan,
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
        }
        if "runtime_seconds" in df:
            row["runtime_seconds_mean"] = float(df["runtime_seconds"].mean())
            row["runtime_seconds_total"] = float(df["runtime_seconds"].sum())
        if "fallback_count" in df:
            row["fallback_count"] = int(df["fallback_count"].sum())
            row["fallback_rate"] = float(df["fallback_count"].sum() / max(1, len(df)))
        if "solver_status" in df:
            nonempty = df["solver_status"].astype(str)
            nonempty = nonempty[nonempty != ""]
            row["solver_status"] = str(nonempty.iloc[-1]) if len(nonempty) else ""
        if "h3_table_path" in df:
            nonempty_path = df["h3_table_path"].astype(str)
            nonempty_path = nonempty_path[nonempty_path != ""]
            row["h3_table_path"] = str(nonempty_path.iloc[-1]) if len(nonempty_path) else ""
        if "dhp_table_path" in df:
            nonempty_path = df["dhp_table_path"].astype(str)
            nonempty_path = nonempty_path[nonempty_path != ""]
            row["dhp_table_path"] = str(nonempty_path.iloc[-1]) if len(nonempty_path) else ""
        if "table_path" in df:
            nonempty_path = df["table_path"].astype(str)
            nonempty_path = nonempty_path[nonempty_path != ""]
            row["table_path"] = str(nonempty_path.iloc[-1]) if len(nonempty_path) else ""
        if "calibration_mode" in df:
            modes = df["calibration_mode"].astype(str)
            modes = modes[modes != ""]
            row["calibration_mode"] = str(modes.iloc[-1]) if len(modes) else ""
        if "calibration_runtime_seconds" in df:
            vals = df["calibration_runtime_seconds"].dropna()
            row["calibration_runtime_seconds"] = float(vals.iloc[0]) if len(vals) else np.nan
        if "n_calibration_edges" in df:
            vals = df["n_calibration_edges"].dropna()
            row["n_calibration_edges"] = int(vals.iloc[0]) if len(vals) else 0
        summary_rows.append(row)
    return pd.concat(episode_rows, ignore_index=True), pd.DataFrame(summary_rows)


def paired_policy_comparison(
    episode_df: pd.DataFrame,
    reference_policy: str,
    metric: str = "cost",
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    if episode_df.empty or metric not in episode_df:
        return pd.DataFrame()
    ref = episode_df[episode_df["policy"] == reference_policy].set_index("episode")[metric]
    if ref.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for policy, group in episode_df.groupby("policy"):
        if policy == reference_policy:
            continue
        cur = group.set_index("episode")[metric]
        common = np.asarray(sorted(set(ref.index).intersection(cur.index)), dtype=int)
        if common.size == 0:
            continue
        diff = cur.loc[common].to_numpy(dtype=float) - ref.loc[common].to_numpy(dtype=float)
        boot_means = np.empty(int(n_bootstrap), dtype=float)
        for idx in range(int(n_bootstrap)):
            sample = rng.integers(0, diff.size, size=diff.size)
            boot_means[idx] = float(np.mean(diff[sample]))
        ref_mean = float(ref.loc[common].mean())
        rows.append(
            {
                "policy": policy,
                "reference_policy": reference_policy,
                "episodes": int(common.size),
                f"{metric}_mean_minus_reference": float(np.mean(diff)),
                f"{metric}_pct_minus_reference": 100.0 * float(np.mean(diff)) / max(1e-9, ref_mean),
                "bootstrap_ci95_low": float(np.quantile(boot_means, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(boot_means, 0.975)),
            }
        )
    return pd.DataFrame(rows)
