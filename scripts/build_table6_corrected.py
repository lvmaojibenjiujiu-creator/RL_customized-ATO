from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POLICY_ORDER = ["RLBR", "NVD", "DTP", "RH-SPT", "SAA-OBCA", "DHP"]
PATTERN_ORDER = ["poisson", "seasonal"]
SCALE_ORDER = ["5/15", "10/20", "20/100"]
T_CRIT_975 = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
    10: 2.2281,
    11: 2.2010,
    12: 2.1788,
    13: 2.1604,
    14: 2.1448,
    15: 2.1314,
    16: 2.1199,
    17: 2.1098,
    18: 2.1009,
    19: 2.0930,
    20: 2.0860,
    25: 2.0595,
    30: 2.0423,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corrected Table 6 episodes and summary with 95% CIs.")
    parser.add_argument("--legacy-episode-dir", default="outputs/paper_official_table4_preview")
    parser.add_argument("--existing-runtime-episodes", default="outputs/revision_v6/table4_existing_policy_episodes.csv")
    parser.add_argument("--added-episodes", default="outputs/revision_v6/table4_added_episodes.csv")
    parser.add_argument("--table4-source", default="outputs/revision_v6/table4_summary_with_rh_spt_dhp.csv")
    parser.add_argument("--out-dir", default="outputs/revision_v6")
    parser.add_argument("--horizon", type=int, default=40)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table4 = pd.read_csv(args.table4_source)
    pi_lookup = _pi_lookup(table4)

    frames = []
    runtime_lookup = _runtime_lookup(Path(args.existing_runtime_episodes))
    legacy = _load_legacy_table4_episodes(Path(args.legacy_episode_dir), args.horizon)
    if not legacy.empty:
        frames.append(_attach_runtime(legacy, runtime_lookup))
    added_path = Path(args.added_episodes)
    if added_path.exists():
        frames.append(_normalize_episode_frame(pd.read_csv(added_path), args.horizon, corrected=False))
    if not frames:
        raise FileNotFoundError("No episode files were available for corrected Table 6.")

    episodes = pd.concat(frames, ignore_index=True, sort=False)
    for idx, row in episodes.iterrows():
        key = (row["demand_pattern"], row["scale"])
        if pd.isna(row.get("pi_cost_mean", np.nan)):
            episodes.at[idx, "pi_cost_mean"] = pi_lookup.get(key, np.nan)

    episodes = episodes.sort_values(
        by=["demand_pattern", "scale", "policy", "episode"],
        key=lambda col: col.map(_sort_key) if col.name in {"demand_pattern", "scale", "policy"} else col,
    )
    episodes.to_csv(out_dir / "table6_corrected_episodes.csv", index=False)

    summary = _summarize(episodes)
    summary.to_csv(out_dir / "table6_corrected_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"saved: {out_dir / 'table6_corrected_episodes.csv'}")
    print(f"saved: {out_dir / 'table6_corrected_summary.csv'}")


def _normalize_policy(policy: str) -> str:
    mapping = {"SAA-BS-OBCA": "SAA-OBCA", "DHP-SBR": "DHP"}
    return mapping.get(str(policy), str(policy))


def _load_legacy_table4_episodes(episode_dir: Path, horizon: int) -> pd.DataFrame:
    frames = []
    for pattern in PATTERN_ORDER:
        for scale in SCALE_ORDER:
            n_products, n_components = scale.split("/")
            path = episode_dir / f"episodes_{pattern}_{n_products}x{n_components}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["demand_pattern"] = pattern
            df["scale"] = scale
            frames.append(_normalize_episode_frame(df, horizon, corrected=False))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _normalize_episode_frame(df: pd.DataFrame, horizon: int, corrected: bool) -> pd.DataFrame:
    result = df.copy()
    result["policy"] = result["policy"].map(_normalize_policy)
    if "policy_name" in result:
        result["policy_name"] = result["policy"]
    if "residual_inventory_ratio" not in result:
        if corrected:
            result["residual_inventory_ratio"] = result["mismatch_rate"]
        else:
            result["residual_inventory_ratio"] = result["mismatch_rate"] * float(horizon)
    elif not corrected and "mismatch_rate" in result:
        legacy_mask = result["residual_inventory_ratio"].astype(float).abs() < 0.25
        result.loc[legacy_mask, "residual_inventory_ratio"] = (
            result.loc[legacy_mask, "mismatch_rate"].astype(float) * float(horizon)
        )
    result["mismatch_rate"] = result["residual_inventory_ratio"]
    if "runtime_seconds" not in result:
        result["runtime_seconds"] = np.nan
    if "fallback_count" not in result:
        result["fallback_count"] = 0
    return result


def _runtime_lookup(path: Path) -> dict[tuple[str, str, str, int], float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "runtime_seconds" not in df:
        return {}
    df = df.copy()
    df["policy"] = df["policy"].map(_normalize_policy)
    out = {}
    for row in df.itertuples(index=False):
        out[(str(row.demand_pattern), str(row.scale), str(row.policy), int(row.episode))] = float(row.runtime_seconds)
    return out


def _attach_runtime(df: pd.DataFrame, runtime_lookup: dict[tuple[str, str, str, int], float]) -> pd.DataFrame:
    result = df.copy()
    runtime = []
    for row in result.itertuples(index=False):
        runtime.append(runtime_lookup.get((str(row.demand_pattern), str(row.scale), str(row.policy), int(row.episode)), np.nan))
    result["runtime_seconds"] = runtime
    return result


def _pi_lookup(summary: pd.DataFrame) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for (pattern, scale), group in summary.groupby(["demand_pattern", "scale"]):
        vals = group["pi_cost_mean"].dropna()
        out[(str(pattern), str(scale))] = float(vals.iloc[0]) if len(vals) else np.nan
    return out


def _summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pattern, scale, policy), grp in episodes.groupby(["demand_pattern", "scale", "policy"], sort=False):
        pi_mean = float(grp["pi_cost_mean"].dropna().iloc[0]) if grp["pi_cost_mean"].notna().any() else np.nan
        pi_gap_episode = (grp["cost"].astype(float) - pi_mean) / pi_mean if pi_mean > 1e-9 else np.nan
        runtime = grp["runtime_seconds"].astype(float)
        row = {
            "demand_pattern": pattern,
            "scale": scale,
            "policy": policy,
            "episodes": int(grp["episode"].nunique()),
            "cost_mean": float(grp["cost"].mean()),
            "cost_std": float(grp["cost"].std(ddof=0)),
            "pi_cost_mean": pi_mean,
            "pi_gap": float(np.nanmean(pi_gap_episode)),
            "pi_gap_ci95_low": _ci_low(pi_gap_episode),
            "pi_gap_ci95_high": _ci_high(pi_gap_episode),
            "fill_rate": float(grp["fill_rate"].mean()),
            "fill_rate_ci95_low": _ci_low(grp["fill_rate"]),
            "fill_rate_ci95_high": _ci_high(grp["fill_rate"]),
            "ontime_rate": float(grp["ontime_rate"].mean()),
            "ontime_rate_ci95_low": _ci_low(grp["ontime_rate"]),
            "ontime_rate_ci95_high": _ci_high(grp["ontime_rate"]),
            "residual_inventory_ratio": float(grp["residual_inventory_ratio"].mean()),
            "residual_inventory_ratio_ci95_low": _ci_low(grp["residual_inventory_ratio"]),
            "residual_inventory_ratio_ci95_high": _ci_high(grp["residual_inventory_ratio"]),
            "mismatch_rate": float(grp["residual_inventory_ratio"].mean()),
            "runtime_seconds_mean": float(runtime.mean()) if runtime.notna().any() else np.nan,
            "runtime_seconds_ci95_low": _ci_low(runtime.dropna()),
            "runtime_seconds_ci95_high": _ci_high(runtime.dropna()),
            "fallback_count": int(grp.get("fallback_count", pd.Series(dtype=float)).fillna(0).sum()),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["demand_pattern"] = pd.Categorical(summary["demand_pattern"], PATTERN_ORDER, ordered=True)
    summary["scale"] = pd.Categorical(summary["scale"], SCALE_ORDER, ordered=True)
    summary["policy"] = pd.Categorical(summary["policy"], POLICY_ORDER, ordered=True)
    return summary.sort_values(["demand_pattern", "scale", "policy"]).astype(
        {"demand_pattern": str, "scale": str, "policy": str}
    )


def _ci_low(values: pd.Series | np.ndarray) -> float:
    mean, half_width = _mean_ci(values)
    return float(mean - half_width) if np.isfinite(mean) else np.nan


def _ci_high(values: pd.Series | np.ndarray) -> float:
    mean, half_width = _mean_ci(values)
    return float(mean + half_width) if np.isfinite(mean) else np.nan


def _mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    crit = _tcrit(arr.size - 1)
    return mean, crit * se


def _tcrit(df: int) -> float:
    if df in T_CRIT_975:
        return T_CRIT_975[df]
    if df < 25:
        return T_CRIT_975[20]
    if df < 30:
        return T_CRIT_975[25]
    return 1.96


def _sort_key(value):
    if value in PATTERN_ORDER:
        return PATTERN_ORDER.index(value)
    if value in SCALE_ORDER:
        return SCALE_ORDER.index(value)
    if value in POLICY_ORDER:
        return POLICY_ORDER.index(value)
    return 999


if __name__ == "__main__":
    main()
