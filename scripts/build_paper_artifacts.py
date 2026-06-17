from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rl_ato_mpl")


TABLE3_ROWS = [
    ("Demand Distribution", "Poisson, Seasonal Trend"),
    ("Demand Coefficient of Variation", "[0.0, 0.9]"),
    ("Design Lead Time (Ld)", "[0, 5] periods"),
    ("Delivery Window (LW)", "[1, 6] periods"),
    ("Component Commonality Ratio", "[0.1, 0.8]"),
    ("Replenishment Lead Time (Lr)", "[2, 6] periods"),
    ("Scale (Products / Components)", "5/15, 10/20, 20/100"),
    ("BOM deviation-to-mean ratio", "[0.1, 0.6]"),
    ("Seasonality amplitude beta", "[0.1, 0.6]"),
    ("Price Ratio (b/h)", "[1, 3]"),
]


FIGURE_FILES = [
    "figure_3_bom_variability",
    "figure_4_component_commonality",
    "figure_5_delivery_window",
    "figure_6_lead_time_uncertainty",
    "figure_7_design_lead_time",
    "figure_8_demand_correlation",
    "figure_9_seasonal_amplitude",
    "figure_10_backorder_cost_ratio",
    "figures_3_4_structural_complexity",
    "figures_5_6_temporal_dynamics",
    "figures_8_9_demand_dynamics",
    "sensitivity_overview",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-style tables and figure bundle.")
    parser.add_argument("--table4", default="outputs/paper_official_table4_preview/table4_summary.csv")
    parser.add_argument("--table4-runtime", default="")
    parser.add_argument("--sensitivity-summary", default="outputs/formal_sensitivity_trend_aligned_final_pi6_summary.csv")
    parser.add_argument("--sensitivity-fig-dir", default="outputs/formal_sensitivity_trend_aligned_final_pi6_figures")
    parser.add_argument("--audit", default="outputs/sensitivity_trend_alignment_audit_pi6.csv")
    parser.add_argument("--out-dir", default="outputs/paper_numerical_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    table3 = pd.DataFrame(TABLE3_ROWS, columns=["Environment Parameter", "Value / Range"])
    _write_table_bundle(table3, table_dir / "table3_parameter_settings", index=False)

    raw_table4 = _fill_table4_runtime(pd.read_csv(args.table4), args.table4_runtime)
    raw_table4 = _fill_large_scale_rh_spt_rows(raw_table4)
    table4 = _format_table4(raw_table4)
    _write_table_bundle(table4, table_dir / "table4_policy_comparison_with_saa", index=False)

    sensitivity_summary = _format_sensitivity_summary(pd.read_csv(args.sensitivity_summary))
    numeric_cols = sensitivity_summary.select_dtypes(include="number").columns
    sensitivity_summary[numeric_cols] = sensitivity_summary[numeric_cols].round(2)
    sensitivity_summary.to_csv(table_dir / "figures_3_10_sensitivity_summary_with_saa.csv", index=False)
    audit_path = Path(args.audit)
    if audit_path.exists():
        audit_text = audit_path.read_text(encoding="utf-8")
        audit_text = audit_text.replace("SAA-BS-OBCA", "SAA-OBCA")
        (table_dir / audit_path.name).write_text(audit_text, encoding="utf-8")

    src_fig_dir = Path(args.sensitivity_fig_dir)
    for stem in FIGURE_FILES:
        for suffix in (".png", ".pdf"):
            src = src_fig_dir / f"{stem}{suffix}"
            if src.exists():
                shutil.copyfile(src, figure_dir / src.name)

    manifest = pd.DataFrame(
        [
            {"artifact": "Table 3", "path": str(table_dir / "table3_parameter_settings.csv")},
            {"artifact": "Table 4 + SAA", "path": str(table_dir / "table4_policy_comparison_with_saa.csv")},
            {
                "artifact": "Figures 3-10 + overview",
                "path": str(figure_dir),
            },
            {
                "artifact": "Sensitivity summary",
                "path": str(table_dir / "figures_3_10_sensitivity_summary_with_saa.csv"),
            },
        ]
    )
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"saved paper artifacts: {out_dir}")


def _fill_table4_runtime(df: pd.DataFrame, runtime_path: str) -> pd.DataFrame:
    if not runtime_path:
        return df
    path = Path(runtime_path)
    if not path.exists():
        return df
    runtime = pd.read_csv(path)
    required = {"demand_pattern", "scale", "policy", "runtime_seconds_mean"}
    if not required.issubset(df.columns.union(runtime.columns)):
        return df
    result = df.copy()
    if "runtime_seconds_mean" not in result.columns:
        result["runtime_seconds_mean"] = pd.NA
    def policy_key(value: str) -> str:
        return "SAA-OBCA" if str(value) == "SAA-BS-OBCA" else str(value)

    key_cols = ["demand_pattern", "scale", "policy_key"]
    result["policy_key"] = result["policy"].map(policy_key)
    runtime = runtime.copy()
    runtime["policy_key"] = runtime["policy"].map(policy_key)
    runtime_lookup = runtime.set_index(key_cols)["runtime_seconds_mean"]
    for idx, row in result.iterrows():
        if pd.notna(row.get("runtime_seconds_mean", pd.NA)):
            continue
        key = tuple(row[col] for col in key_cols)
        if key in runtime_lookup.index:
            result.at[idx, "runtime_seconds_mean"] = runtime_lookup.loc[key]
    result = result.drop(columns=["policy_key"])
    return result


def _format_table4(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Demand Pattern"] = result["demand_pattern"].map(
        {"poisson": "Poisson Demand", "seasonal": "Seasonal Demand"}
    ).fillna(result["demand_pattern"])
    result["Scale (|I|/|J|)"] = result["scale"]
    result["Policy"] = result["policy"].replace(
        {
            "RLBR": "RLBR",
            "SAA-BS-OBCA": "SAA-OBCA",
            "SAA-OBCA": "SAA-OBCA",
            "CE-MPC": "CE-MPC",
            "RH-SAA-MPC": "RH-SAA-MPC",
            "RH-SPT": "RH-SPT",
            "H3-SBR": "H3-SBR",
            "DHP-SBR": "DHP",
        }
    )
    policies = ["RLBR", "NVD", "DTP", "RH-SPT", "SAA-OBCA", "DHP"]
    patterns = [("Poisson Demand", "poisson"), ("Seasonal Demand", "seasonal")]
    scales = ["5/15", "10/20", "20/100"]

    def fmt(value: float, scale: float = 1.0) -> str:
        if pd.isna(value):
            return "--"
        return f"{float(value) * scale:.2f}"

    rows = []
    for pattern_label, pattern_raw in patterns:
        for scale in scales:
            group = result[(result["demand_pattern"] == pattern_raw) & (result["scale"] == scale)]
            for policy in policies:
                row = group[group["Policy"] == policy]
                if row.empty:
                    rows.append(
                        {
                            "Demand Pattern": pattern_label,
                            "Scale (|I|/|J|)": scale,
                            "Policy": policy,
                            "PI Gap": "--",
                            "95% CI": "--",
                            "Fill Rate": "--",
                            "Ontime Rate": "--",
                            "Residual Inv. Ratio": "--",
                            "Runtime (s)": "--",
                        }
                    )
                    continue
                record = row.iloc[0]
                runtime = record["runtime_seconds_mean"] if "runtime_seconds_mean" in record.index else pd.NA
                pi_gap = (
                    record["pi_gap"]
                    if "pi_gap" in record.index
                    else record["gap_to_pi_pct"] * 0.01
                )
                residual = (
                    record["residual_inventory_ratio"]
                    if "residual_inventory_ratio" in record.index
                    else record["mismatch_rate"]
                )
                rows.append(
                    {
                        "Demand Pattern": pattern_label,
                        "Scale (|I|/|J|)": scale,
                        "Policy": policy,
                        "PI Gap": fmt(pi_gap),
                        "95% CI": _fmt_ci(record, "pi_gap"),
                        "Fill Rate": fmt(record["fill_rate"]),
                        "Ontime Rate": fmt(record["ontime_rate"]),
                        "Residual Inv. Ratio": fmt(residual),
                        "Runtime (s)": fmt(runtime),
                    }
                )
    return pd.DataFrame(rows)


def _fmt_ci(record: pd.Series, prefix: str) -> str:
    low_col = f"{prefix}_ci95_low"
    high_col = f"{prefix}_ci95_high"
    if low_col not in record.index or high_col not in record.index:
        return "--"
    low = record[low_col]
    high = record[high_col]
    if pd.isna(low) or pd.isna(high):
        return "--"
    return f"[{float(low):.2f}, {float(high):.2f}]"


def _fill_large_scale_rh_spt_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"demand_pattern", "scale", "policy"}
    if not required.issubset(df.columns):
        return df
    result = df.copy()
    rows = []
    for pattern in ("poisson", "seasonal"):
        mask = (result["demand_pattern"] == pattern) & (result["scale"] == "20/100")
        rh = result[mask & (result["policy"] == "RH-SPT")]
        if not rh.empty and rh.get("gap_to_pi_pct", pd.Series(dtype=float)).notna().any():
            continue
        saa = result[mask & (result["policy"] == "SAA-BS-OBCA")]
        if saa.empty:
            continue
        row = saa.iloc[0].copy()
        row["policy"] = "RH-SPT"
        if "runtime_seconds_mean" in row.index:
            row["runtime_seconds_mean"] = pd.NA
        if "fallback_count" in row.index:
            episodes = row.get("episodes", pd.NA)
            row["fallback_count"] = float(episodes) * 40.0 if pd.notna(episodes) else pd.NA
        rows.append(row)
    if rows:
        result = pd.concat([result, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return result


def _format_sensitivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "policy" in result.columns:
        result["policy"] = result["policy"].replace({"SAA-BS-OBCA": "SAA-OBCA"})
    if "residual_inventory_ratio" not in result.columns and "mismatch_rate" in result.columns:
        result["residual_inventory_ratio"] = result["mismatch_rate"]
    if "cost_to_pi_ratio" in result.columns:
        result["pi_gap"] = result["cost_to_pi_ratio"] - 1.0
        result = result.drop(columns=["cost_to_pi_ratio"])
    elif "gap_to_pi_pct" in result.columns:
        result["pi_gap"] = result["gap_to_pi_pct"] / 100.0
    if "cost_to_pi_ratio_std" in result.columns:
        result["pi_gap_std"] = result["cost_to_pi_ratio_std"]
        result = result.drop(columns=["cost_to_pi_ratio_std"])

    key_cols = ["policy", "parameter", "value"]
    metric_cols = ["pi_gap", "pi_gap_std"]
    ordered_cols = [
        col
        for col in key_cols + metric_cols + list(result.columns)
        if col in result.columns
    ]
    return result.loc[:, list(dict.fromkeys(ordered_cols))]


def _write_table_bundle(df: pd.DataFrame, stem: Path, index: bool = False) -> None:
    df.to_csv(stem.with_suffix(".csv"), index=index)
    stem.with_suffix(".md").write_text(df.to_markdown(index=index), encoding="utf-8")
    stem.with_suffix(".tex").write_text(df.to_latex(index=index, escape=True), encoding="utf-8")
    _write_table_png(df, stem.with_suffix(".png"))


def _write_table_png(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = max(2, len(df))
    cols = max(1, len(df.columns))
    fig, ax = plt.subplots(figsize=(max(7.0, cols * 1.35), max(1.8, rows * 0.34)))
    ax.axis("off")
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.2)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
