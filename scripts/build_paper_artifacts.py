#!/usr/bin/env python3
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

    raw_table4 = pd.read_csv(args.table4)
    table4 = _format_table4(raw_table4)
    _write_table_bundle(table4, table_dir / "table4_policy_comparison_with_saa", index=False)

    sensitivity_summary = pd.read_csv(args.sensitivity_summary)
    sensitivity_summary.to_csv(table_dir / "figures_3_10_sensitivity_summary_with_saa.csv", index=False)
    audit_path = Path(args.audit)
    if audit_path.exists():
        shutil.copyfile(audit_path, table_dir / audit_path.name)

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


def _format_table4(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Demand Pattern"] = result["demand_pattern"].map(
        {"poisson": "Poisson Demand", "seasonal": "Seasonal Demand"}
    ).fillna(result["demand_pattern"])
    result["Scale (|I|/|J|)"] = result["scale"]
    result["Policy"] = result["policy"].replace({"RLBR": "RLBR", "SAA-BS-OBCA": "SAA-BS-OBCA"})
    result["Cost Mean"] = result["cost_mean"].round(2)
    result["Gap to PI (%)"] = result["gap_to_pi_pct"].round(2)
    result["Fill Rate"] = result["fill_rate"].round(3)
    result["Ontime Rate"] = result["ontime_rate"].round(3)
    result["Mismatch Rate"] = result["mismatch_rate"].round(3)
    policy_order = {"RLBR": 0, "SAA-BS-OBCA": 1, "NVD": 2, "DTP": 3}
    pattern_order = {"Poisson Demand": 0, "Seasonal Demand": 1}
    scale_order = {"5/15": 0, "10/20": 1, "20/100": 2}
    result["_pattern_order"] = result["Demand Pattern"].map(pattern_order).fillna(99)
    result["_scale_order"] = result["Scale (|I|/|J|)"].map(scale_order).fillna(99)
    result["_policy_order"] = result["Policy"].map(policy_order).fillna(99)
    result = result.sort_values(["_pattern_order", "_scale_order", "_policy_order"])
    return result[
        [
            "Demand Pattern",
            "Scale (|I|/|J|)",
            "Policy",
            "Cost Mean",
            "Gap to PI (%)",
            "Fill Rate",
            "Ontime Rate",
            "Mismatch Rate",
        ]
    ]


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
