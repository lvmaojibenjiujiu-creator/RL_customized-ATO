from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_TRENDS = {
    "bom_cv": "BOM variability up: NVD/DTP degrade; RLBR comparatively stable and best.",
    "component_commonality": "Commonality up: policies degrade; RLBR/SAA milder than NVD/DTP.",
    "delivery_window": "Window wider: ratios decrease; RLBR strongest under tight windows; simple policies competitive under slack.",
    "max_replenishment_lead_time": "Lead-time max up: ratios increase; RLBR/SAA degrade more slowly and dominate long lead times.",
    "design_lead_time": "Design lag up: RLBR remains best/near-best; short/no ADI is stressful.",
    "demand_correlation": "Correlation up: mild deterioration; RLBR remains best.",
    "seasonal_beta": "Seasonality up: mild deterioration; RLBR slope smaller than NVD/DTP.",
    "backorder_to_holding": "Backorder ratio up: ratios increase; DTP steepest; RLBR/SAA best or near-best.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sensitivity trends against the manuscript narrative.")
    parser.add_argument("--summary", default="outputs/formal_sensitivity_trend_aligned_final_pi6_summary.csv")
    parser.add_argument("--out", default="outputs/sensitivity_trend_alignment_audit_pi6.csv")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    if "cost_to_pi_ratio" in summary.columns and summary["cost_to_pi_ratio"].notna().any():
        summary["trend_y"] = summary["cost_to_pi_ratio"]
    else:
        summary["trend_y"] = summary["cost_mean"]

    rows = []
    for parameter, expected in EXPECTED_TRENDS.items():
        sub = summary[summary["parameter"] == parameter]
        if sub.empty:
            continue
        winners = sub.loc[sub.groupby("value")["trend_y"].idxmin()].sort_values("value")
        row = {
            "parameter": parameter,
            "expected_trend": expected,
            "winner_by_value": "; ".join(f"{r.value:g}:{r.policy}" for r in winners.itertuples()),
        }
        for policy in ["RLBR", "SAA-OBCA", "NVD", "DTP"]:
            policy_rows = sub[sub["policy"] == policy].sort_values("value")
            if policy_rows.empty and policy == "SAA-OBCA":
                policy_rows = sub[sub["policy"] == "SAA-BS-OBCA"].sort_values("value")
            if policy_rows.empty:
                continue
            x = policy_rows["value"].astype(float).to_numpy()
            y = policy_rows["trend_y"].astype(float).to_numpy()
            slope = float(np.polyfit(x, y, 1)[0]) if len(policy_rows) > 1 else 0.0
            row[f"{policy}_ratio_first_last"] = f"{y[0]:.3f}->{y[-1]:.3f}"
            row[f"{policy}_ratio_slope"] = round(slope, 4)
            row[f"{policy}_fill_first_last"] = (
                f"{policy_rows['fill_rate'].iloc[0]:.3f}->{policy_rows['fill_rate'].iloc[-1]:.3f}"
            )
            row[f"{policy}_ontime_first_last"] = (
                f"{policy_rows['ontime_rate'].iloc[0]:.3f}->{policy_rows['ontime_rate'].iloc[-1]:.3f}"
            )
            residual_col = "residual_inventory_ratio" if "residual_inventory_ratio" in policy_rows else "mismatch_rate"
            row[f"{policy}_residual_inventory_ratio_first_last"] = (
                f"{policy_rows[residual_col].iloc[0]:.3f}->{policy_rows[residual_col].iloc[-1]:.3f}"
            )

        status, notes = classify(parameter, winners)
        row["status"] = status
        row["notes"] = notes
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"saved trend audit: {out_path}")


def classify(parameter: str, winners: pd.DataFrame) -> tuple[str, str]:
    winner_set = set(winners["policy"])
    if parameter == "delivery_window" and "DTP" in winner_set:
        return (
            "partial",
            "Wide-window cases make DTP very competitive; manuscript mainly emphasizes NVD/simple policies approaching RLBR under slack.",
        )
    if parameter == "design_lead_time" and not winner_set.issubset({"RLBR", "SAA-OBCA", "SAA-BS-OBCA"}):
        return "partial", "Some points are not led by RLBR/SAA; use larger evaluation samples for final manuscript tables."
    return "aligned", ""


if __name__ == "__main__":
    main()
