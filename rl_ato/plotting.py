from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FigureSpec:
    number: int
    title: str
    xlabel: str
    filename: str


FIGURE_SPECS = {
    "bom_cv": FigureSpec(
        3,
        "Effect of BOM Variability",
        r"BOM variability ($\sigma_{\mathrm{BOM}}/\mu_{\mathrm{BOM}}$)",
        "figure_3_bom_variability",
    ),
    "component_commonality": FigureSpec(
        4,
        "Effect of Component Commonality Ratio",
        "Component commonality ratio",
        "figure_4_component_commonality",
    ),
    "delivery_window": FigureSpec(
        5,
        r"Effect of Delivery Window ($L_W$)",
        r"Delivery window ($L_W$)",
        "figure_5_delivery_window",
    ),
    "max_replenishment_lead_time": FigureSpec(
        6,
        "Effect of Maximum Replenishment Lead Time",
        r"Maximum replenishment lead time ($L_{\max}^r$)",
        "figure_6_lead_time_uncertainty",
    ),
    "design_lead_time": FigureSpec(
        7,
        r"Effect of Design Lead Time ($L_d$)",
        r"Design lead time ($L_d$)",
        "figure_7_design_lead_time",
    ),
    "demand_correlation": FigureSpec(
        8,
        r"Effect of Demand Correlation ($\rho$)",
        r"Demand correlation ($\rho$)",
        "figure_8_demand_correlation",
    ),
    "seasonal_beta": FigureSpec(
        9,
        r"Effect of Amplitude ($\beta$)",
        r"Seasonality amplitude ($\beta$)",
        "figure_9_seasonal_amplitude",
    ),
    "backorder_to_holding": FigureSpec(
        10,
        r"Effect of Backorder Cost Ratio ($b/h$)",
        r"Backorder cost ratio ($b/h$)",
        "figure_10_backorder_cost_ratio",
    ),
}

POLICY_ORDER = ["RLBR", "DTP", "NVD", "SAA-BS-OBCA"]
POLICY_STYLES = {
    "RLBR": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "label": "RLBR"},
    "DTP": {"color": "#ff7f0e", "marker": "s", "linestyle": "-", "label": "DTP"},
    "NVD": {"color": "#2ca02c", "marker": "^", "linestyle": "-", "label": "NVD"},
    "SAA-BS-OBCA": {"color": "#9467bd", "marker": "D", "linestyle": "-", "label": "SAA-BS-OBCA"},
}
FIGURE_PAIRS = [
    (("bom_cv", "component_commonality"), "figures_3_4_structural_complexity"),
    (("delivery_window", "max_replenishment_lead_time"), "figures_5_6_temporal_dynamics"),
    (("demand_correlation", "seasonal_beta"), "figures_8_9_demand_dynamics"),
]


def sensitivity_plot_table(
    summary: pd.DataFrame, episodes: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, str, str]:
    use_ratio = "cost_to_pi_ratio" in summary.columns and summary["cost_to_pi_ratio"].notna().any()
    y_col = "cost_to_pi_ratio" if use_ratio else "cost_mean"
    y_label = "Cost-to-PI ratio (total cost / oracle total cost)" if use_ratio else "Total cost"

    if episodes is not None and not episodes.empty:
        ep = episodes.copy()
        if use_ratio and "cost_to_pi_ratio" in ep.columns and ep["cost_to_pi_ratio"].notna().any():
            ep_y_col = "cost_to_pi_ratio"
        else:
            ep_y_col = "cost"
        grouped = (
            ep.groupby(["parameter", "value", "policy"], as_index=False)[ep_y_col]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "y_mean", "std": "y_std"})
        )
        grouped["y_std"] = grouped["y_std"].fillna(0.0)
        if ep_y_col == y_col or (not use_ratio and ep_y_col == "cost"):
            return grouped, y_label, ep_y_col

    table = summary[["parameter", "value", "policy", y_col]].copy()
    table = table.rename(columns={y_col: "y_mean"})
    if use_ratio and "cost_to_pi_ratio_std" in summary.columns:
        table["y_std"] = summary["cost_to_pi_ratio_std"].fillna(0.0)
    elif not use_ratio and "cost_std" in summary.columns:
        table["y_std"] = summary["cost_std"].fillna(0.0)
    elif use_ratio and {"cost_std", "pi_cost_mean"}.issubset(summary.columns):
        table["y_std"] = (summary["cost_std"] / summary["pi_cost_mean"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        table["y_std"] = 0.0
    return table, y_label, y_col


def plot_sensitivity_figures(
    summary: pd.DataFrame,
    episodes: pd.DataFrame | None,
    out_dir: str | Path,
    formats: Iterable[str] = ("png", "pdf"),
    overview: bool = True,
    paired: bool = True,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_matplotlib_style(plt)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_table, y_label, _y_col = sensitivity_plot_table(summary, episodes)
    saved: list[Path] = []

    params = [p for p in FIGURE_SPECS if p in set(plot_table["parameter"])]
    for param in params:
        spec = FIGURE_SPECS[param]
        fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
        _draw_parameter_axis(ax, plot_table[plot_table["parameter"] == param], spec, y_label)
        for fmt in formats:
            path = out / f"{spec.filename}.{fmt}"
            fig.savefig(path, dpi=320 if fmt.lower() == "png" else None)
            saved.append(path)
        plt.close(fig)

    if paired:
        available = set(params)
        for pair, filename in FIGURE_PAIRS:
            if not set(pair).issubset(available):
                continue
            fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.15), constrained_layout=True)
            for ax, param in zip(axes, pair):
                spec = FIGURE_SPECS[param]
                _draw_parameter_axis(
                    ax,
                    plot_table[plot_table["parameter"] == param],
                    spec,
                    y_label,
                    compact=True,
                )
            axes[0].set_ylabel(y_label)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
            for fmt in formats:
                path = out / f"{filename}.{fmt}"
                fig.savefig(path, dpi=320 if fmt.lower() == "png" else None, bbox_inches="tight")
                saved.append(path)
            plt.close(fig)

    if overview and params:
        fig, axes = plt.subplots(4, 2, figsize=(9.2, 11.2), constrained_layout=True)
        for ax, param in zip(axes.ravel(), params):
            spec = FIGURE_SPECS[param]
            _draw_parameter_axis(
                ax,
                plot_table[plot_table["parameter"] == param],
                spec,
                y_label,
                compact=True,
            )
        for ax in axes.ravel()[len(params) :]:
            ax.axis("off")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
        for fmt in formats:
            path = out / f"sensitivity_overview.{fmt}"
            fig.savefig(path, dpi=320 if fmt.lower() == "png" else None, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)
    return saved


def _set_matplotlib_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


def _draw_parameter_axis(ax, sub: pd.DataFrame, spec: FigureSpec, y_label: str, compact: bool = False) -> None:
    for policy in POLICY_ORDER:
        grp = sub[sub["policy"] == policy].sort_values("value")
        if grp.empty:
            continue
        x = grp["value"].astype(float).to_numpy()
        y = grp["y_mean"].astype(float).to_numpy()
        band = grp["y_std"].fillna(0.0).astype(float).to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue
        x, y, band = x[valid], y[valid], np.nan_to_num(band[valid], nan=0.0)
        style = POLICY_STYLES.get(policy, {"color": None, "marker": "o", "linestyle": "-", "label": policy})
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=4.3,
            label=style["label"],
        )
        if np.any(band > 1e-12):
            lower = np.maximum(0.0, y - band)
            upper = y + band
            ax.fill_between(x, lower, upper, color=style["color"], alpha=0.14, linewidth=0)

    ax.set_title(f"Fig {spec.number}. {spec.title}", pad=7)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(y_label if not compact else "")
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.7)
    ax.grid(True, axis="x", alpha=0.08, linewidth=0.5)
    ax.margins(x=0.04, y=0.12)
    ax.tick_params(direction="out", length=3, width=0.8)
    if not compact:
        ax.legend(frameon=False, loc="best")
