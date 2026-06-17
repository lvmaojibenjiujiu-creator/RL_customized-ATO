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

POLICY_ORDER = ["RLBR", "NVD", "DTP", "RH-SPT", "SAA-BS-OBCA", "SAA-OBCA", "DHP-SBR", "DHP"]
POLICY_STYLES = {
    "RLBR": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "label": "RLBR"},
    "NVD": {"color": "#2ca02c", "marker": "^", "linestyle": "-", "label": "NVD"},
    "DTP": {"color": "#ff7f0e", "marker": "s", "linestyle": "-", "label": "DTP"},
    "RH-SPT": {"color": "#d62728", "marker": "v", "linestyle": "-", "label": "RH-SPT"},
    "SAA-BS-OBCA": {"color": "#9467bd", "marker": "D", "linestyle": "-", "label": "SAA-OBCA"},
    "SAA-OBCA": {"color": "#9467bd", "marker": "D", "linestyle": "-", "label": "SAA-OBCA"},
    "DHP-SBR": {"color": "#7f7f7f", "marker": "X", "linestyle": "-", "label": "DHP"},
    "DHP": {"color": "#7f7f7f", "marker": "X", "linestyle": "-", "label": "DHP"},
}
FIGURE_PAIRS = [
    (("bom_cv", "component_commonality"), "figures_3_4_structural_complexity"),
    (("delivery_window", "max_replenishment_lead_time"), "figures_5_6_temporal_dynamics"),
    (("demand_correlation", "seasonal_beta"), "figures_8_9_demand_dynamics"),
]
LEFT_PAIR_PARAMS = {pair[0][0] for pair in FIGURE_PAIRS}
RIGHT_PAIR_PARAMS = {pair[0][1] for pair in FIGURE_PAIRS}
PAIRED_PARAMS = LEFT_PAIR_PARAMS | RIGHT_PAIR_PARAMS


def sensitivity_plot_table(
    summary: pd.DataFrame, episodes: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, str, str]:
    use_pi_gap = (
        "cost_to_pi_ratio" in summary.columns and summary["cost_to_pi_ratio"].notna().any()
    ) or ("gap_to_pi_pct" in summary.columns and summary["gap_to_pi_pct"].notna().any())
    y_col = "pi_gap" if use_pi_gap else "cost_mean"
    y_label = "PI Gap" if use_pi_gap else "Total cost"

    if episodes is not None and not episodes.empty:
        ep = episodes.copy()
        if use_pi_gap and "cost_to_pi_ratio" in ep.columns and ep["cost_to_pi_ratio"].notna().any():
            ep_y_col = "pi_gap"
            ep[ep_y_col] = ep["cost_to_pi_ratio"] - 1.0
        elif use_pi_gap and {"cost", "pi_cost"}.issubset(ep.columns):
            ep_y_col = "pi_gap"
            ep[ep_y_col] = ep["cost"] / ep["pi_cost"].replace(0.0, np.nan) - 1.0
        else:
            ep_y_col = "cost"
        grouped = (
            ep.groupby(["parameter", "value", "policy"], as_index=False)[ep_y_col]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "y_mean", "std": "y_std"})
        )
        grouped["y_std"] = grouped["y_std"].fillna(0.0)
        if ep_y_col == y_col or (not use_pi_gap and ep_y_col == "cost"):
            return grouped, y_label, ep_y_col

    table = summary[["parameter", "value", "policy"]].copy()
    if use_pi_gap and "cost_to_pi_ratio" in summary.columns:
        table["y_mean"] = summary["cost_to_pi_ratio"] - 1.0
    elif use_pi_gap and "gap_to_pi_pct" in summary.columns:
        table["y_mean"] = summary["gap_to_pi_pct"] / 100.0
    else:
        table["y_mean"] = summary[y_col]
    if use_pi_gap and "cost_to_pi_ratio_std" in summary.columns:
        table["y_std"] = summary["cost_to_pi_ratio_std"].fillna(0.0)
    elif not use_pi_gap and "cost_std" in summary.columns:
        table["y_std"] = summary["cost_std"].fillna(0.0)
    elif use_pi_gap and {"cost_std", "pi_cost_mean"}.issubset(summary.columns):
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
    legend_paths = _save_vertical_legend(plt, out, formats)
    saved.extend(legend_paths)
    for param in params:
        spec = FIGURE_SPECS[param]
        if param in PAIRED_PARAMS:
            fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=False)
            fig.subplots_adjust(
                left=0.18 if param in LEFT_PAIR_PARAMS else 0.10,
                right=0.98,
                bottom=0.18,
                top=0.88,
            )
            _draw_parameter_axis(
                ax,
                plot_table[plot_table["parameter"] == param],
                spec,
                y_label,
                compact=param in RIGHT_PAIR_PARAMS,
                legend="none",
            )
        else:
            fig, ax = plt.subplots(figsize=(5.15, 3.0), constrained_layout=False)
            fig.subplots_adjust(left=0.15, right=0.73, bottom=0.18, top=0.88)
            _draw_parameter_axis(
                ax,
                plot_table[plot_table["parameter"] == param],
                spec,
                y_label,
                legend="right",
            )
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
            fig = plt.figure(figsize=(8.8, 3.1), constrained_layout=False)
            gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.18, 1.0], wspace=0.18)
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
            legend_ax = fig.add_subplot(gs[0, 1])
            fig.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.88)
            for idx, (ax, param) in enumerate(zip(axes, pair)):
                spec = FIGURE_SPECS[param]
                _draw_parameter_axis(
                    ax,
                    plot_table[plot_table["parameter"] == param],
                    spec,
                    y_label,
                    compact=idx == 1,
                    legend="none",
                )
            handles, labels = axes[0].get_legend_handles_labels()
            _draw_vertical_legend_axis(legend_ax, handles, labels)
            for fmt in formats:
                path = out / f"{filename}.{fmt}"
                fig.savefig(path, dpi=320 if fmt.lower() == "png" else None, bbox_inches="tight")
                saved.append(path)
            plt.close(fig)

    if overview and params:
        fig, axes = plt.subplots(4, 2, figsize=(9.2, 11.2), constrained_layout=False)
        fig.subplots_adjust(left=0.08, right=0.84, bottom=0.06, top=0.96, hspace=0.58, wspace=0.18)
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
        fig.legend(
            handles,
            labels,
            loc="center right",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(0.985, 0.5),
            handlelength=1.8,
            borderaxespad=0.0,
        )
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


def _save_vertical_legend(plt, out: Path, formats: Iterable[str]) -> list[Path]:
    handles, labels = _legend_handles()
    fig, ax = plt.subplots(figsize=(1.15, 1.85), constrained_layout=True)
    _draw_vertical_legend_axis(ax, handles, labels)
    saved: list[Path] = []
    for fmt in formats:
        path = out / f"sensitivity_legend_vertical.{fmt}"
        fig.savefig(path, dpi=320 if fmt.lower() == "png" else None, transparent=True)
        saved.append(path)
    plt.close(fig)
    return saved


def _legend_handles():
    from matplotlib.lines import Line2D

    seen: set[str] = set()
    handles = []
    labels = []
    for policy in POLICY_ORDER:
        style = POLICY_STYLES[policy]
        label = str(style["label"])
        if label in seen:
            continue
        seen.add(label)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.2,
                markersize=4.3,
            )
        )
        labels.append(label)
    return handles, labels


def _draw_vertical_legend_axis(ax, handles, labels) -> None:
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        loc="center",
        ncol=1,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.65,
    )


def _draw_parameter_axis(
    ax,
    sub: pd.DataFrame,
    spec: FigureSpec,
    y_label: str,
    compact: bool = False,
    legend: str = "right",
) -> None:
    for policy in POLICY_ORDER:
        grp = sub[sub["policy"] == policy].sort_values("value")
        if grp.empty:
            continue
        x = grp["value"].astype(float).to_numpy()
        y = grp["y_mean"].astype(float).to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue
        x, y = x[valid], y[valid]
        style = POLICY_STYLES.get(policy, {"color": None, "marker": "o", "linestyle": "-", "label": policy})
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.2,
            markersize=4.3,
            label=style["label"],
        )
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(y_label if not compact else "")
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.7)
    ax.grid(True, axis="x", alpha=0.08, linewidth=0.5)
    ax.margins(x=0.04, y=0.12)
    ax.tick_params(direction="out", length=3, width=0.8)
    if legend == "right":
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            handlelength=1.8,
            handletextpad=0.6,
            borderaxespad=0.0,
            labelspacing=0.65,
        )
