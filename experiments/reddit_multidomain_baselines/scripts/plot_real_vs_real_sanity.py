#!/usr/bin/env python3
"""Create publication-ready real-vs-real sanity-check figures.

The p-value heatmap uses domain x metric pass rates directly.  The radar figure
uses the unweighted mean of metric-level distance values within each of the five
declared metric families.  Twelve domains are split across three radar rows so
that individual lines remain readable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "experiments"
    / "reddit_multidomain_baselines"
    / "results"
    / "real_vs_real_sanity"
    / "repeated_real_vs_real_metric_summary.csv"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "figures"

DOMAIN_ORDER = (
    "camera",
    "celebrity",
    "cellphone",
    "credit_cards",
    "game",
    "headphones",
    "health_issue",
    "laptop",
    "movies",
    "news",
    "sports",
    "tv_series",
)
DOMAIN_LABELS = {
    "camera": "Camera",
    "celebrity": "Celebrity",
    "cellphone": "Cellphone",
    "credit_cards": "Credit Cards",
    "game": "Game",
    "headphones": "Headphones",
    "health_issue": "Health Issue",
    "laptop": "Laptop",
    "movies": "Movies",
    "news": "News",
    "sports": "Sports",
    "tv_series": "TV Series",
}
FAMILY_ORDER = ("Uniformity", "Expression", "Tone", "Interaction", "Form")
METRIC_ORDER = (
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    "mean_story_probability",
    "emotion_entropy",
    "polite_rate",
    "neutral_rate",
    "impolite_rate",
    "avg_depth",
    "hard_disagree_rate",
    "structural_virality",
    "length_cv",
)
METRIC_LABELS = {
    "self_bleu_4": "Self BLEU 4",
    "self_bertscore_mean_f1": "Self BERTScore Mean F1",
    "semantic_mean_cosine": "Semantic Mean Cosine",
    "mean_story_probability": "Mean Story Probability",
    "emotion_entropy": "Emotion Entropy",
    "polite_rate": "Polite Rate",
    "neutral_rate": "Neutral Rate",
    "impolite_rate": "Impolite Rate",
    "avg_depth": "Average Depth",
    "hard_disagree_rate": "Hard Disagree Rate",
    "structural_virality": "Structural Virality",
    "length_cv": "Length CV",
}
RADAR_GROUPS = (
    DOMAIN_ORDER[0:4],
    DOMAIN_ORDER[4:8],
    DOMAIN_ORDER[8:12],
)
DOMAIN_COLORS = dict(
    zip(
        DOMAIN_ORDER,
        (
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#E45756",
            "#9C6BCF",
            "#72B7B2",
            "#B279A2",
            "#FF9DA6",
            "#79706E",
            "#D6A84B",
            "#5B8E7D",
            "#A15D98",
        ),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path.expanduser().resolve())
    required = {
        "domain",
        "family",
        "metric",
        "mwu_p_ge_alpha_pct",
        "ks_p_ge_alpha_pct",
        "mean_wasserstein_distance",
        "mean_abs_cliffs_delta",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    expected = {(domain, metric) for domain in DOMAIN_ORDER for metric in METRIC_ORDER}
    observed = set(zip(frame["domain"].astype(str), frame["metric"].astype(str)))
    if observed != expected:
        missing_pairs = sorted(expected - observed)
        extra_pairs = sorted(observed - expected)
        raise ValueError(
            f"Expected exactly 12 domains x 12 metrics; "
            f"missing={missing_pairs[:5]} extra={extra_pairs[:5]}"
        )
    return frame.copy()


def _pivot(frame: pd.DataFrame, value: str) -> np.ndarray:
    wide = frame.pivot(index="metric", columns="domain", values=value)
    return wide.reindex(index=METRIC_ORDER, columns=DOMAIN_ORDER).to_numpy(dtype=float)


def plot_pvalue_heatmap(frame: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "miro_pass_rate", ("#E7B7B0", "#F3F0E8", "#BDD5DF")
    )
    norm = TwoSlopeNorm(vmin=89.0, vcenter=95.0, vmax=100.0)
    panels = (
        ("Mann–Whitney U p-value pass rate", "mwu_p_ge_alpha_pct"),
        ("Kolmogorov–Smirnov p-value pass rate", "ks_p_ge_alpha_pct"),
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(18.8, 7.0), sharey=True)
    images = []
    for panel_index, (ax, (title, column)) in enumerate(zip(axes, panels)):
        values = _pivot(frame, column)
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        images.append(image)
        ax.set_title(title, pad=13)
        ax.set_xticks(np.arange(len(DOMAIN_ORDER)))
        ax.set_xticklabels(
            [DOMAIN_LABELS[domain] for domain in DOMAIN_ORDER],
            rotation=38,
            ha="right",
        )
        ax.set_yticks(np.arange(len(METRIC_ORDER)))
        ax.set_yticklabels([METRIC_LABELS[metric] for metric in METRIC_ORDER])
        ax.tick_params(length=0)

        ax.set_xticks(np.arange(-0.5, len(DOMAIN_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(METRIC_ORDER), 1), minor=True)
        ax.grid(which="minor", color="#FFFFFF", linewidth=1.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        for boundary in (2.5, 4.5, 7.5, 10.5):
            ax.axhline(boundary, color="#5D6470", linewidth=1.35)

        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    color="#18202D",
                    fontweight="bold" if value >= 95.0 else "normal",
                )

        for spine in ax.spines.values():
            spine.set_color("#D7DADE")
            spine.set_linewidth(0.9)
        if panel_index == 0:
            ax.set_ylabel("Thread-level metric")

    fig.subplots_adjust(left=0.145, right=0.895, top=0.955, bottom=0.175, wspace=0.075)
    colorbar_ax = fig.add_axes((0.914, 0.175, 0.012, 0.78))
    colorbar = fig.colorbar(
        images[0], cax=colorbar_ax, ticks=(90, 92, 95, 97, 100)
    )
    colorbar.set_label("% of repeated samples with p ≥ 0.05", rotation=90, labelpad=14)
    colorbar.outline.set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"repeated_real_vs_real_pvalue_heatmap_12_domains.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def build_family_table(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["domain", "family"], as_index=False, sort=False)
        .agg(
            metric_count=("metric", "nunique"),
            mean_wasserstein_distance=("mean_wasserstein_distance", "mean"),
            mean_abs_cliffs_delta=("mean_abs_cliffs_delta", "mean"),
        )
    )
    grouped["domain"] = pd.Categorical(
        grouped["domain"], categories=DOMAIN_ORDER, ordered=True
    )
    grouped["family"] = pd.Categorical(
        grouped["family"], categories=FAMILY_ORDER, ordered=True
    )
    return grouped.sort_values(["domain", "family"]).reset_index(drop=True)


def _closed_angles() -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, len(FAMILY_ORDER), endpoint=False)
    return angles, np.r_[angles, angles[0]]


def _style_radar(
    ax: plt.Axes,
    *,
    ticks: tuple[float, ...],
    tick_labels: tuple[str, ...],
    limits: tuple[float, float],
) -> None:
    angles, _ = _closed_angles()
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(FAMILY_ORDER, fontsize=9)
    ax.tick_params(axis="x", pad=8)
    ax.set_ylim(*limits)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=7, color="#505865")
    ax.set_rlabel_position(18)
    ax.grid(color="#C7CBD0", linewidth=0.65)
    ax.spines["polar"].set_color("#2C313A")
    ax.spines["polar"].set_linewidth(0.9)


def _plot_radar_lines(
    ax: plt.Axes,
    family_table: pd.DataFrame,
    domains: tuple[str, ...],
    value_column: str,
    *,
    log10: bool,
) -> list:
    _, angles_closed = _closed_angles()
    handles = []
    for domain in domains:
        selected = family_table[family_table["domain"] == domain].set_index("family")
        values = selected.reindex(FAMILY_ORDER)[value_column].to_numpy(dtype=float)
        if log10:
            values = np.log10(values)
        values_closed = np.r_[values, values[0]]
        (line,) = ax.plot(
            angles_closed,
            values_closed,
            color=DOMAIN_COLORS[domain],
            linewidth=1.65,
            marker="o",
            markersize=3.3,
            label=DOMAIN_LABELS[domain],
        )
        ax.fill(angles_closed, values_closed, color=DOMAIN_COLORS[domain], alpha=0.035)
        handles.append(line)
    return handles


def plot_family_radars(
    family_table: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
        }
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(20.0, 9.8),
        subplot_kw={"projection": "polar"},
    )
    w1_raw_ticks = (0.01, 0.02, 0.05, 0.10, 0.15)
    w1_ticks = tuple(float(np.log10(value)) for value in w1_raw_ticks)
    cliff_ticks = (0.04, 0.05, 0.06)

    for column, domains in enumerate(RADAR_GROUPS):
        top, bottom = axes[:, column]
        _style_radar(
            top,
            ticks=w1_ticks,
            tick_labels=tuple(f"{value:g}" for value in w1_raw_ticks),
            limits=(float(np.log10(0.009)), float(np.log10(0.16))),
        )
        w1_handles = _plot_radar_lines(
            top,
            family_table,
            domains,
            "mean_wasserstein_distance",
            log10=True,
        )
        top.set_title(
            "Raw family-mean Wasserstein distance",
            fontsize=10,
            fontweight="semibold",
            pad=17,
        )

        _style_radar(
            bottom,
            ticks=cliff_ticks,
            tick_labels=tuple(f"{value:.2f}" for value in cliff_ticks),
            limits=(0.034, 0.061),
        )
        cliff_handles = _plot_radar_lines(
            bottom,
            family_table,
            domains,
            "mean_abs_cliffs_delta",
            log10=False,
        )
        bottom.set_title(
            "Raw family-mean |Cliff’s delta|",
            fontsize=10,
            fontweight="semibold",
            pad=17,
        )
        for ax, handles in ((top, w1_handles), (bottom, cliff_handles)):
            ax.legend(
            handles=handles,
            labels=[DOMAIN_LABELS[domain] for domain in domains],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.095),
            ncol=2,
            frameon=False,
            fontsize=7.7,
            handlelength=1.5,
            columnspacing=0.8,
            handletextpad=0.35,
            )

    fig.subplots_adjust(
        left=0.035, right=0.985, top=0.965, bottom=0.04, hspace=0.40, wspace=0.30
    )
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"repeated_real_vs_real_family_radars_12_domains.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_summary(args.input)
    family_table = build_family_table(frame)
    family_table.to_csv(output_dir / "real_vs_real_family_radar_values.csv", index=False)
    plot_pvalue_heatmap(frame, output_dir, args.dpi)
    plot_family_radars(family_table, output_dir, args.dpi)
    print(f"[complete] figures={output_dir}")


if __name__ == "__main__":
    main()
