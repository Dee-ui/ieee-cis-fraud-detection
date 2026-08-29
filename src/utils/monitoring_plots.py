"""Charts for the monitoring stage."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from config.config import PSI_SIGNIFICANT  # noqa: E402

FRAUD_COLOUR = "#c0392b"
LEGIT_COLOUR = "#2c7fb8"
NEUTRAL_COLOUR = "#7f8c8d"
ACCENT_COLOUR = "#16a085"

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    print(f"    saved {path.name}")
    return path


def plot_performance_over_time(period_metrics: pd.DataFrame, output_dir: Path) -> Path:
    """
    Model quality week by week, on labelled data it never trained on.

    Plotted as lift over each week's own baseline rather than raw PR-AUC.
    The fraud rate moves from week to week, and PR-AUC's floor is that rate,
    so raw scores from different weeks stand on different floors and cannot
    be compared directly. On this project the raw scores looked flat, roughly
    a 2% decline, while the lift showed a 21% decline over the same weeks.
    """
    figure, axis = plt.subplots(figsize=(11, 5))

    full = period_metrics[period_metrics.get("is_full_week", True)]

    axis.plot(
        period_metrics["period"],
        period_metrics["pr_auc_lift"],
        marker="o",
        color=ACCENT_COLOUR,
        linewidth=1.8,
        label="lift over that week's baseline",
    )

    # Mark partial weeks hollow so a short week at the edge is not misread.
    partial = period_metrics[~period_metrics.get("is_full_week", True)]
    if not partial.empty:
        axis.scatter(
            partial["period"],
            partial["pr_auc_lift"],
            facecolors="white",
            edgecolors=NEUTRAL_COLOUR,
            zorder=5,
            s=90,
            label="partial week, fewer rows",
        )

    if len(full) >= 2:
        mean_value = full["pr_auc_lift"].mean()
        axis.axhline(
            mean_value,
            color=NEUTRAL_COLOUR,
            linestyle="--",
            label=f"mean of full weeks {mean_value:.1f}x",
        )

    for _, row in period_metrics.iterrows():
        axis.annotate(
            f"{row['pr_auc_lift']:.1f}x",
            (row["period"], row["pr_auc_lift"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    axis.set_xlabel("Week of the held-out validation period")
    axis.set_ylabel("PR-AUC lift over the period's own fraud rate")
    axis.set_title("Model advantage over guessing, week by week")
    axis.legend()
    figure.autofmt_xdate(rotation=30)

    return _save(figure, output_dir / "16_performance_over_time.png")


def plot_feature_drift(
    drift: pd.DataFrame, top_features: list[str], output_dir: Path
) -> Path:
    """
    A grid of PSI: the model's most important features against each month.

    Darker means more drift. Reading down a column shows how one month
    compares with training. Reading across a row shows whether one feature
    keeps getting worse.
    """
    pivot = (
        drift[drift["feature"].isin(top_features)]
        .pivot_table(index="feature", columns="period", values="psi")
        .reindex(top_features)
    )

    figure, axis = plt.subplots(figsize=(1.6 * max(len(pivot.columns), 4) + 5, 9))

    image = axis.imshow(
        pivot.to_numpy(),
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(PSI_SIGNIFICANT * 2, float(np.nanmax(pivot.to_numpy())) or 0.5),
    )

    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels(pivot.index, fontsize=9)

    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if value < PSI_SIGNIFICANT else "white",
                )

    axis.set_title(
        f"Feature drift (PSI) against training. "
        f"Above {PSI_SIGNIFICANT} means investigate."
    )
    figure.colorbar(image, ax=axis, label="PSI")
    axis.grid(False)

    return _save(figure, output_dir / "17_feature_drift.png")


def plot_score_drift(score_drift: pd.DataFrame, output_dir: Path) -> Path:
    """
    How the model's risk scores move month by month.

    The scores are the model's opinion. If the shape of that opinion shifts
    while the threshold stays fixed, the number of alerts changes even though
    nothing about the model changed.
    """
    figure, axis = plt.subplots(figsize=(11, 5))

    for column, label, colour in (
        ("score_p50", "median", NEUTRAL_COLOUR),
        ("score_p90", "90th percentile", LEGIT_COLOUR),
        ("score_p99", "99th percentile", FRAUD_COLOUR),
    ):
        axis.plot(
            score_drift["period"],
            score_drift[column],
            marker="o",
            label=label,
            color=colour,
            linewidth=1.6,
        )

    axis.set_xlabel("Period")
    axis.set_ylabel("Predicted fraud probability")
    axis.set_title("Risk score distribution over the unlabelled test period")
    axis.legend()
    figure.autofmt_xdate(rotation=30)

    return _save(figure, output_dir / "18_score_drift.png")


def plot_alert_rate(
    score_drift: pd.DataFrame, expected_rate: float, output_dir: Path
) -> Path:
    """
    The share of transactions crossing the fixed threshold, month by month.

    This is the number an operations manager feels directly, because it is
    how much work arrives in the review queue. If it doubles, the team cannot
    cope, and that happens without anyone changing anything.
    """
    figure, axis = plt.subplots(figsize=(11, 5))

    bars = axis.bar(
        score_drift["period"],
        score_drift["alert_rate"] * 100,
        color=ACCENT_COLOUR,
        alpha=0.85,
    )
    axis.axhline(
        expected_rate * 100,
        color=FRAUD_COLOUR,
        linestyle="--",
        label=f"expected {expected_rate:.1%}",
    )

    for bar, value in zip(bars, score_drift["alert_rate"], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 100,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set_xlabel("Period")
    axis.set_ylabel("Share of transactions alerted (%)")
    axis.set_title("Review queue volume at the fixed threshold")
    axis.legend()
    figure.autofmt_xdate(rotation=30)

    return _save(figure, output_dir / "19_alert_rate.png")
