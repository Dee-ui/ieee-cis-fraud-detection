"""
Charts for the training stage.

Same setup as the EDA charts: the Agg backend is selected before pyplot is
imported, so nothing tries to open a window when this runs from a terminal
or inside a container.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

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


def plot_model_comparison(comparison: pd.DataFrame, output_dir: Path) -> Path:
    """PR-AUC for every candidate, with the random baseline drawn in."""
    ordered = comparison.sort_values("pr_auc")

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.barh(ordered["model"], ordered["pr_auc"], color=ACCENT_COLOUR)

    baseline = float(ordered["pr_auc_baseline"].iloc[0])
    axis.axvline(
        baseline,
        color=NEUTRAL_COLOUR,
        linestyle="--",
        label=f"random baseline {baseline:.4f}",
    )

    for bar, value in zip(bars, ordered["pr_auc"], strict=True):
        axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.4f}",
            va="center",
            fontsize=9,
        )

    axis.set_xlabel("PR-AUC on the validation period")
    axis.set_title("Model comparison")
    axis.set_xlim(0, max(ordered["pr_auc"]) * 1.25)
    axis.legend(loc="lower right")

    return _save(figure, output_dir / "11_model_comparison.png")


def plot_precision_recall_curves(
    y_true, score_sets: dict[str, np.ndarray], output_dir: Path
) -> Path:
    """
    The trade-off curve for every model.

    Reading it: moving right catches more fraud, moving down means more of
    what you flag is a false alarm. A better model sits higher for the same
    recall. The flat dashed line is what random guessing achieves.
    """
    figure, axis = plt.subplots(figsize=(9, 6))

    for name, scores in score_sets.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        axis.plot(recall, precision, linewidth=1.6, label=name)

    prevalence = float(np.mean(y_true))
    axis.axhline(
        prevalence,
        color=NEUTRAL_COLOUR,
        linestyle="--",
        label=f"random ({prevalence:.3f})",
    )

    axis.set_xlabel("Recall: share of all fraud caught")
    axis.set_ylabel("Precision: share of flags that were really fraud")
    axis.set_title("Precision against recall, validation period")
    axis.legend()

    return _save(figure, output_dir / "12_precision_recall_curves.png")


def plot_cost_curve(
    curve: pd.DataFrame,
    unconstrained: dict,
    constrained: dict,
    capacity_rate: float,
    output_dir: Path,
) -> Path:
    """
    Total cost against how much you review.

    The shape tells the story. Review nothing and you pay for every fraud.
    Review everything and you pay for a vast number of pointless reviews.
    The minimum in between is the operating point worth arguing for.
    """
    trimmed = curve[curve["review_rate"] <= 0.20]

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        trimmed["review_rate"] * 100,
        trimmed["total_cost"],
        color=FRAUD_COLOUR,
        linewidth=1.8,
    )

    axis.axvline(
        capacity_rate * 100,
        color=NEUTRAL_COLOUR,
        linestyle=":",
        label=f"review capacity {capacity_rate:.1%}",
    )
    axis.scatter(
        [unconstrained["review_rate"] * 100],
        [unconstrained["total_cost"]],
        color=ACCENT_COLOUR,
        zorder=5,
        s=70,
        label=f"cheapest overall at {unconstrained['review_rate']:.2%}",
    )
    axis.scatter(
        [constrained["review_rate"] * 100],
        [constrained["total_cost"]],
        color=LEGIT_COLOUR,
        zorder=5,
        s=70,
        label=f"cheapest within capacity at {constrained['review_rate']:.2%}",
    )

    axis.set_xlabel("Share of transactions sent for manual review (%)")
    axis.set_ylabel("Total cost over the validation period (USD)")
    axis.set_title("Cost against review rate")
    axis.legend()

    return _save(figure, output_dir / "13_cost_curve.png")


def plot_score_distribution(y_true, scores, output_dir: Path) -> Path:
    """
    Where the two classes sit on the risk scale.

    Good separation looks like two humps that barely overlap. Heavy overlap
    means the model is unsure about most transactions, which caps how well
    any threshold can perform.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)

    figure, axis = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 60)

    axis.hist(
        s[y == 0],
        bins=bins,
        alpha=0.6,
        density=True,
        label="Legitimate",
        color=LEGIT_COLOUR,
    )
    axis.hist(
        s[y == 1], bins=bins, alpha=0.6, density=True, label="Fraud", color=FRAUD_COLOUR
    )

    axis.set_yscale("log")
    axis.set_xlabel("Predicted fraud probability")
    axis.set_ylabel("Density (log scale)")
    axis.set_title("Score distribution by true class")
    axis.legend()

    return _save(figure, output_dir / "14_score_distribution.png")


def plot_cv_stability(cv_results: pd.DataFrame, output_dir: Path) -> Path:
    """
    PR-AUC across expanding-window folds.

    A flat line means the model performs consistently through time. A
    downward slope would mean it gets worse as the data moves on, which is a
    warning about how quickly it will need retraining.
    """
    figure, axis = plt.subplots(figsize=(9, 5))

    axis.plot(
        cv_results["fold"],
        cv_results["pr_auc"],
        marker="o",
        color=ACCENT_COLOUR,
        linewidth=1.8,
    )
    mean_score = cv_results["pr_auc"].mean()
    axis.axhline(
        mean_score,
        color=NEUTRAL_COLOUR,
        linestyle="--",
        label=f"mean {mean_score:.4f}",
    )

    for _, row in cv_results.iterrows():
        axis.annotate(
            f"{row['pr_auc']:.4f}",
            (row["fold"], row["pr_auc"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
        )

    axis.set_xlabel("Fold (each trains on more history than the last)")
    axis.set_ylabel("PR-AUC")
    axis.set_title("Stability across expanding time windows")
    axis.set_xticks(cv_results["fold"].tolist())
    axis.legend()

    return _save(figure, output_dir / "15_cv_stability.png")
