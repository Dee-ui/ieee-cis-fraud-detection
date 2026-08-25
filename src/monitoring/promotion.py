"""
The gates a model must pass before it is allowed to serve.

Training produces a candidate. Deciding a candidate is fit for production is
a different decision, made against different evidence, and it should not
happen automatically as a side effect of a training run. Decision D-56.

Gate 1 alone would have stopped version 1 of this project's registry, which
is a 150-round quick-mode test model that registered itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.config import (
    PROMOTION_MAX_CV_SPREAD,
    PROMOTION_MIN_PR_AUC,
    PROMOTION_REGRESSION_TOLERANCE,
)


@dataclass
class Gate:
    """One check, its result, and enough detail to explain the result."""

    name: str
    passed: bool
    detail: str


def evaluate_gates(
    run_tags: dict,
    run_metrics: dict,
    metadata: dict | None,
    transformer_features: list[str] | None,
    production_metrics: dict | None,
) -> list[Gate]:
    """
    Run every gate and return the results, rather than stopping at the first
    failure. Seeing all six at once is far more useful than fixing them one
    at a time across six runs.
    """
    gates: list[Gate] = []

    # --- Gate 1: it came from a real run ------------------------------
    mode = run_tags.get("run_mode")
    gates.append(
        Gate(
            name="full training run",
            passed=mode == "full",
            detail=(
                f"run_mode = '{mode}'"
                if mode
                else "run_mode tag missing. Runs made before this tag existed "
                "can be corrected with MlflowClient().set_tag(...)"
            ),
        )
    )

    # --- Gate 2: it clears the quality floor ---------------------------
    pr_auc = run_metrics.get("selection_pr_auc") or run_metrics.get("valid_pr_auc")
    gates.append(
        Gate(
            name=f"PR-AUC at least {PROMOTION_MIN_PR_AUC}",
            passed=pr_auc is not None and pr_auc >= PROMOTION_MIN_PR_AUC,
            detail=f"PR-AUC = {pr_auc:.5f}" if pr_auc else "no PR-AUC recorded",
        )
    )

    # --- Gate 3: it is stable across time ------------------------------
    spread = run_metrics.get("cv_pr_auc_std")
    gates.append(
        Gate(
            name=f"cross-validation spread under {PROMOTION_MAX_CV_SPREAD}",
            passed=spread is not None and spread <= PROMOTION_MAX_CV_SPREAD,
            detail=f"spread = {spread:.5f}" if spread is not None else "not recorded",
        )
    )

    # --- Gate 4: it is not a step backwards ----------------------------
    if production_metrics:
        current = production_metrics.get("selection_pr_auc") or production_metrics.get(
            "valid_pr_auc"
        )
        acceptable = (
            pr_auc is not None
            and current is not None
            and pr_auc >= current - PROMOTION_REGRESSION_TOLERANCE
        )
        gates.append(
            Gate(
                name="no regression against production",
                passed=acceptable,
                detail=(
                    f"candidate {pr_auc:.5f} against production {current:.5f}"
                    if pr_auc and current
                    else "cannot compare"
                ),
            )
        )
    else:
        gates.append(
            Gate(
                name="no regression against production",
                passed=True,
                detail="nothing in production yet, so nothing to regress against",
            )
        )

    # --- Gate 5: it has a real operating threshold ---------------------
    threshold = (metadata or {}).get("chosen_threshold")
    gates.append(
        Gate(
            name="operating threshold chosen deliberately",
            passed=threshold is not None and abs(threshold - 0.5) > 1e-9,
            detail=(
                f"threshold = {threshold}"
                if threshold is not None
                else "no threshold recorded"
            ),
        )
    )

    # --- Gate 6: the model and the transformer still agree -------------
    # This catches the case where the feature engineer is rebuilt, the
    # feature count changes, and the model silently expects columns that no
    # longer exist. Nothing about that raises an error on its own.
    model_features = (metadata or {}).get("feature_names")
    if model_features and transformer_features:
        matches = list(model_features) == list(transformer_features)
        gates.append(
            Gate(
                name="feature list matches the transformer",
                passed=matches,
                detail=f"model expects {len(model_features)}, transformer "
                f"produces {len(transformer_features)}"
                + ("" if matches else "  MISMATCH"),
            )
        )
    else:
        gates.append(
            Gate(
                name="feature list matches the transformer",
                passed=False,
                detail="could not read one of the two feature lists",
            )
        )

    return gates


def all_passed(gates: list[Gate]) -> bool:
    return all(gate.passed for gate in gates)


def format_gates(gates: list[Gate]) -> str:
    lines = []
    for gate in gates:
        mark = "PASS" if gate.passed else "FAIL"
        lines.append(f"  [{mark}]  {gate.name}")
        lines.append(f"          {gate.detail}")
    return "\n".join(lines)
