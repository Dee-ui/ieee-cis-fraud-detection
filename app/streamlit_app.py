"""
The project dashboard.

Built for someone technical but not an ML specialist, looking at this for
about two minutes, possibly on a phone, alongside several other projects.
That is decision D-45, and it rules things out as much as in:

  - Everything comes from one JSON file, so the page opens instantly.
  - The money goes first, because it is the thing people repeat.
  - One interactive element, because reading is passive and clicking is not.
  - The full EDA stays in reports/. This is not the report.

Run locally:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_FILE = HERE / "dashboard_data.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FRAUD_COLOUR = "#c0392b"
SAFE_COLOUR = "#2c7fb8"
ACCENT_COLOUR = "#16a085"
MUTED_COLOUR = "#7f8c8d"

st.set_page_config(
    page_title="Fraud Detection, end to end",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_bundle() -> dict:
    """
    Read the bundle once and keep it.

    cache_data means Streamlit reruns the whole script on every interaction
    but does not re-read the file, so clicking a button is instant.
    """
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.0f}"


data = load_bundle()
model = data["model"]
dataset = data["dataset"]
costs = data["cost_assumptions"]

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("### IEEE-CIS Fraud Detection")
    st.caption("An end-to-end machine learning and MLOps project.")

    st.markdown(f"[Source code]({data['repo_url']})")
    st.markdown(f"[Live API]({data['api_url']}/docs)")
    st.markdown(f"[Model artefacts]({data['model_hub_url']})")

    st.divider()
    st.markdown("**Serving now**")
    st.write(f"{model['family']}, version {model['version']}")
    st.write(f"{model['n_features']} features")
    st.write(f"Threshold {model['threshold']:.4f}")
    st.caption(f"Bundle built {data['generated_utc'][:10]}")

# ---------------------------------------------------------
# 1. The headline
# ---------------------------------------------------------

st.title("Catching card fraud, and knowing what it is worth")

st.markdown(
    "A model that ranks card transactions by fraud risk, plus the engineering "
    "around it: a tested pipeline, drift monitoring, promotion gates, and a "
    "deployed API you can call right now."
)

first, second, third, fourth = st.columns(4)
first.metric(
    "Fraud caught",
    f"{dataset['recall_by_count']:.1%}",
    help="Share of fraudulent transactions flagged while reviewing only 2% of all traffic.",
)
second.metric(
    "Worth per year",
    money(model["savings_annual"]),
    help="Under a documented cost model with five stated assumptions. An order of magnitude, not a forecast.",
)
third.metric(
    "Kaggle private leaderboard",
    f"{dataset['kaggle_private']:.4f}",
    help="ROC-AUC, single model, no ensembling, no test-set leakage.",
)
fourth.metric(
    "Better than guessing",
    f"{model['pr_auc'] / dataset['fraud_rate']:.1f}x",
    help="PR-AUC against a random baseline equal to the fraud rate.",
)

st.divider()

# ---------------------------------------------------------
# 2. The problem
# ---------------------------------------------------------

st.header("The problem")

left, right = st.columns([2, 3])

with left:
    st.markdown(f"""
Fraud is **rare**: {dataset['frauds']:,} of {dataset['rows']:,} transactions,
a rate of **{dataset['fraud_rate']:.2%}**.

That rarity breaks the obvious measure. A model predicting "never fraud" is
**{1 - dataset['fraud_rate']:.1%} accurate** and catches nothing. Accuracy is
not reported anywhere in this project.

Fraud is also **expensive in two different ways**. A missed fraud is a direct
loss. A false alarm blocks a paying customer. So the system is tuned against a
real review capacity rather than optimised in the abstract.
        """)

with right:
    balance = pd.DataFrame(
        {
            "Class": ["Legitimate", "Fraud"],
            "Transactions": [
                dataset["rows"] - dataset["frauds"],
                dataset["frauds"],
            ],
        }
    )
    figure = px.bar(
        balance,
        x="Class",
        y="Transactions",
        color="Class",
        color_discrete_map={"Legitimate": SAFE_COLOUR, "Fraud": FRAUD_COLOUR},
        log_y=True,
        title="One fraud for every twenty-nine transactions (log scale)",
        text="Transactions",
    )
    figure.update_traces(texttemplate="%{text:,}", textposition="outside")
    figure.update_layout(showlegend=False, height=320, margin=dict(t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 3. How well it works
# ---------------------------------------------------------

st.header("How well it works")

st.caption(
    f"Measured on {dataset['train_end']} back to 2018-04-20: 118,108 transactions "
    "the model never saw during training, all of them later in time than "
    "everything it learned from."
)

left, right = st.columns(2)

with left:
    comparison = pd.DataFrame(data["model_comparison"]).sort_values("pr_auc")
    figure = px.bar(
        comparison,
        x="pr_auc",
        y="model",
        orientation="h",
        title="Five candidates, one metric",
        text="pr_auc",
        color_discrete_sequence=[ACCENT_COLOUR],
    )
    figure.add_vline(
        x=dataset["fraud_rate"],
        line_dash="dash",
        line_color=MUTED_COLOUR,
        annotation_text="random guessing",
    )
    figure.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    figure.update_layout(
        height=340,
        xaxis_title="PR-AUC",
        yaxis_title="",
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)

with right:
    thresholds = pd.DataFrame(data["threshold_analysis"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=thresholds["review_rate"] * 100,
            y=thresholds["recall"] * 100,
            mode="lines+markers",
            line=dict(color=FRAUD_COLOUR, width=3),
            name="fraud caught",
        )
    )
    figure.add_vline(
        x=costs["review_capacity"] * 100,
        line_dash="dash",
        line_color=MUTED_COLOUR,
        annotation_text="capacity",
    )
    figure.update_layout(
        title="Review more, catch more",
        xaxis_title="Share of transactions sent for manual review (%)",
        yaxis_title="Share of fraud caught (%)",
        height=340,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)

st.markdown(f"""
**What it is worth.** At a {costs['review_capacity']:.0%} review capacity the
model saves **{money(model['savings_window'])}** over a 42 day window, which
annualises to roughly **{money(model['savings_annual'])}**.

The five assumptions behind that: **${costs['review_per_case']:.2f}** per analyst
review, **${costs['chargeback_fee']:.2f}** chargeback fee per missed fraud,
**${costs['false_alarm_friction']:.2f}** friction per false alarm,
**{costs['recovery_rate']:.0%}** of flagged fraud actually prevented, and a team
able to review **{costs['review_capacity']:.0%}** of transactions. They are
assumptions, not figures from a business, so read the total as an order of
magnitude rather than a forecast.
    """)

st.warning(
    f"**One caveat that changes the number.** The model catches "
    f"**{dataset['recall_by_count']:.1%} of fraud by count** but only "
    f"**{dataset['recall_by_value']:.1%} by value**. Missed frauds average "
    f"${dataset['mean_missed_fraud']:.0f} against ${dataset['mean_caught_fraud']:.0f} "
    f"for caught ones, because a large fraudulent purchase looks much like a "
    f"large legitimate one. The cost model handles this correctly, but "
    f"multiplying the recall figure by total fraud losses would overstate the "
    f"benefit by about 43%."
)

with st.expander("What the model actually looks at"):
    importance = (
        pd.DataFrame(data["feature_importance"]).head(15).sort_values("mean_abs_shap")
    )
    figure = px.bar(
        importance,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color_discrete_sequence=[SAFE_COLOUR],
        title="Top 15 features by average influence on a prediction",
    )
    figure.update_layout(
        height=440,
        xaxis_title="mean absolute SHAP",
        yaxis_title="",
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Mean influence understates features that matter rarely but decisively. "
        "One column here holds the same value on 99.7% of rows, yet nearly half "
        "of the remaining rows are fraudulent. Average importance ranks it 259th "
        "of 284."
    )

st.divider()

# ---------------------------------------------------------
# 4. Try it
# ---------------------------------------------------------

st.header("Try it")

st.markdown(
    f"This calls the live API at `{data['api_url']}`. Most transaction fields "
    "are optional: anything you leave out is treated as unknown, which the "
    "model handles natively at every split."
)

presets = {
    "Ordinary purchase": data["example_transaction"],
    "Unfamiliar card, higher amount": {
        **data["example_transaction"],
        "TransactionID": 3663550,
        "TransactionAmt": 892.5,
        "card1": 17188,
        "ProductCD": "C",
        "card6": "credit",
        "P_emaildomain": "mail.com",
        "DeviceType": "mobile",
        "C13": 24,
        "C14": 18,
        "D1": 0,
    },
}

choice = st.selectbox("Start from", list(presets))
payload = st.text_area(
    "Transaction (edit freely)",
    value=json.dumps(presets[choice], indent=2),
    height=260,
)

go_button, note = st.columns([1, 3])
with go_button:
    pressed = st.button("Score it", type="primary", use_container_width=True)
with note:
    st.caption(
        "Free-tier hosting sleeps when idle. The first call can take up to a "
        "minute to wake the service. Later calls are immediate."
    )

result = None
if pressed:
    try:
        transaction = json.loads(payload)
    except json.JSONDecodeError as error:
        st.error(f"That is not valid JSON: {error}")
        transaction = None

    if transaction is not None:
        with st.spinner("Waking the service and scoring ..."):
            try:
                response = requests.post(
                    f"{data['api_url']}/predict",
                    json={"transaction": transaction, "explain": True},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
            except Exception as error:  # noqa: BLE001
                st.info(
                    f"The live service did not answer ({type(error).__name__}). "
                    "Showing a saved response instead so you can still see the shape."
                )
                result = data["example_response"]
else:
    result = data["example_response"]
    st.caption("Showing a saved response. Press the button for a live one.")

if result:
    score_column, decision_column, threshold_column = st.columns(3)
    probability = result["fraud_probability"]

    score_column.metric("Fraud probability", f"{probability:.4f}")
    decision_column.metric(
        "Decision",
        result["decision"].upper(),
        delta=(
            "above threshold" if result["decision"] == "review" else "below threshold"
        ),
        delta_color="inverse" if result["decision"] == "review" else "normal",
    )
    threshold_column.metric("Threshold", f"{result['threshold']:.4f}")

    if result.get("explanation"):
        explanation = pd.DataFrame(result["explanation"])
        explanation["direction"] = explanation["contribution"].apply(
            lambda value: "towards fraud" if value > 0 else "away from fraud"
        )
        figure = px.bar(
            explanation.sort_values("contribution"),
            x="contribution",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "towards fraud": FRAUD_COLOUR,
                "away from fraud": SAFE_COLOUR,
            },
            title="What drove this particular score",
        )
        figure.update_layout(
            height=320,
            yaxis_title="",
            xaxis_title="SHAP contribution",
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(figure, use_container_width=True)

st.caption(
    "The threshold is 0.4222, not 0.5. It was chosen by a cost model at a 2% "
    "review capacity. Nothing about 0.5 relates to this problem."
)

st.divider()

# ---------------------------------------------------------
# 5. Is it still working
# ---------------------------------------------------------

st.header("Is it still working?")

st.markdown(
    "A model is not finished when it is trained. The test period runs six "
    "months past the end of training and has no labels, which is exactly the "
    "position production puts you in: you score for weeks before you learn "
    "whether the scores were any good. So the inputs get watched instead."
)

weekly = pd.DataFrame(data["weekly_performance"])
monthly = pd.DataFrame(data["monthly_drift"])

left, right = st.columns(2)

with left:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=weekly["period"],
            y=weekly["pr_auc_lift"],
            mode="lines+markers",
            line=dict(color=ACCENT_COLOUR, width=3),
            name="lift",
        )
    )
    figure.update_layout(
        title="Advantage over guessing, week by week",
        yaxis_title="PR-AUC lift over that week's own baseline",
        xaxis_title="",
        height=340,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Raw PR-AUC across these weeks declines about 3%, which reads as noise. "
        "Measured as lift over each week's own fraud rate, which moved from "
        "2.98% to 4.00%, the decline is 21%."
    )

with right:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=monthly["period"],
            y=monthly["weighted_psi"],
            mode="lines+markers",
            line=dict(color=FRAUD_COLOUR, width=3),
            name="weighted PSI",
        )
    )
    figure.add_hline(
        y=data["thresholds"]["retrain_weighted_psi"],
        line_dash="dash",
        line_color=MUTED_COLOUR,
        annotation_text="retrain trigger",
    )
    figure.update_layout(
        title="Input drift, weighted by what the model relies on",
        yaxis_title="weighted PSI",
        xaxis_title="",
        height=340,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Rises 53% across six months and now sits at 83% of the retrain "
        "trigger. Verdict: watch, retrain soon."
    )

figure = px.bar(
    monthly,
    x="period",
    y=monthly["alert_rate"] * 100,
    title="How much work lands in the review queue",
    color_discrete_sequence=[SAFE_COLOUR],
)
figure.add_hline(
    y=model["review_rate"] * 100,
    line_dash="dash",
    line_color=FRAUD_COLOUR,
    annotation_text="what the team is staffed for",
)
figure.update_layout(
    height=320,
    yaxis_title="transactions alerted (%)",
    xaxis_title="",
    margin=dict(t=50, b=10),
)
st.plotly_chart(figure, use_container_width=True)

st.markdown("""
**Nothing about the model changed and the threshold is fixed.** The whole
swing, 1.77 times more work in July than in November, is the data moving
underneath a fixed line. That is invisible from any accuracy measure and it is
the number an operations manager feels every day.
    """)

with st.expander("The feature that explains most of it"):
    st.markdown("""
One feature trips the alarm in **every single month**: `uid_freq`, the model's
8th most important input. It counts how often a customer fingerprint appeared
during training.

| Month | Drift score | Missing values |
|-------|------------|----------------|
| July | 1.35 | **0.0%** |
| December | **2.46** | **0.0%** |

**Nothing is ever missing.** By December most transactions carry a fingerprint
that did not exist during training, so the count comes back zero. The feature
has not broken. It has gone quiet, which is worse, because nothing complains.

A missing-value check would have reported everything healthy for six months
running. Comparing whole distributions catches it immediately, which is why
that is the primary drift signal here.
        """)
    top_drift = pd.DataFrame(data["top_drift"]).head(10)
    st.dataframe(top_drift, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------
# 6. How it was built
# ---------------------------------------------------------

st.header("How it was built")

one, two, three = st.columns(3)

with one:
    st.markdown(f"""
**Data**

{dataset['rows']:,} transactions across two tables joined on a shared key.
{dataset['features_raw']} raw columns reduced to **{dataset['features_final']}**
engineered features.

339 of the columns are anonymised. They turned out to fall into 15 groups that
go blank on identical rows, and eight of those groups weave through each other's
numbering, so the structure is invisible unless you compare the actual patterns.
        """)

with two:
    st.markdown("""
**Not leaking**

Validation is the last 20% of the training period by time, never a random split,
because the real test set is 30 days in the future.

Every learned transformation is fitted on the training portion only and saved as
an object, so training and serving cannot drift apart. A test asserts that
scoring one transaction gives the same answer as scoring a batch containing it.
        """)

with three:
    st.markdown("""
**Staying honest**

Tests on synthetic data, so they run anywhere. Linting, formatting, tests and a
container build on every push.

A model reaches production only by passing six gates. One exists because a
throwaway test model once registered itself.
        """)

architecture = FIGURES_DIR / "20_architecture.png"
if architecture.exists():
    st.image(str(architecture), caption="End to end", use_container_width=True)

st.markdown(f"""
**Stack.** Python 3.11, pandas, LightGBM, scikit-learn, MLflow, SHAP, DVC,
pytest, ruff, GitHub Actions, FastAPI, Docker, Render, Streamlit.

[Source code]({data['repo_url']}) ·
[Live API]({data['api_url']}/docs) ·
[Model artefacts]({data['model_hub_url']})
    """)
