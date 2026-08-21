# Step 4: Model Training and Experiment Tracking
### A cost model, MLflow, four candidates, an ablation, time-aware validation, and a threshold that means something

**Project:** IEEE-CIS Fraud Detection
**Repository:** https://github.com/Dee-ui/ieee-cis-fraud-detection
**Local path:** `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`
**Platform:** Windows, VS Code, PowerShell, Python 3.11.9
**Estimated time:** 4 to 5 hours, of which 45 to 75 minutes is the machine running
**Step 4 of 7**

---

## 0. How to use this document

Work top to bottom.

Sections 1 and 2 read your Step 3 results, including one finding that changes the plan. Section 3 builds the cost model you asked for, which is the most important section for the PM track. Section 5 explains MLflow from scratch. Sections 6 to 12 are the code. Section 13 runs it.

Section 21 has the checklist. Do not start Step 5 until every box ticks.

---

## 1. Where Step 3 left you

Your run completed in 2 minutes 25 seconds with all four verification checks passing.

| Item | Result |
|------|--------|
| Input columns | 435 |
| Final features | **284** |
| V columns | 337 reduced to **137** |
| Dropped, single value | 0 |
| Dropped, near-constant | 2 |
| Rescued | 22 |
| Split boundary | 2018-04-20, TransactionDT 12,192,854 |
| Train portion | 472,432 rows, 16,599 frauds, 3.5135% |
| Valid portion | 118,108 rows, 4,064 frauds, 3.4409% |
| Unseen lookups in test | 6.81% |
| Transformer file | 28.0 MB |

DVC is set up and `dvc pull` restored a deleted file successfully. Every checklist item ticked.

**Feature composition:**

| Kind | Count |
|------|-------|
| base_numeric | 199 |
| category_code | 38 |
| aggregate | 18 |
| frequency | 18 |
| derived_amount | 3 |
| derived_screen | 3 |
| derived_time | 2 |
| derived_match | 2 |
| derived_email | 1 |

The arithmetic reconciles exactly: 435 columns minus 3 passthrough gives 432 candidates, minus 2 dropped gives 430 survivors, of which 337 are V columns and 93 are not. The 337 V columns reduce to 137, so 93 plus 137 gives 230 base columns. Of those, 31 are text and 199 are numeric. The 31 text columns plus 7 derived text columns give the 38 category codes. Everything adds to 284.

---

## 2. Reading your Step 3 results

### 2.1 The rescue rule earned its place

I was uneasy about the near-constant rule in Step 3 and built the rescue check because dropping a column that is 99% one value felt risky when the target only occurs 3.5% of the time. Your results show that unease was justified, and by a wide margin.

Only **2 columns** were actually dropped. **22 were rescued.** Here are the most striking:

| Column | Dominant value | Share | Rare rows | Fraud rate among rare rows | Lift |
|--------|----------------|-------|-----------|---------------------------|------|
| V111 | 1.0 | 99.71% | 1,370 | **46.35%** | 13.2x |
| V113 | 1.0 | 99.65% | 1,645 | 39.51% | 11.2x |
| V117 | 1.0 | 99.88% | 578 | 31.14% | 8.9x |
| V112 | 1.0 | 99.49% | 2,431 | 29.25% | 8.3x |
| V108 | 1.0 | 99.52% | 2,283 | 28.03% | 8.0x |
| C3 | 0.0 | 99.60% | 1,872 | **0.053%** | 0.015x |

**V111 is the headline.** It holds 1.0 on 99.71% of rows. A blanket "drop anything 99% constant" rule, which is what most tutorials recommend, would have thrown it away without comment. But on the 1,370 rows where it is not 1.0, **nearly half are fraudulent**. Against a base rate of 3.5%, that is one of the sharpest single signals in the entire dataset, and it was one line of threshold away from being deleted.

**C3 is the mirror image and just as valuable.** When C3 is not 0, the fraud rate collapses to 0.053%, which is about one sixty-sixth of the base rate. That is a strong *safety* signal. It only survived because the rescue rule was written to trigger in both directions, on rare values that are unusually risky and rare values that are unusually safe. Had I only checked for elevated fraud, C3 would be gone.

**The nine identity columns were flagged and then rescued.** In Step 3 I said the near-constant rule would catch `id_07`, `id_08`, `id_21` through `id_27`. It did flag all nine, and the rescue kept every one of them at roughly 2.2x lift. So my description of what would happen was half right: they were caught, but they were not removed. That is the rule working exactly as designed.

**The two that were dropped were dropped for the right reason.** V107 had 189 rows that were not 1.0, and V305 had 16. Both fell below the 500 row minimum, so there were not enough examples to judge whether the rare values meant anything. Dropping on insufficient evidence rather than on a bad fraud rate is the correct call.

The practical lesson for the PM track: a rule that looks sensible in the abstract can be badly wrong on a specific dataset. What made the difference here was not a cleverer rule, it was writing every decision to a file with the evidence attached so it could be checked afterwards.

### 2.2 The one real problem: the uid features do not transfer to test

This is the finding that changes what Step 4 does.

I told you in Step 3 to compare `missing_pct_train` against `missing_pct_test` in the manifest. Doing that turns up exactly six features with a gap over 20 percentage points, and all six are the uid aggregates:

| Feature | Missing in train | Missing in test | Gap |
|---------|------------------|-----------------|-----|
| `TransactionAmt_mean_by_uid` | 11.30% | **81.94%** | 70.6 |
| `TransactionAmt_ratio_to_uid_mean` | 11.30% | **81.94%** | 70.6 |
| `D15_mean_by_uid` | 20.63% | **82.20%** | 61.6 |
| `TransactionAmt_std_by_uid` | 30.54% | **84.96%** | 54.4 |
| `D15_std_by_uid` | 36.42% | **85.23%** | 48.8 |
| `D15_ratio_to_uid_mean` | 39.72% | **84.01%** | 44.3 |

Everything else in the table is well behaved. The seventh largest gap is 5.96 points, on `TransactionAmt_mean_by_card1_addr1`. So the problem is sharply bounded: **6 features out of 284, all from one family.**

**Why it happens.** The uid combines `card1`, `addr1`, and the card's first-seen day. For that to produce a usable group average on a test row, the exact same fingerprint has to have appeared in the training portion. Test runs from July to December 2018, starting 30 days after training ends and continuing for six months. Over that stretch most customers are either genuinely new or their fingerprint has shifted, so 82% of test rows land on a uid the model has never encountered and get a blank average.

**Why it matters.** The model will be trained on data where these six features are usually present and validated on data where they are mostly present, then asked to score data where they are almost always absent. It may learn to lean on them. Tree models handle blanks gracefully, so nothing will crash, but a feature that is informative in training and absent in production is dead weight at best and actively misleading at worst.

**What we do about it.** We measure, rather than guess. Section 12 runs a deliberate experiment: train the winning model twice, once with all 284 features and once with those 6 removed, and compare.

The decision rule is set **now, before seeing the result**, because a rule chosen after the fact is not a rule:

> If the model without the uid features scores within **0.005 PR-AUC** of the model with them, we drop them and ship the smaller model. A negligible validation gain is not worth carrying six features that are unavailable for 82% of the data we actually want to score.
>
> If the gap is larger than 0.005, we keep them, and flag all six for close watching in the Step 5 drift monitoring.

Deciding the threshold in advance and writing it down is the difference between an experiment and a justification. It is also a strong thing to be able to say out loud in an interview.

### 2.3 The V reduction worked, and unevenly, which is informative

337 V columns became 137, a 59% cut. But the rate varies a lot by block:

| Block | Before | After | Kept |
|-------|--------|-------|------|
| 14 | 11 | 2 | 18% |
| 4 | 31 | 8 | 26% |
| 5, 6, 7 | 23, 22, 20 | 8, 8, 8 | 35 to 40% |
| 10 | 18 | 6 | 33% |
| 1 | 46 | 13 | 28% |
| 2 | 42 | 24 | **57%** |
| 3 | 31 | 15 | 48% |
| 13 | 11 | 7 | 64% |

Block 14 collapsed from 11 columns to 2, which means nine of those columns were saying almost exactly the same thing as one of the other two. Block 2 held on to 24 of 42, so those columns are genuinely measuring different things despite sharing a source.

That spread is the point. A flat rule such as "keep every fifth V column" would have over-cut block 2 and under-cut block 14. The structure decided how much to cut in each place, which is what we wanted.

### 2.4 The split landed well

| Portion | Rows | Frauds | Fraud rate | Period |
|---------|------|--------|------------|--------|
| train | 472,432 | 16,599 | 3.5135% | 2017-12-01 to 2018-04-20 |
| valid | 118,108 | 4,064 | 3.4409% | 2018-04-20 to 2018-05-31 |

The validation fraud rate of 3.4409% is close to the training rate of 3.5135% but not identical, which is exactly right. A time split does not preserve the fraud rate, and if it had come out identical I would suspect the split was not really by time.

4,064 fraud cases in validation is comfortably enough to measure PR-AUC reliably. That was the whole reason for choosing this dataset.

The validation window is about 41 days, which we use in Section 3 to annualise the cost figures.

### 2.5 Fix the FutureWarning before running anything else

Your run printed this:

```
src/features/engineer.py:252: FutureWarning: The behavior of DataFrame concatenation
with empty or all-NA entries is deprecated.
```

**What causes it.** No columns had a single distinct value, so `constant_columns` was an empty list. The code still built a DataFrame from that empty list, producing a table with zero rows but columns whose types pandas could not infer. Concatenating that empty, typeless table with the real one is the deprecated behaviour.

**Why fix it now.** Today it is a warning. In a future pandas release the behaviour changes, and when it does the code will not error, it will quietly produce a different result. Warnings that predict silent future changes are the ones worth clearing immediately.

Open `src/features/engineer.py` and find this block:

```python
        constant_records = pd.DataFrame(
            {
                "column": constant_columns,
                "dominant_value": None,
                "dominant_share": 1.0,
                "rare_rows": 0,
                "rare_fraud_rate": None,
                "fraud_lift": None,
                "decision": "drop",
                "reason": "single distinct value",
            }
        )
        self.column_decisions_ = pd.concat(
            [constant_records, assessment], ignore_index=True
        )
```

Replace it with:

```python
        # Build the constant-column records as a list of dictionaries and only
        # turn them into a table when there is something to put in it.
        #
        # Building an empty DataFrame from an empty list produces a table with
        # no rows and no usable column types. Concatenating that is deprecated
        # in pandas, and in a future release it will silently change the types
        # of the result rather than warning. Skipping the concat entirely when
        # there is nothing to add avoids the problem instead of suppressing it.
        constant_records = [
            {
                "column": column,
                "dominant_value": None,
                "dominant_share": 1.0,
                "rare_rows": 0,
                "rare_fraud_rate": None,
                "fraud_lift": None,
                "decision": "drop",
                "reason": "single distinct value",
            }
            for column in constant_columns
        ]

        if constant_records:
            self.column_decisions_ = pd.concat(
                [pd.DataFrame(constant_records), assessment], ignore_index=True
            )
        else:
            self.column_decisions_ = assessment.reset_index(drop=True)
```

You do not need to re-run Step 3. The output was correct, only the route to it was noisy. The fix takes effect next time the stage runs.

---

## 3. The cost model

You asked me to build realistic cost figures rather than wait for the business to supply them. This section does that. It is the most valuable thing in Step 4 for the PM track, because it is what turns "PR-AUC 0.7" into a sentence a finance director can act on.

### 3.1 Read this first

**These numbers are assumptions, not measurements.** Nobody at a real company has told us what a missed fraud costs. I have built a model from published industry norms and from the structure of this dataset, and every figure below comes with the reasoning behind it so it can be argued with and replaced.

That framing matters. A cost model presented as fact, when it is actually a set of assumptions, is worse than no cost model, because it invites decisions that nobody has stress-tested. Presented honestly, as "here is a defensible starting point, and here is exactly which number to change when you learn better", it is one of the most useful things an analyst can produce.

Everything lives in `config/config.py`, so changing one figure and re-running gives a full updated answer in minutes.

### 3.2 The operating model

There are two ways a fraud system can act on a risky transaction.

**Block it automatically.** Fast, but every mistake is a real customer whose payment was declined. Expensive in goodwill and hard to price.

**Send it to a human review queue.** Slower, but mistakes are usually invisible to the customer, because the transaction is checked and released. The constraint becomes how many cases the team can handle per day.

We model the **review queue**, for three reasons. It is what most mid-sized operations actually do. It matches the "recall at a 1% review rate" metric we chose back in Step 2. And its costs are far easier to defend, because analyst time has a known price while customer goodwill does not.

### 3.3 The four outcomes and what each one costs

Every transaction lands in one of four boxes.

| Outcome | What happens | Cost |
|---------|--------------|------|
| **Missed fraud** (not flagged, was fraud) | Goes through. Customer disputes it later. | Transaction amount + chargeback fee |
| **Caught fraud** (flagged, was fraud) | Analyst reviews and stops it, usually | Review cost + the share not recovered |
| **False alarm** (flagged, was legitimate) | Analyst reviews and releases it | Review cost + small friction cost |
| **Correct pass** (not flagged, legitimate) | Nothing happens | Nothing |

### 3.4 The five numbers, and where each comes from

**Analyst review cost: $4.00 per case.**
A fully loaded fraud analyst, meaning salary plus employment costs plus tooling, runs around $60,000 a year. Across roughly 2,080 working hours that is about $29 an hour. A routine review takes about five minutes, which is $2.40. Rounding up to $4.00 covers supervision, quality checks, and the cases that need a customer phone call.

**Chargeback administration fee: $25.00 per missed fraud.**
When a customer disputes a transaction, the card networks charge a per-dispute fee on top of the money being clawed back. Published fees run from roughly $15 to $40 depending on the network and the merchant's dispute history. $25 sits in the middle.

**Customer friction cost: $1.00 per false alarm.**
Most reviewed-and-released transactions are invisible to the customer. Some are not: a delayed payment, a verification call, occasionally an abandoned purchase. $1.00 is the expected value across all of them. This is the softest number in the model and the one most worth replacing with real data.

**Fraud recovery rate: 90%.**
Flagging a fraud is not the same as stopping it. Reviews take time, some cases are judged wrongly, and some transactions have already settled. We assume 90% of flagged fraud is genuinely prevented and 10% still goes through.

**Review capacity: 2% of transactions.**
The cost model on its own would happily recommend reviewing 15% of transactions if that minimised cost. No real team can. 2% is roughly one transaction in fifty, which on this data volume is about 65 cases a day, or one analyst working a full shift. The code reports both the unconstrained optimum and the best point within capacity, so you can see what the constraint actually costs you.

### 3.5 The arithmetic

For a given threshold:

```
cost of missed fraud   = sum of their amounts + (count x $25)
cost of caught fraud   = (count x $4) + 10% x (their amounts + count x $25)
cost of false alarms   = count x ($4 + $1)
cost of correct passes = 0

total cost = the three above added together
```

The comparison point is doing nothing at all. With no model, every fraud is missed:

```
baseline cost = all fraud amounts + (all fraud count x $25)
savings = baseline cost - total cost
```

Note that the cost is **weighted by the actual transaction amount**, not a flat penalty per fraud. A missed $2,000 fraud costs more than a missed $20 one. That is both more realistic and more useful, because it means the optimal threshold naturally leans towards catching expensive fraud. A flat cost matrix, which is what most tutorials use, cannot express that.

The code computes this exactly, at every possible threshold, using cumulative sums over the score-sorted validation set. No grid search and no approximation.

### 3.6 What comes out

Your run produces, all computed from your own validation data:

- The **cost curve**: total cost against review rate, with the minimum marked
- The **unconstrained optimal threshold** and what it would cost
- The **best point within 2% capacity**, and the gap between the two
- **Savings per 1,000 transactions**
- An **annualised figure**, scaled from the validation window's actual length to a full year using your real transaction rate

The last one is the sentence for the PM deck. Something in the shape of: at a 1% review rate this model prevents X dollars of fraud loss per year, at a review cost of Y, for a net saving of Z.

### 3.7 One honest caveat to carry into the deck

These savings are calculated on a validation set from 2018, using assumed unit costs, with a model that has never faced live traffic. Treat the figure as an order of magnitude, not a forecast.

The right way to present it: "under these five assumptions, which are written down and can be changed, the model is worth roughly this much. Here is the sensitivity to each assumption." Anyone who presents a single number without that framing is overselling, and experienced reviewers notice.

---

## 4. Decisions made in this step

| ID | Decision | Why |
|----|----------|-----|
| D-34 | A cost model with five explicitly stated assumptions, stored in config, drives the threshold choice | Turns an abstract metric into money. Keeping the assumptions in config means challenging one and re-running takes minutes. |
| D-35 | Costs are weighted by the actual transaction amount, not a flat penalty per fraud | A missed $2,000 fraud is not the same as a missed $20 one. Amount weighting makes the optimal threshold naturally favour expensive fraud, which a flat cost matrix cannot express. |
| D-36 | The uid features get a pre-registered ablation, with the decision threshold set at 0.005 PR-AUC before the result is seen | Those six features are missing on 82% of test rows. A rule chosen after seeing the result is a justification, not a rule. |
| D-37 | Four candidates: a stratified dummy, logistic regression, LightGBM, XGBoost, CatBoost | The dummy gives the true floor. Logistic regression shows what a classical model achieves, so the boosters have to earn their complexity rather than being assumed to be better. |
| D-38 | No class weighting or resampling | We care about ranking, and the threshold is chosen separately by the cost model. Weighting mostly shifts probability values without improving the ordering, and it makes the probabilities harder to interpret. Revisit only if the ranking metrics disappoint. |
| D-39 | Category codes are treated as ordinary numbers, not declared as categorical to the boosters | We already supply frequency counts for the same columns, so the information is available in a form that cannot overfit. Native categorical handling on a 1,786-value column such as `DeviceInfo` tends to memorise. |
| D-40 | Early stopping on validation PR-AUC. Time-aware cross-validation runs afterwards with the iteration count fixed | Early stopping inside every CV fold makes each fold slightly optimistic about itself. Fixing the count first keeps the CV honest as a stability check. |
| D-41 | The final model is retrained on all labelled data, with the iteration count scaled by the extra data | Validation picks the settings; the final model should still see every labelled row. Scaling iterations by the row ratio, about 1.25x here, is the standard adjustment. |
| D-42 | MLflow tracking URI built with `.as_posix()` | On Windows the path contains backslashes, which are unreliable inside a SQLAlchemy database URL. Forward slashes work everywhere. |
| D-43 | The chosen model is registered in the MLflow registry under an alias, not a stage | MLflow 3 deprecated stages in favour of aliases. Step 5 promotes by moving the alias. |
| D-44 | Deployment target for Step 6 is Hugging Face Spaces using the Docker SDK | Answers Q-04. Reasoning in Section 18. |
| D-45 | The Step 7 dashboard is built for a hiring manager reading it cold, in under two minutes | Answers Q-12. Reasoning in Section 19. |
| D-46 | A Kaggle late submission is produced, scored on ROC-AUC | Answers Q-05. Free, and it gives one externally verified number that nobody has to take our word for. |

---

## 5. MLflow, from scratch

You have MLflow 3.15.1 installed. This section explains what it is before we use it.

### 5.1 The problem it solves

You are about to train five models, then an ablation, then four cross-validation folds, then a final model. Each has settings, produces metrics, and generates a model file.

Without a tool, that state lives in your terminal scrollback and your memory. Two days later somebody asks why you chose LightGBM over CatBoost and what learning rate you used, and the honest answer is that you are not sure. Worse, you cannot reproduce it.

MLflow records every run automatically: what settings went in, what numbers came out, what files were produced, which Git commit was checked out. It stores all of it in a database you can query and browse in a web page.

### 5.2 The four things it records

**Parameters.** Settings you chose before training: learning rate, number of leaves, random seed. Fixed for a run.

**Metrics.** Numbers that came out: PR-AUC, ROC-AUC, recall at 1%. Can be logged repeatedly over training to give a curve.

**Artifacts.** Files: the model itself, charts, the feature list, the cost curve.

**Tags.** Labels for filtering: which model family, which experiment phase.

A **run** is one training attempt holding all four. An **experiment** is a named collection of runs. Ours is `ieee-cis-fraud-detection`.

### 5.3 Where it all goes

MLflow needs a place to store this. Ours is a SQLite database file, `mlflow.db`, in the project root. SQLite is a complete database inside a single file with no server to install, which suits a local project and still supports the model registry.

One Windows-specific problem, which is decision D-42. The tracking location is written as a URL, like `sqlite:///C:/Users/Name/project/mlflow.db`. Building that with a Windows `Path` gives backslashes, and backslashes inside a database URL are interpreted inconsistently. `.as_posix()` converts a Windows path to forward slashes, which works everywhere. Section 6 fixes it.

### 5.4 The model registry

Tracking records experiments. The **registry** is different: it is the catalogue of models you might actually deploy.

You register a model, it gets a version number, and you attach an **alias** such as `candidate` or `production` to point at a specific version. Step 5 promotes a model by moving the alias. Step 6 loads whatever the alias currently points at, so deploying a new model means moving a pointer rather than editing code.

MLflow 2 used named "stages" for this. MLflow 3 deprecated stages in favour of aliases, which are more flexible. Since you are on 3.15.1, we use aliases. That is D-43, and it is worth knowing about because most tutorials you will find online still show the old way.

### 5.5 One compatibility note

MLflow 3 changed how models are logged. The old `artifact_path` argument was replaced by `name`. The old one still works but warns.

Rather than guess which your exact patch release expects, the code in Section 9 inspects the function and uses whichever it accepts. That is a useful pattern in general: when a library is mid-transition, check what is actually there rather than assuming.

---

## 6. Update `config/config.py`

### 6.1 First, fix the MLflow URI

Find the existing block:

```python
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
)
MLFLOW_EXPERIMENT_NAME = "ieee-cis-fraud-detection"
```

Replace it with:

```python
# A database URL must use forward slashes. Building it from a Windows Path
# produces backslashes, which SQLAlchemy handles inconsistently. .as_posix()
# converts "C:\Users\...\mlflow.db" into "C:/Users/.../mlflow.db", which is
# understood on every platform. This is decision D-42.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)
MLFLOW_EXPERIMENT_NAME = "ieee-cis-fraud-detection"
REGISTERED_MODEL_NAME = "ieee-cis-fraud-detector"
MODEL_ALIAS_CANDIDATE = "candidate"
```

### 6.2 Then append the Step 4 settings

Add this at the end of the file, before `ensure_directories`.

```python
# =========================================================
# STEP 4: MODEL TRAINING
# =========================================================

# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

FINAL_MODEL_FILE = MODELS_DIR / "final_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "final_model_metadata.json"

MODEL_COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"
THRESHOLD_ANALYSIS_FILE = REPORTS_DIR / "threshold_analysis.csv"
COST_CURVE_FILE = REPORTS_DIR / "cost_curve.csv"
CV_RESULTS_FILE = REPORTS_DIR / "cv_results.csv"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "feature_importance.csv"
TRAINING_SUMMARY_FILE = REPORTS_DIR / "training_summary.md"

KAGGLE_SUBMISSION_FILE = PROCESSED_DATA_DIR / "kaggle_submission.csv"


# ---------------------------------------------------------
# The cost model. See step4.md section 3.
#
# THESE ARE ASSUMPTIONS, not figures supplied by a business. Each one has
# stated reasoning behind it. Change any of them and re-run to get a fully
# updated answer.
# ---------------------------------------------------------

# Fully loaded analyst at about $60k a year is roughly $29 an hour. A review
# takes about five minutes, so $2.40. Rounded up for supervision and the
# cases that need a customer call.
COST_REVIEW_PER_CASE = 4.00

# Card networks charge a per-dispute fee on top of the money clawed back.
# Published fees run from roughly $15 to $40.
COST_CHARGEBACK_FEE = 25.00

# Expected cost of holding and releasing a legitimate customer. The softest
# number in the model and the first one to replace with real data.
COST_FALSE_ALARM_FRICTION = 1.00

# Flagging fraud is not the same as stopping it. Some cases are judged
# wrongly and some have already settled.
FRAUD_RECOVERY_RATE = 0.90

# The team can review about one transaction in fifty. The cost model would
# otherwise happily recommend reviewing 15%, which no real team can do.
REVIEW_CAPACITY_RATE = 0.02

# Review rates reported in every summary, for comparison.
HEADLINE_REVIEW_RATES = [0.005, 0.01, 0.02, 0.05]


# ---------------------------------------------------------
# Training settings
# ---------------------------------------------------------

EARLY_STOPPING_ROUNDS = 100
MAX_BOOSTING_ROUNDS = 1500
QUICK_BOOSTING_ROUNDS = 150      # used by run.py --quick

# Expanding-window cross-validation folds, run after a winner is chosen.
CV_N_SPLITS = 4

# Rows sampled for SHAP. Explaining all 118,108 validation rows would take
# far longer and tell you nothing extra.
SHAP_SAMPLE_SIZE = 5000

# Any feature whose name contains one of these belongs to the uid family.
# Quarantined here so the ablation in D-36 can find them by rule rather than
# by a hand-maintained list that would go stale.
UID_FEATURE_MARKERS = ["_by_uid", "_to_uid_", "uid_freq"]

# The ablation decision threshold, set in advance. See D-36.
UID_ABLATION_TOLERANCE = 0.005
```

### 6.3 Check it

```powershell
python -c "from config.config import MLFLOW_TRACKING_URI, COST_CHARGEBACK_FEE, UID_FEATURE_MARKERS; print(MLFLOW_TRACKING_URI); print(COST_CHARGEBACK_FEE, UID_FEATURE_MARKERS)"
```

**Expected:** a URI with forward slashes, then `25.0` and the three markers. If you see backslashes, the `.as_posix()` edit did not take.

---

## 7. Create `src/utils/metrics.py`

### 7.1 What is in here

Every number this project reports, in one place: the ranking metrics, the review-rate metrics, and the cost model.

Putting them in one module rather than scattering them means every model is scored identically. When two models are compared, any difference is the model, not the measuring.

### 7.2 How the cost curve is computed

Worth understanding, because it is the cleverest piece of code in the step and it is only about fifteen lines.

The obvious approach is to pick 200 thresholds, and for each one loop over all 118,108 rows counting outcomes. That is 23 million operations and it only samples 200 of the possible thresholds.

The better approach: sort the rows once by score, highest first. Now "flag the top k" for every possible k is just a running total. Cumulative sums give the number of frauds caught and the value of fraud caught at every k in one pass. Everything else is arithmetic on those two arrays.

The result is exact rather than sampled, covers every possible threshold, and runs in well under a second.

### 7.3 The file

```python
"""
Every metric this project reports, in one place.

Three groups:
  1. Ranking metrics: how well the model orders transactions by risk
  2. Review-rate metrics: what you catch at a given manual review budget
  3. The cost model: what a threshold is actually worth in money

Keeping them together means every model is measured identically, so any
difference between two models is the model and not the measuring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(y_true, scores) -> dict:
    """
    How well does the model order transactions by risk?

    PR-AUC (average precision) is the primary metric. Its baseline is the
    fraud rate itself, about 0.035 here, so the lift figure tells you how
    many times better than guessing the model is.

    ROC-AUC is reported because it was the competition metric and is widely
    understood. It is less useful here, because with 569,877 legitimate
    transactions the false positive rate barely moves no matter how many
    real customers you wrongly flag.
    """
    y = np.asarray(y_true)
    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, scores))

    return {
        "pr_auc": pr_auc,
        "pr_auc_baseline": prevalence,
        "pr_auc_lift": pr_auc / prevalence if prevalence else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def review_rate_metrics(y_true, scores, review_rate: float) -> dict:
    """
    If the team can review this share of transactions, what do they catch?

    This is the metric a business person actually understands. "We review
    the riskiest 1% and catch 55% of all fraud" is a sentence that needs no
    explanation, unlike an area under a curve.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype="float64")
    n = len(y)

    n_reviewed = max(1, int(round(n * review_rate)))

    # Sort descending. mergesort is stable, so ties always break the same
    # way and the numbers are reproducible between runs.
    order = np.argsort(-s, kind="mergesort")
    reviewed = order[:n_reviewed]

    caught = float(y[reviewed].sum())
    total_fraud = float(y.sum())

    return {
        "review_rate": review_rate,
        "n_reviewed": n_reviewed,
        "threshold": float(s[order[n_reviewed - 1]]),
        "frauds_caught": int(caught),
        "recall": caught / total_fraud if total_fraud else 0.0,
        "precision": caught / n_reviewed,
    }


def cost_curve(
    y_true,
    scores,
    amounts,
    review_cost: float,
    chargeback_fee: float,
    friction_cost: float,
    recovery_rate: float,
) -> pd.DataFrame:
    """
    Total cost at every possible threshold.

    How it works, because the trick is worth knowing. Sort every transaction
    by score, riskiest first. Then "flag the top k" for every k from 0 to n
    is just a running total, and cumulative sums give the count and value of
    fraud caught at every k in a single pass.

    That makes this exact rather than a sample of a few hundred thresholds,
    and it runs in well under a second on 118,000 rows.

    The four outcomes, priced:
      missed fraud   : the amount is lost, plus a chargeback fee
      caught fraud   : a review is paid for, and the part not recovered is lost
      false alarm    : a review is paid for, plus a small friction cost
      correct pass   : nothing

    Costs are weighted by the real transaction amount, so a missed $2,000
    fraud counts for more than a missed $20 one. A flat per-fraud penalty
    cannot express that.
    """
    y = np.asarray(y_true, dtype="float64")
    s = np.asarray(scores, dtype="float64")
    # A blank amount would poison every sum. There should be none, since
    # TransactionAmt has no missing values, but a guard costs nothing.
    a = np.nan_to_num(np.asarray(amounts, dtype="float64"), nan=0.0)

    order = np.argsort(-s, kind="mergesort")
    y_sorted, s_sorted, a_sorted = y[order], s[order], a[order]

    n = len(y_sorted)
    k = np.arange(n + 1)  # 0 flagged, 1 flagged, ... all flagged

    # Prepend a zero so index k means "the first k rows".
    caught = np.concatenate([[0.0], np.cumsum(y_sorted)])
    caught_value = np.concatenate([[0.0], np.cumsum(y_sorted * a_sorted)])

    total_fraud = caught[-1]
    total_fraud_value = caught_value[-1]

    missed = total_fraud - caught
    missed_value = total_fraud_value - caught_value
    false_alarms = k - caught

    cost_missed = missed_value + missed * chargeback_fee
    cost_caught = caught * review_cost + (1 - recovery_rate) * (
        caught_value + caught * chargeback_fee
    )
    cost_false_alarm = false_alarms * (review_cost + friction_cost)

    total_cost = cost_missed + cost_caught + cost_false_alarm

    # Doing nothing at all: every fraud is missed.
    baseline_cost = total_fraud_value + total_fraud * chargeback_fee

    # The threshold that produces exactly k flags is the score of the k-th
    # row. Flagging nothing needs a threshold above every score.
    thresholds = np.concatenate([[np.inf], s_sorted])

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(caught, k, out=np.zeros_like(caught), where=k > 0)

    return pd.DataFrame(
        {
            "n_flagged": k,
            "review_rate": k / n,
            "threshold": thresholds,
            "frauds_caught": caught,
            "frauds_missed": missed,
            "false_alarms": false_alarms,
            "recall": caught / total_fraud if total_fraud else 0.0,
            "precision": precision,
            "cost_missed": cost_missed,
            "cost_caught": cost_caught,
            "cost_false_alarm": cost_false_alarm,
            "total_cost": total_cost,
            "savings": baseline_cost - total_cost,
        }
    )


def best_operating_point(
    curve: pd.DataFrame, capacity_rate: float | None = None
) -> dict:
    """
    Find the cheapest threshold, optionally limited by review capacity.

    Two answers are useful and they are usually different. The unconstrained
    optimum is what the maths wants. The constrained optimum is what the team
    can actually staff. The gap between them is the price of the constraint,
    which is exactly the number to take to a manager when asking for another
    analyst.
    """
    working = curve
    if capacity_rate is not None:
        working = curve[curve["review_rate"] <= capacity_rate]
        if working.empty:
            working = curve.head(1)

    best = working.loc[working["total_cost"].idxmin()]

    return {
        "threshold": float(best["threshold"]),
        "review_rate": float(best["review_rate"]),
        "n_flagged": int(best["n_flagged"]),
        "recall": float(best["recall"]),
        "precision": float(best["precision"]),
        "total_cost": float(best["total_cost"]),
        "savings": float(best["savings"]),
    }


def evaluate(
    y_true,
    scores,
    amounts,
    review_rates: list[float],
    cost_settings: dict,
    capacity_rate: float,
) -> dict:
    """Run every metric at once and return one flat dictionary."""
    results = ranking_metrics(y_true, scores)

    for rate in review_rates:
        point = review_rate_metrics(y_true, scores, rate)
        label = f"{rate:.3%}".rstrip("0").rstrip("%").replace(".", "p")
        results[f"recall_at_{label}pct"] = point["recall"]
        results[f"precision_at_{label}pct"] = point["precision"]

    curve = cost_curve(y_true, scores, amounts, **cost_settings)
    unconstrained = best_operating_point(curve, capacity_rate=None)
    constrained = best_operating_point(curve, capacity_rate=capacity_rate)

    results["best_savings_unconstrained"] = unconstrained["savings"]
    results["best_review_rate_unconstrained"] = unconstrained["review_rate"]
    results["best_savings_within_capacity"] = constrained["savings"]
    results["best_threshold_within_capacity"] = constrained["threshold"]
    results["best_recall_within_capacity"] = constrained["recall"]

    return results


def downsample_curve(curve: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """
    Thin the cost curve before writing it to a file.

    The full curve has one row per transaction, which is 118,109 rows. That
    is right for finding the exact minimum and unnecessary for a CSV nobody
    will read line by line.
    """
    if len(curve) <= max_rows:
        return curve
    step = len(curve) // max_rows
    return curve.iloc[::step].reset_index(drop=True)
```

---

## 8. Create `src/models/candidates.py`

### 8.1 Why a separate file

Each library trains a little differently. LightGBM takes early stopping through callbacks, XGBoost takes it in the constructor, CatBoost takes it in `fit`. Rather than scatter those differences through the pipeline with `if` statements, each one gets a small adapter here, and the pipeline treats them all the same.

Adding a sixth model later means adding one entry to this file and nothing else.

### 8.2 On class weighting, and why we are not using it

Most guides on imbalanced data reach immediately for `scale_pos_weight`, or SMOTE, or undersampling. We are using none of them, which deserves an explanation.

The question is what those techniques actually improve. Class weighting tells the model that a fraud example counts for, say, 28 times as much as a legitimate one. That shifts the predicted probabilities upward across the board. It does not reliably change the **order** in which transactions are ranked.

And ordering is all we need. PR-AUC measures ordering. Recall at a 1% review rate measures ordering. The cost model picks the threshold separately, from the full curve, so we never rely on a probability crossing 0.5.

Weighting also makes the output probabilities meaningless as probabilities: a score of 0.4 no longer means a 40% chance of fraud. That matters for the dashboard in Step 7, where showing a calibrated risk score is far more useful than showing a number that has been artificially inflated.

So: no weighting, and if the ranking metrics disappoint we revisit. That is D-38. Being able to explain why you did **not** use a well-known technique is often more convincing than using it.

### 8.3 Create the package

```powershell
New-Item -ItemType Directory -Force -Path "src\models" | Out-Null
New-Item -ItemType File -Force -Path "src\models\__init__.py" | Out-Null
```

### 8.4 The file

```python
"""
The model candidates and how each one is trained.

Each library handles early stopping differently: LightGBM through callbacks,
XGBoost through the constructor, CatBoost through fit. Rather than spread
those differences through the pipeline, each gets a small adapter here and
the pipeline treats them all identically.

Adding another model later means adding one entry to build_candidates and
changing nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.config import EARLY_STOPPING_ROUNDS, RANDOM_SEED


@dataclass
class Candidate:
    """One model, its settings, and how to fit it."""

    name: str
    flavor: str                      # which MLflow logger to use
    build: Callable[[int], Any]      # takes max rounds, returns an estimator
    fit: Callable                    # (model, X_tr, y_tr, X_va, y_va) -> (model, best_round)
    params: dict = field(default_factory=dict)
    supports_shap: bool = False


# ---------------------------------------------------------
# Fit adapters
# ---------------------------------------------------------

def _fit_plain(model, X_train, y_train, X_valid, y_valid):
    """For models with no early stopping: the dummy and logistic regression."""
    model.fit(X_train, y_train)
    return model, None


def _fit_lightgbm(model, X_train, y_train, X_valid, y_valid):
    """
    LightGBM takes early stopping as a callback.

    eval_metric "average_precision" is PR-AUC, so training stops when the
    metric we actually care about stops improving, rather than when log loss
    does. Those are not the same point on an imbalanced problem.
    """
    import lightgbm as lgb

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    return model, int(model.best_iteration_)


def _fit_xgboost(model, X_train, y_train, X_valid, y_valid):
    """
    XGBoost 2 and later take early stopping in the constructor, not in fit.

    "aucpr" is XGBoost's name for PR-AUC. best_iteration counts from zero,
    so we add one to get a round count.
    """
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=200)
    return model, int(model.best_iteration) + 1


def _fit_catboost(model, X_train, y_train, X_valid, y_valid):
    """
    CatBoost takes the evaluation set in fit and rolls back to the best
    iteration itself when use_best_model is on.
    """
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=200,
    )
    return model, int(model.get_best_iteration()) + 1


# ---------------------------------------------------------
# The candidates
# ---------------------------------------------------------

def build_candidates(max_rounds: int, include: list[str] | None = None) -> list[Candidate]:
    """
    Build the list of models to train.

    The two baselines are not filler. The dummy establishes the true floor,
    so every later number has something honest to be measured against. The
    logistic regression forces the boosted trees to earn their complexity
    rather than being assumed better because they are fashionable.
    """
    candidates: list[Candidate] = []

    # --- Baseline 1: predict the same thing for everyone ------------------
    # PR-AUC comes out at the fraud rate and ROC-AUC at exactly 0.5. We fit
    # it rather than asserting those numbers, because a floor you measured
    # is worth more than a floor you assumed.
    candidates.append(
        Candidate(
            name="dummy",
            flavor="sklearn",
            build=lambda rounds: DummyClassifier(strategy="prior"),
            fit=_fit_plain,
            params={"strategy": "prior"},
        )
    )

    # --- Baseline 2: classical linear model -------------------------------
    # Wrapped in a Pipeline because logistic regression cannot handle blanks
    # or wildly different scales, unlike the trees. The imputer and scaler
    # are fitted inside the pipeline, so they learn from training data only.
    candidates.append(
        Candidate(
            name="logistic_regression",
            flavor="sklearn",
            build=lambda rounds: Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            fit=_fit_plain,
            params={"max_iter": 1000, "solver": "lbfgs"},
        )
    )

    # --- LightGBM -----------------------------------------------------------
    lightgbm_params = {
        "n_estimators": max_rounds,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    candidates.append(
        Candidate(
            name="lightgbm",
            flavor="lightgbm",
            build=lambda rounds, p=lightgbm_params: __import__(
                "lightgbm"
            ).LGBMClassifier(**{**p, "n_estimators": rounds}),
            fit=_fit_lightgbm,
            params=lightgbm_params,
            supports_shap=True,
        )
    )

    # --- XGBoost -------------------------------------------------------------
    xgboost_params = {
        "n_estimators": max_rounds,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    candidates.append(
        Candidate(
            name="xgboost",
            flavor="xgboost",
            build=lambda rounds, p=xgboost_params: __import__(
                "xgboost"
            ).XGBClassifier(**{**p, "n_estimators": rounds}),
            fit=_fit_xgboost,
            params=xgboost_params,
            supports_shap=True,
        )
    )

    # --- CatBoost -------------------------------------------------------------
    catboost_params = {
        "iterations": max_rounds,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "eval_metric": "PRAUC",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,   # stops CatBoost littering catboost_info/
    }
    candidates.append(
        Candidate(
            name="catboost",
            flavor="catboost",
            build=lambda rounds, p=catboost_params: __import__(
                "catboost"
            ).CatBoostClassifier(**{**p, "iterations": rounds}),
            fit=_fit_catboost,
            params=catboost_params,
            supports_shap=True,
        )
    )

    if include:
        candidates = [c for c in candidates if c.name in include]

    return candidates


def rebuild_for_refit(candidate: Candidate, n_rounds: int):
    """
    Build a fresh copy of a model with a fixed number of rounds and no early
    stopping, for retraining on all labelled data where there is no held-out
    set to stop against.
    """
    model = candidate.build(n_rounds)

    # XGBoost keeps early stopping in the constructor, so it has to be
    # switched off explicitly or fit will demand an eval_set it will not get.
    if candidate.name == "xgboost":
        model.set_params(early_stopping_rounds=None)

    return model


def expanding_window_splits(times: np.ndarray, n_splits: int):
    """
    Cross-validation folds that respect time.

    The time range is cut into equal-sized chunks. Each fold trains on
    everything up to a point and validates on the chunk immediately after,
    so the training window expands with each fold:

        fold 1: train on chunk 1,        validate on chunk 2
        fold 2: train on chunks 1 to 2,  validate on chunk 3
        fold 3: train on chunks 1 to 3,  validate on chunk 4
        ...

    This is the same shape as the real problem repeated several times: learn
    from the past, predict the next period. Ordinary k-fold would train on
    the future and score the past, which is not a thing you can ever do.
    """
    edges = np.quantile(times, np.linspace(0, 1, n_splits + 2))

    for index in range(1, n_splits + 1):
        train_mask = times <= edges[index]
        valid_mask = (times > edges[index]) & (times <= edges[index + 1])
        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue
        yield index, train_mask, valid_mask
```

---

## 9. Create `src/utils/mlflow_utils.py`

```python
"""
MLflow setup and a small compatibility shim.

MLflow 3 renamed the log_model argument from artifact_path to name. The old
one still works but warns, and which is preferred varies across patch
releases. Rather than guess, the helper below inspects the function and uses
whichever it accepts.

That is a useful habit whenever a library is mid-transition: check what is
actually installed instead of assuming.
"""

from __future__ import annotations

import inspect
from typing import Any

import mlflow

from config.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def configure_mlflow(experiment_name: str | None = None) -> str:
    """Point MLflow at the local database and select the experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    name = experiment_name or MLFLOW_EXPERIMENT_NAME
    mlflow.set_experiment(name)
    return MLFLOW_TRACKING_URI


def flavor_module(flavor: str):
    """Return the MLflow logger for a model library."""
    import mlflow.catboost
    import mlflow.lightgbm
    import mlflow.sklearn
    import mlflow.xgboost

    return {
        "sklearn": mlflow.sklearn,
        "lightgbm": mlflow.lightgbm,
        "xgboost": mlflow.xgboost,
        "catboost": mlflow.catboost,
    }[flavor]


def log_model_compatibly(flavor: str, model: Any, name: str, signature=None):
    """
    Log a model, using whichever argument name this MLflow version wants.

    Returns the ModelInfo object, which carries the model_uri needed to
    register the model afterwards.
    """
    module = flavor_module(flavor)
    parameters = inspect.signature(module.log_model).parameters

    if "name" in parameters:
        return module.log_model(model, name=name, signature=signature)
    return module.log_model(model, artifact_path=name, signature=signature)


def log_params_safely(params: dict) -> None:
    """
    Log parameters, keeping each value within MLflow's length limit.

    MLflow rejects very long parameter values. Truncating is better than
    having the whole run fail because one setting was a long list.
    """
    for key, value in params.items():
        text = str(value)
        if len(text) > 480:
            text = text[:477] + "..."
        mlflow.log_param(key, text)


def log_metrics_safely(metrics: dict, prefix: str = "") -> None:
    """Log only the numeric entries, skipping anything MLflow cannot store."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value == value:  # value == value filters NaN
            mlflow.log_metric(f"{prefix}{key}", float(value))
```

---

## 10. Create `src/utils/model_plots.py`

```python
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

    for bar, value in zip(bars, ordered["pr_auc"]):
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

    axis.hist(s[y == 0], bins=bins, alpha=0.6, density=True,
              label="Legitimate", color=LEGIT_COLOUR)
    axis.hist(s[y == 1], bins=bins, alpha=0.6, density=True,
              label="Fraud", color=FRAUD_COLOUR)

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
```

---

## 11. Create `src/pipelines/training.py`

### 11.1 The shape of the stage

Eight phases, in order:

1. Load the processed features and split on the `split` column
2. Train every candidate, score on validation, log each to MLflow
3. Pick the winner by validation PR-AUC
4. Run the uid ablation against the pre-registered rule
5. Cross-validate the winner across expanding time windows
6. Run the threshold and cost analysis
7. Explain the winner with SHAP
8. Retrain on all labelled data, save, register, and score the test set

### 11.2 The file

```python
"""
Model training stage.

Input:  data/processed/train_features.parquet
        data/processed/test_features.parquet
Output: models/final_model.joblib
        models/final_model_metadata.json
        data/processed/kaggle_submission.csv
        reports/model_comparison.csv, threshold_analysis.csv, cost_curve.csv,
                cv_results.csv, feature_importance.csv, training_summary.md
        reports/figures/11 to 15
        reports/explainability/*.png
        Every run recorded in MLflow.

Run with:
    python run.py --step training
    python run.py --step training --quick
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from config.config import (
    COST_CHARGEBACK_FEE,
    COST_FALSE_ALARM_FRICTION,
    COST_REVIEW_PER_CASE,
    CV_N_SPLITS,
    CV_RESULTS_FILE,
    COST_CURVE_FILE,
    EXPLAINABILITY_DIR,
    FEATURE_IMPORTANCE_FILE,
    FEATURES_TEST_FILE,
    FEATURES_TRAIN_FILE,
    FIGURES_DIR,
    FINAL_MODEL_FILE,
    FRAUD_RECOVERY_RATE,
    HEADLINE_REVIEW_RATES,
    ID_COLUMN,
    KAGGLE_SUBMISSION_FILE,
    MAX_BOOSTING_ROUNDS,
    MODEL_ALIAS_CANDIDATE,
    MODEL_COMPARISON_FILE,
    MODEL_METADATA_FILE,
    QUICK_BOOSTING_ROUNDS,
    REFERENCE_DATETIME,
    REGISTERED_MODEL_NAME,
    REVIEW_CAPACITY_RATE,
    SHAP_SAMPLE_SIZE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    THRESHOLD_ANALYSIS_FILE,
    TIME_COLUMN,
    TRAINING_SUMMARY_FILE,
    TRAIN_SPLIT_LABEL,
    UID_ABLATION_TOLERANCE,
    UID_FEATURE_MARKERS,
    VALID_SPLIT_LABEL,
    RANDOM_SEED,
    ensure_directories,
)
from src.models.candidates import (
    build_candidates,
    expanding_window_splits,
    rebuild_for_refit,
)
from src.utils.metrics import (
    best_operating_point,
    cost_curve,
    downsample_curve,
    evaluate,
    ranking_metrics,
    review_rate_metrics,
)
from src.utils.mlflow_utils import (
    configure_mlflow,
    log_metrics_safely,
    log_model_compatibly,
    log_params_safely,
)
from src.utils.model_plots import (
    plot_cost_curve,
    plot_cv_stability,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_score_distribution,
)

COST_SETTINGS = {
    "review_cost": COST_REVIEW_PER_CASE,
    "chargeback_fee": COST_CHARGEBACK_FEE,
    "friction_cost": COST_FALSE_ALARM_FRICTION,
    "recovery_rate": FRAUD_RECOVERY_RATE,
}


def _as_date(seconds: float) -> str:
    reference = pd.Timestamp(REFERENCE_DATETIME)
    return (reference + pd.to_timedelta(int(seconds), unit="s")).date().isoformat()


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    """Everything except the carried-along columns is a feature."""
    excluded = {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN, SPLIT_COLUMN}
    return [column for column in frame.columns if column not in excluded]


def _uid_features(features: list[str]) -> list[str]:
    """Find the uid family by rule, so the list cannot go stale."""
    return [
        name
        for name in features
        if any(marker in name for marker in UID_FEATURE_MARKERS)
    ]


def _score(model, X: pd.DataFrame) -> np.ndarray:
    """Predicted probability of fraud, as a plain array."""
    return model.predict_proba(X)[:, 1]


# =========================================================
# Phase 2: train and compare the candidates
# =========================================================

def _train_candidates(
    candidates, X_train, y_train, X_valid, y_valid, amounts_valid, max_rounds
):
    rows = []
    score_sets = {}
    fitted = {}

    for candidate in candidates:
        print(f"\n  --- {candidate.name} ---")
        started = time.time()

        with mlflow.start_run(run_name=f"candidate_{candidate.name}"):
            mlflow.set_tag("phase", "candidate_comparison")
            mlflow.set_tag("model_family", candidate.name)
            log_params_safely({**candidate.params, "n_features": X_train.shape[1]})

            model = candidate.build(max_rounds)
            model, best_round = candidate.fit(
                model, X_train, y_train, X_valid, y_valid
            )

            scores = _score(model, X_valid)
            metrics = evaluate(
                y_valid,
                scores,
                amounts_valid,
                HEADLINE_REVIEW_RATES,
                COST_SETTINGS,
                REVIEW_CAPACITY_RATE,
            )
            elapsed = time.time() - started

            log_metrics_safely(metrics, prefix="valid_")
            mlflow.log_metric("fit_seconds", elapsed)
            if best_round is not None:
                mlflow.log_metric("best_round", best_round)

            signature = infer_signature(X_valid.head(50), scores[:50])
            log_model_compatibly(candidate.flavor, model, "model", signature=signature)

            print(
                f"    PR-AUC {metrics['pr_auc']:.5f}  "
                f"({metrics['pr_auc_lift']:.1f}x baseline)   "
                f"ROC-AUC {metrics['roc_auc']:.5f}   "
                f"{elapsed / 60:.1f} min"
            )

            rows.append(
                {
                    "model": candidate.name,
                    "best_round": best_round,
                    "fit_minutes": round(elapsed / 60, 2),
                    **metrics,
                }
            )
            score_sets[candidate.name] = scores
            fitted[candidate.name] = (candidate, model)

    return pd.DataFrame(rows), score_sets, fitted


# =========================================================
# Phase 4: the uid ablation
# =========================================================

def _run_uid_ablation(
    candidate, X_train, y_train, X_valid, y_valid, amounts_valid,
    max_rounds, baseline_pr_auc, uid_features,
):
    """
    Train the winner again without the uid features and compare.

    The decision rule was fixed before the result was seen (D-36): if
    removing them costs less than UID_ABLATION_TOLERANCE of PR-AUC, remove
    them, because they are blank on 82% of test rows.
    """
    print(f"\n  --- uid ablation: retraining {candidate.name} without "
          f"{len(uid_features)} uid features ---")

    kept = [column for column in X_train.columns if column not in set(uid_features)]

    with mlflow.start_run(run_name=f"ablation_no_uid_{candidate.name}"):
        mlflow.set_tag("phase", "uid_ablation")
        log_params_safely(
            {**candidate.params, "n_features": len(kept), "uid_removed": True}
        )

        model = candidate.build(max_rounds)
        model, best_round = candidate.fit(
            model, X_train[kept], y_train, X_valid[kept], y_valid
        )

        scores = _score(model, X_valid[kept])
        metrics = evaluate(
            y_valid, scores, amounts_valid, HEADLINE_REVIEW_RATES,
            COST_SETTINGS, REVIEW_CAPACITY_RATE,
        )
        log_metrics_safely(metrics, prefix="valid_")

    difference = baseline_pr_auc - metrics["pr_auc"]
    drop_uid = difference < UID_ABLATION_TOLERANCE

    print(f"    with uid   : PR-AUC {baseline_pr_auc:.5f}")
    print(f"    without uid: PR-AUC {metrics['pr_auc']:.5f}")
    print(f"    difference : {difference:+.5f}  "
          f"(pre-registered tolerance {UID_ABLATION_TOLERANCE})")
    print(f"    DECISION   : {'drop the uid features' if drop_uid else 'keep the uid features'}")

    return {
        "with_uid_pr_auc": baseline_pr_auc,
        "without_uid_pr_auc": metrics["pr_auc"],
        "difference": difference,
        "tolerance": UID_ABLATION_TOLERANCE,
        "drop_uid": bool(drop_uid),
        "uid_features": uid_features,
        "model": model if drop_uid else None,
        "kept_features": kept,
        "best_round": best_round,
        "metrics": metrics,
    }


# =========================================================
# Phase 5: time-aware cross-validation
# =========================================================

def _cross_validate(candidate, X, y, times, n_rounds, n_splits):
    """
    Expanding-window folds, with the round count fixed.

    Early stopping inside each fold would let each fold choose its own best
    stopping point using its own validation data, which makes every fold
    look slightly better than it is. Fixing the count first keeps this an
    honest stability check rather than another round of tuning. That is D-40.
    """
    print(f"\n  Cross-validating {candidate.name} over {n_splits} expanding windows ...")
    rows = []

    for fold, train_mask, valid_mask in expanding_window_splits(times, n_splits):
        model = rebuild_for_refit(candidate, n_rounds)
        model.fit(X[train_mask], y[train_mask])

        scores = _score(model, X[valid_mask])
        metrics = ranking_metrics(y[valid_mask], scores)

        rows.append(
            {
                "fold": fold,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "valid_start": _as_date(times[valid_mask].min()),
                "valid_end": _as_date(times[valid_mask].max()),
                **metrics,
            }
        )
        print(
            f"    fold {fold}: train {int(train_mask.sum()):>7,}  "
            f"valid {int(valid_mask.sum()):>7,}  "
            f"PR-AUC {metrics['pr_auc']:.5f}"
        )

    return pd.DataFrame(rows)


# =========================================================
# Phase 7: SHAP
# =========================================================

def _explain(model, X_valid, feature_names):
    """
    Explain the model with SHAP, on a sample.

    SHAP works out how much each feature pushed one prediction away from the
    average. Averaging those across many rows gives an importance ranking
    that reflects real influence on predictions, unlike the built-in
    importance of a tree model, which just counts how often a feature was
    used for a split.

    A sample is used because explaining all 118,108 validation rows would
    take far longer and change nothing about the answer.
    """
    try:
        import shap
    except ImportError:
        print("    shap not available, skipping")
        return None, None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_size = min(SHAP_SAMPLE_SIZE, len(X_valid))
    sample = X_valid.sample(sample_size, random_state=RANDOM_SEED)

    print(f"    computing SHAP values on {sample_size:,} rows ...")
    explainer = shap.TreeExplainer(model)
    values = explainer(sample)

    # Some libraries return one set of values per class. For a binary
    # problem we want the positive class.
    if values.values.ndim == 3:
        values = values[:, :, 1]

    EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(values, max_display=25, show=False)
    plt.title("What drives the model, top 25 features")
    plt.savefig(EXPLAINABILITY_DIR / "shap_beeswarm.png", bbox_inches="tight", dpi=130)
    plt.close()
    print("    saved shap_beeswarm.png")

    shap.plots.bar(values, max_display=25, show=False)
    plt.title("Average impact on the prediction")
    plt.savefig(EXPLAINABILITY_DIR / "shap_bar.png", bbox_inches="tight", dpi=130)
    plt.close()
    print("    saved shap_bar.png")

    # One worked example: the row the model considered riskiest.
    riskiest = int(np.argmax(np.abs(values.values).sum(axis=1)))
    shap.plots.waterfall(values[riskiest], max_display=18, show=False)
    plt.title("One transaction explained")
    plt.savefig(EXPLAINABILITY_DIR / "shap_waterfall_example.png",
                bbox_inches="tight", dpi=130)
    plt.close()
    print("    saved shap_waterfall_example.png")

    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(values.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return importance, sample_size


# =========================================================
# The stage
# =========================================================

def run_training(quick: bool = False, only_models: list[str] | None = None) -> dict:
    print("=" * 60)
    print("STAGE: MODEL TRAINING")
    print("=" * 60)

    ensure_directories()
    tracking_uri = configure_mlflow()
    print(f"  MLflow tracking: {tracking_uri}")

    max_rounds = QUICK_BOOSTING_ROUNDS if quick else MAX_BOOSTING_ROUNDS
    if quick:
        print(f"  QUICK MODE: boosting capped at {max_rounds} rounds. "
              "Results are for checking the code runs, not for reporting.")

    # --- phase 1: load and split ------------------------------------------
    print(f"\n  Loading {FEATURES_TRAIN_FILE.name} ...")
    data = pd.read_parquet(FEATURES_TRAIN_FILE)
    features = _feature_columns(data)
    print(f"    {len(data):,} rows, {len(features)} features")

    train_mask = (data[SPLIT_COLUMN] == TRAIN_SPLIT_LABEL).to_numpy()
    valid_mask = (data[SPLIT_COLUMN] == VALID_SPLIT_LABEL).to_numpy()

    X_train = data.loc[train_mask, features]
    y_train = data.loc[train_mask, TARGET_COLUMN].to_numpy()
    X_valid = data.loc[valid_mask, features]
    y_valid = data.loc[valid_mask, TARGET_COLUMN].to_numpy()
    amounts_valid = data.loc[valid_mask, "TransactionAmt"].to_numpy()

    valid_times = data.loc[valid_mask, TIME_COLUMN].to_numpy()
    valid_days = (valid_times.max() - valid_times.min()) / 86400

    print(f"    train {len(X_train):,} rows, {int(y_train.sum()):,} frauds")
    print(f"    valid {len(X_valid):,} rows, {int(y_valid.sum()):,} frauds, "
          f"{valid_days:.0f} days")

    uid_features = _uid_features(features)
    print(f"    uid family: {len(uid_features)} features")

    # --- phase 2: candidates ------------------------------------------------
    print("\n  Training candidates ...")
    candidates = build_candidates(max_rounds, include=only_models)
    comparison, score_sets, fitted = _train_candidates(
        candidates, X_train, y_train, X_valid, y_valid, amounts_valid, max_rounds
    )

    comparison = comparison.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    comparison.to_csv(MODEL_COMPARISON_FILE, index=False)
    print(f"\n  Wrote {MODEL_COMPARISON_FILE.name}")

    # --- phase 3: pick a winner -----------------------------------------------
    winner_name = comparison.loc[0, "model"]
    winner_candidate, winner_model = fitted[winner_name]
    winner_pr_auc = float(comparison.loc[0, "pr_auc"])
    winner_round = comparison.loc[0, "best_round"]
    print(f"\n  Winner: {winner_name}, validation PR-AUC {winner_pr_auc:.5f}")

    # --- phase 4: the uid ablation ---------------------------------------------
    ablation = None
    if uid_features and winner_candidate.supports_shap:
        ablation = _run_uid_ablation(
            winner_candidate, X_train, y_train, X_valid, y_valid, amounts_valid,
            max_rounds, winner_pr_auc, uid_features,
        )
        if ablation["drop_uid"]:
            features = ablation["kept_features"]
            winner_model = ablation["model"]
            winner_round = ablation["best_round"]
            X_train = X_train[features]
            X_valid = X_valid[features]
            score_sets[winner_name] = _score(winner_model, X_valid)
            print(f"    feature count now {len(features)}")

    winner_scores = score_sets[winner_name]

    # --- phase 5: cross-validation ------------------------------------------------
    n_rounds = int(winner_round) if winner_round else max_rounds
    all_X = data[features]
    all_y = data[TARGET_COLUMN].to_numpy()
    all_times = data[TIME_COLUMN].to_numpy()

    cv_results = _cross_validate(
        winner_candidate, all_X, all_y, all_times, n_rounds, CV_N_SPLITS
    )
    cv_results.to_csv(CV_RESULTS_FILE, index=False)
    print(f"    PR-AUC across folds: mean {cv_results['pr_auc'].mean():.5f}, "
          f"spread {cv_results['pr_auc'].std():.5f}")

    # --- phase 6: thresholds and cost ------------------------------------------------
    print("\n  Threshold and cost analysis ...")
    curve = cost_curve(y_valid, winner_scores, amounts_valid, **COST_SETTINGS)
    unconstrained = best_operating_point(curve, capacity_rate=None)
    constrained = best_operating_point(curve, capacity_rate=REVIEW_CAPACITY_RATE)

    downsample_curve(curve).to_csv(COST_CURVE_FILE, index=False)

    threshold_rows = []
    for rate in HEADLINE_REVIEW_RATES:
        point = review_rate_metrics(y_valid, winner_scores, rate)
        at_rate = curve.iloc[point["n_reviewed"]]
        threshold_rows.append(
            {
                **point,
                "total_cost": float(at_rate["total_cost"]),
                "savings": float(at_rate["savings"]),
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(THRESHOLD_ANALYSIS_FILE, index=False)

    baseline_cost = float(curve.loc[0, "total_cost"])
    annual_factor = 365.0 / max(valid_days, 1.0)

    print(f"    doing nothing costs        : ${baseline_cost:,.0f} over {valid_days:.0f} days")
    print(f"    cheapest overall           : {unconstrained['review_rate']:.2%} reviewed, "
          f"saves ${unconstrained['savings']:,.0f}")
    print(f"    cheapest within {REVIEW_CAPACITY_RATE:.0%} capacity: "
          f"{constrained['review_rate']:.2%} reviewed, saves ${constrained['savings']:,.0f}")
    print(f"    annualised saving          : ${constrained['savings'] * annual_factor:,.0f}")

    # --- phase 7: SHAP -------------------------------------------------------------------
    print("\n  Explaining the model ...")
    importance, shap_rows = (None, None)
    if winner_candidate.supports_shap:
        importance, shap_rows = _explain(winner_model, X_valid, features)
        if importance is not None:
            importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
            print(f"    Wrote {FEATURE_IMPORTANCE_FILE.name}")
            print("    top 10 features:")
            for _, row in importance.head(10).iterrows():
                print(f"      {row['feature']:<45} {row['mean_abs_shap']:.5f}")

    # --- charts ------------------------------------------------------------------------
    print("\n  Generating charts ...")
    plot_model_comparison(comparison, FIGURES_DIR)
    plot_precision_recall_curves(y_valid, score_sets, FIGURES_DIR)
    plot_cost_curve(curve, unconstrained, constrained, REVIEW_CAPACITY_RATE, FIGURES_DIR)
    plot_score_distribution(y_valid, winner_scores, FIGURES_DIR)
    plot_cv_stability(cv_results, FIGURES_DIR)

    # --- phase 8: final model, registry, submission ---------------------------------------
    # Retrain on every labelled row. Validation chose the settings; the model
    # that ships should still see all the data. The round count is scaled by
    # how much more data it now sees, which is the standard adjustment. D-41.
    scale = len(data) / len(X_train)
    final_rounds = max(1, int(round(n_rounds * scale)))
    print(f"\n  Retraining {winner_name} on all {len(data):,} labelled rows "
          f"({n_rounds} rounds scaled by {scale:.2f} to {final_rounds}) ...")

    with mlflow.start_run(run_name=f"final_{winner_name}") as final_run:
        mlflow.set_tag("phase", "final")
        mlflow.set_tag("model_family", winner_name)
        log_params_safely(
            {
                **winner_candidate.params,
                "n_estimators": final_rounds,
                "n_features": len(features),
                "uid_features_dropped": bool(ablation and ablation["drop_uid"]),
                "trained_on_rows": len(data),
            }
        )

        final_model = rebuild_for_refit(winner_candidate, final_rounds)
        final_model.fit(all_X, all_y)

        # These are the validation numbers from the model selection step, not
        # a score for the final model. The final model has no clean holdout
        # left, which is exactly why we validated before retraining.
        log_metrics_safely(
            {
                "selection_pr_auc": winner_pr_auc,
                "cv_pr_auc_mean": float(cv_results["pr_auc"].mean()),
                "cv_pr_auc_std": float(cv_results["pr_auc"].std()),
                "chosen_threshold": constrained["threshold"],
                "savings_within_capacity": constrained["savings"],
                "annualised_savings": constrained["savings"] * annual_factor,
            }
        )

        signature = infer_signature(all_X.head(50), _score(final_model, all_X.head(50)))
        model_info = log_model_compatibly(
            winner_candidate.flavor, final_model, "model", signature=signature
        )

        for path in (
            MODEL_COMPARISON_FILE, THRESHOLD_ANALYSIS_FILE,
            CV_RESULTS_FILE, COST_CURVE_FILE,
        ):
            if path.exists():
                mlflow.log_artifact(str(path))

        final_run_id = final_run.info.run_id

    joblib.dump(final_model, FINAL_MODEL_FILE)
    print(f"  Saved {FINAL_MODEL_FILE.name} "
          f"({FINAL_MODEL_FILE.stat().st_size / 1024 ** 2:.1f} MB)")

    # Register it and point the candidate alias at this version.
    registered_version = None
    try:
        registered = mlflow.register_model(model_info.model_uri, REGISTERED_MODEL_NAME)
        registered_version = registered.version
        mlflow.MlflowClient().set_registered_model_alias(
            REGISTERED_MODEL_NAME, MODEL_ALIAS_CANDIDATE, registered_version
        )
        print(f"  Registered as {REGISTERED_MODEL_NAME} version "
              f"{registered_version}, alias '{MODEL_ALIAS_CANDIDATE}'")
    except Exception as error:  # noqa: BLE001
        print(f"  Registry step failed: {error}")
        print("  The model file and the MLflow run are still saved.")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": winner_name,
        "mlflow_run_id": final_run_id,
        "registered_version": registered_version,
        "n_features": len(features),
        "feature_names": features,
        "uid_features_dropped": bool(ablation and ablation["drop_uid"]),
        "n_estimators": final_rounds,
        "selection_pr_auc": winner_pr_auc,
        "cv_pr_auc_mean": float(cv_results["pr_auc"].mean()),
        "chosen_threshold": constrained["threshold"],
        "chosen_review_rate": constrained["review_rate"],
        "cost_assumptions": COST_SETTINGS,
        "review_capacity_rate": REVIEW_CAPACITY_RATE,
    }
    MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Wrote {MODEL_METADATA_FILE.name}")

    # --- Kaggle submission ---------------------------------------------------------
    print(f"\n  Scoring the test set ...")
    test = pd.read_parquet(FEATURES_TEST_FILE)
    test_scores = _score(final_model, test[features])
    pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN].to_numpy(), TARGET_COLUMN: test_scores}
    ).to_csv(KAGGLE_SUBMISSION_FILE, index=False)
    print(f"    Wrote {KAGGLE_SUBMISSION_FILE.name} ({len(test):,} rows)")

    results = {
        "comparison": comparison,
        "winner": winner_name,
        "winner_pr_auc": winner_pr_auc,
        "ablation": ablation,
        "cv_results": cv_results,
        "unconstrained": unconstrained,
        "constrained": constrained,
        "threshold_table": threshold_table,
        "baseline_cost": baseline_cost,
        "annual_factor": annual_factor,
        "valid_days": valid_days,
        "importance": importance,
        "n_features": len(features),
        "final_rounds": final_rounds,
        "registered_version": registered_version,
        "final_run_id": final_run_id,
    }
    _write_summary(results)

    print("\n" + "=" * 60)
    print("TRAINING HEADLINES")
    print("=" * 60)
    print(f"  Winner                : {winner_name}")
    print(f"  Validation PR-AUC     : {winner_pr_auc:.5f} "
          f"({winner_pr_auc / 0.0349:.1f}x baseline)")
    print(f"  CV PR-AUC             : {cv_results['pr_auc'].mean():.5f} "
          f"+/- {cv_results['pr_auc'].std():.5f}")
    print(f"  Features used         : {len(features)}")
    print(f"  Chosen threshold      : {constrained['threshold']:.4f} "
          f"at {constrained['review_rate']:.2%} review rate")
    print(f"  Recall at that point  : {constrained['recall']:.1%}")
    print(f"  Annualised saving     : ${constrained['savings'] * annual_factor:,.0f}")
    print(f"\n  Full report: {TRAINING_SUMMARY_FILE}")

    return results


def _write_summary(results: dict) -> None:
    """Write the human-readable training summary."""
    lines: list[str] = []
    add = lines.append

    constrained = results["constrained"]
    unconstrained = results["unconstrained"]
    cv = results["cv_results"]

    add("# Model Training Summary")
    add("")
    add("Generated automatically by `src/pipelines/training.py`. "
        "Do not edit by hand, it is overwritten on every run.")
    add("")

    add("## 1. Candidate comparison")
    add("")
    display = results["comparison"][
        ["model", "pr_auc", "pr_auc_lift", "roc_auc", "best_round", "fit_minutes"]
    ].round(5)
    add(display.to_markdown(index=False))
    add("")
    add(f"Winner: **{results['winner']}**, validation PR-AUC "
        f"**{results['winner_pr_auc']:.5f}**.")
    add("")

    if results["ablation"]:
        ablation = results["ablation"]
        add("## 2. The uid ablation")
        add("")
        add(f"Six uid features are blank on about 82% of test rows, so the "
            f"winner was retrained without them. The decision rule was fixed "
            f"in advance: drop them if the cost is under "
            f"{ablation['tolerance']} PR-AUC.")
        add("")
        add("| Model | Validation PR-AUC |")
        add("|-------|-------------------|")
        add(f"| with uid features | {ablation['with_uid_pr_auc']:.5f} |")
        add(f"| without uid features | {ablation['without_uid_pr_auc']:.5f} |")
        add(f"| difference | {ablation['difference']:+.5f} |")
        add("")
        add(f"**Decision: {'dropped' if ablation['drop_uid'] else 'kept'}.** "
            f"Final feature count {results['n_features']}.")
        add("")

    add("## 3. Stability across time")
    add("")
    add(cv[["fold", "train_rows", "valid_rows", "valid_start",
            "valid_end", "pr_auc", "roc_auc"]].round(5).to_markdown(index=False))
    add("")
    add(f"Mean PR-AUC **{cv['pr_auc'].mean():.5f}**, "
        f"spread **{cv['pr_auc'].std():.5f}**. Each fold trains on more "
        "history than the last and is scored on the period straight after, "
        "which is the same shape as the real problem.")
    add("")

    add("## 4. What it is worth")
    add("")
    add("Costs use the assumptions in `config/config.py`. They are stated "
        "assumptions, not figures supplied by a business. See step4.md "
        "section 3.")
    add("")
    add("| Assumption | Value |")
    add("|------------|-------|")
    add(f"| Analyst review | ${COST_REVIEW_PER_CASE:.2f} per case |")
    add(f"| Chargeback fee | ${COST_CHARGEBACK_FEE:.2f} per missed fraud |")
    add(f"| False alarm friction | ${COST_FALSE_ALARM_FRICTION:.2f} |")
    add(f"| Fraud recovered when caught | {FRAUD_RECOVERY_RATE:.0%} |")
    add(f"| Review capacity | {REVIEW_CAPACITY_RATE:.0%} of transactions |")
    add("")
    add(f"Over the {results['valid_days']:.0f} day validation period, doing "
        f"nothing costs **${results['baseline_cost']:,.0f}** in fraud losses "
        "and chargeback fees.")
    add("")
    add("| Operating point | Review rate | Recall | Savings |")
    add("|-----------------|-------------|--------|---------|")
    add(f"| Cheapest overall | {unconstrained['review_rate']:.2%} | "
        f"{unconstrained['recall']:.1%} | ${unconstrained['savings']:,.0f} |")
    add(f"| Cheapest within capacity | {constrained['review_rate']:.2%} | "
        f"{constrained['recall']:.1%} | ${constrained['savings']:,.0f} |")
    add("")
    add(f"**Annualised, at the within-capacity operating point: "
        f"${constrained['savings'] * results['annual_factor']:,.0f} a year.**")
    add("")
    add(f"The chosen threshold is **{constrained['threshold']:.4f}**.")
    add("")
    add("Recall and cost at each headline review rate:")
    add("")
    add(results["threshold_table"][
        ["review_rate", "n_reviewed", "threshold", "recall", "precision", "savings"]
    ].round(5).to_markdown(index=False))
    add("")

    if results["importance"] is not None:
        add("## 5. What drives the model")
        add("")
        add(results["importance"].head(20).round(5).to_markdown(index=False))
        add("")
        add("Charts in `reports/explainability/`.")
        add("")

    add("## 6. Carried into Step 5")
    add("")
    add(f"1. Registered model `{REGISTERED_MODEL_NAME}` version "
        f"{results['registered_version']}, alias `{MODEL_ALIAS_CANDIDATE}`.")
    add(f"2. MLflow run id `{results['final_run_id']}`.")
    add(f"3. Operating threshold {constrained['threshold']:.4f}, chosen by "
        "cost within review capacity, not left at 0.5.")
    add("4. Watch the uid family in drift monitoring, whether or not it was "
        "dropped. It was the clearest train-to-test shift in the data.")
    add("5. `models/final_model_metadata.json` holds the exact feature list "
        "the service must supply.")
    add("")

    TRAINING_SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {TRAINING_SUMMARY_FILE.name}")
```

---

## 12. Update `run.py`

**Add this function** below `run_features_stage`:

```python
def run_training_stage(args: argparse.Namespace) -> dict:
    from src.pipelines.training import run_training

    return run_training(quick=args.quick, only_models=args.models)
```

**Add two arguments** next to the existing ones:

```python
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Cap boosting rounds so the training stage finishes fast. "
             "For checking the code runs, not for real results.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Train only these models, for example: --models lightgbm xgboost",
    )
```

**Update the choices:**

```python
        choices=["ingestion", "eda", "features", "training", "all"],
```

**Update the dispatch:**

```python
    elif args.step == "training":
        run_training_stage(args)
    elif args.step == "all":
        run_ingestion_stage(args)
        run_eda_stage(args)
        run_features_stage(args)
        run_training_stage(args)
```

---

## 13. Run it

### 13.1 Branch and import check

```powershell
git switch main
git pull
git switch -c step-04-training

python -c "from src.pipelines.training import run_training; from src.models.candidates import build_candidates; print('imports OK')"
```

### 13.2 Quick pass first

Always. Fifteen minutes of debugging beats discovering a typo forty minutes into a real run.

```powershell
python run.py --step training --quick --models dummy logistic_regression lightgbm
```

This caps boosting at 150 rounds. Expect 5 to 10 minutes. The numbers are not meaningful, only the fact that every phase completes is.

**Watch for:** all three candidates train, the ablation runs, four CV folds complete, the cost analysis prints dollar figures, SHAP saves three charts, the final model is registered, and the submission file is written.

### 13.3 The real run

```powershell
python run.py --step training
```

**Expect 45 to 75 minutes.** Rough split: logistic regression 3 to 8 minutes, LightGBM 4 to 8, XGBoost 5 to 10, CatBoost 10 to 20, the ablation another 4 to 8, four CV folds 10 to 20, SHAP 2 to 5.

If CatBoost is taking too long, cancel with `Ctrl+C` and re-run without it:

```powershell
python run.py --step training --models dummy logistic_regression lightgbm xgboost
```

### 13.4 What the output should look like

```
============================================================
STAGE: MODEL TRAINING
============================================================
  MLflow tracking: sqlite:///C:/Users/.../mlflow.db

  Loading train_features.parquet ...
    590,540 rows, 284 features
    train 472,432 rows, 16,599 frauds
    valid 118,108 rows, 4,064 frauds, 41 days
    uid family: 6 features

  Training candidates ...

  --- dummy ---
    PR-AUC 0.03441  (1.0x baseline)   ROC-AUC 0.50000   0.0 min
  --- logistic_regression ---
    PR-AUC 0.xxxxx  ...
  --- lightgbm ---
    PR-AUC 0.xxxxx  ...
  ...

  Winner: xxx, validation PR-AUC 0.xxxxx

  --- uid ablation: retraining xxx without 6 uid features ---
    with uid   : PR-AUC 0.xxxxx
    without uid: PR-AUC 0.xxxxx
    difference : +0.000xx  (pre-registered tolerance 0.005)
    DECISION   : ...

  Cross-validating xxx over 4 expanding windows ...
    fold 1: ...

  Threshold and cost analysis ...
    doing nothing costs        : $xxx,xxx over 41 days
    cheapest overall           : x.xx% reviewed, saves $xxx,xxx
    cheapest within 2% capacity: x.xx% reviewed, saves $xxx,xxx
    annualised saving          : $x,xxx,xxx
```

The dummy is the one line whose value I can predict: PR-AUC will come out at the validation fraud rate of 0.0344 and ROC-AUC at exactly 0.5. If it does not, something is wrong with the metric code, and it is far better to find that out on a model whose answer you already know.

### 13.5 Confirm the outputs

```powershell
Get-ChildItem reports -File | Select-Object Name
Get-ChildItem reports\figures | Select-Object Name
Get-ChildItem reports\explainability | Select-Object Name
Get-ChildItem models | Select-Object Name, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,1)}}
```

---

## 14. The MLflow interface

Now look at what was recorded.

```powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000
```

Run this from the project root, in a **second terminal** with the environment activated, since it holds the terminal while running. Open `http://localhost:5000`.

### 14.1 What to do in there

1. **Click the `ieee-cis-fraud-detection` experiment.** Every run is listed.
2. **Add columns.** The column picker lets you show `valid_pr_auc`, `valid_roc_auc`, `fit_seconds`. Now you have a sortable comparison table you never had to build.
3. **Select two runs and click Compare.** Side by side parameters and metrics, with the differences highlighted. This is the view that answers "what exactly was different about that run".
4. **Filter by tag.** Type `tags.phase = 'candidate_comparison'` into the search box to hide the ablation and final runs.
5. **Open the final run and click Artifacts.** The model is there, along with the CSVs, browsable in the page.
6. **Click Models in the top navigation.** The registered model, its version, and the `candidate` alias. Step 5 moves that alias to promote a model, and Step 6 loads whatever it points at.

Stop the server with `Ctrl+C`.

### 14.2 Why this matters more than it looks

When you demonstrate this project, the model is the least surprising part. Everyone building a portfolio has a model.

What separates a project that reads as engineered is being able to open this page, sort by PR-AUC, click two runs, and say "here is exactly what I tried, here is what each one scored, here is the one I chose and why". That is the difference between showing a result and showing a process.

---

## 15. The Kaggle late submission

This answers Q-05. It is free, the competition is closed for prizes but still scores late submissions, and it gives you one number nobody has to take your word for.

### 15.1 Submit

```powershell
kaggle competitions submit `
  -c ieee-fraud-detection `
  -f data/processed/kaggle_submission.csv `
  -m "LightGBM/XGBoost/CatBoost comparison, time-based split, 284 engineered features"
```

Adjust the message to name the model that actually won.

### 15.2 Check the score

```powershell
kaggle competitions submissions -c ieee-fraud-detection
```

You get a public and a private score, both **ROC-AUC**, because that was the competition metric. Note the mismatch: we optimised for PR-AUC because it suits the business problem, and Kaggle scores ROC-AUC. Both numbers are logged, so you can report the Kaggle figure honestly without pretending it was the target.

The private score is the meaningful one; it is computed on a held-out portion of the test set that was never on the public leaderboard.

### 15.3 What a good number looks like

For context on the leaderboard: the competition winner scored around 0.945 private ROC-AUC, and roughly the top half of several thousand teams landed above 0.92.

A well-built single model with sound feature engineering and no leakage typically lands somewhere in the 0.91 to 0.94 range. The very top scores came from heavy ensembling and aggressive customer-identity reconstruction, some of which does not transfer to a production setting.

Whatever you score, the honest framing for the README is the useful one: this is a single model, trained with production constraints, no ensembling, no test-set leakage, and here is what it scored against several thousand teams.

---

## 16. Update the README results

Replace the Results section with the real numbers from your run, and add the business figure, which is the part a hiring manager will actually remember.

```markdown
## Results

Validation is the last 20% of the training period by time, from 2018-04-20 to
2018-05-31: 118,108 transactions containing 4,064 frauds. The model never sees
any of it during training.

| Metric | Random baseline | This model |
|--------|-----------------|------------|
| PR-AUC | 0.034 | **TBD** |
| ROC-AUC | 0.500 | **TBD** |
| Recall at 1% review rate | 1.0% | **TBD** |
| Kaggle private leaderboard (ROC-AUC) | 0.500 | **TBD** |

Stability across four expanding time windows: PR-AUC TBD, spread TBD.

### What it is worth

Under a cost model with five stated assumptions, documented in
`docs/steps/step4.md`, running the model at a 2% manual review capacity is
worth roughly **$TBD a year** in prevented fraud, net of review costs.

The assumptions: $4.00 per analyst review, $25.00 chargeback fee per missed
fraud, $1.00 friction per false alarm, 90% of flagged fraud actually prevented,
and a team able to review 2% of transactions. All five live in
`config/config.py`. Change one, re-run, and the figure updates.

These are assumptions rather than figures from a business, and the savings
estimate should be read as an order of magnitude rather than a forecast.
```

Fill in every TBD from `reports/training_summary.md`.

---

## 17. Commit, merge, tag

```powershell
git add config/config.py src/features/engineer.py
git commit -m "feat: add cost model settings and fix the pandas concat warning"

git add src/utils/metrics.py src/utils/mlflow_utils.py src/utils/model_plots.py
git commit -m "feat: add metrics, cost model, mlflow helpers, and training charts"

git add src/models/
git commit -m "feat: add model candidates with per-library fit adapters"

git add src/pipelines/training.py run.py
git commit -m "feat: add training stage with ablation, time-aware cv, and cost-based threshold"

git add reports/ models/final_model_metadata.json
git commit -m "docs: add training reports, figures, and shap explanations"

git add README.md docs/
git commit -m "docs: add step 4 guide and update readme with results"

# DVC: track the new submission file
dvc add data/processed/kaggle_submission.csv
git add data/processed/kaggle_submission.csv.dvc data/processed/.gitignore
git commit -m "chore: track kaggle submission with dvc"
dvc push

git push -u origin step-04-training

gh pr create --base main --head step-04-training `
  --title "Step 4: model training and experiment tracking" `
  --body "Cost model with stated assumptions, MLflow tracking and registry, five candidates compared, pre-registered uid ablation, expanding-window cross-validation, cost-based threshold selection, SHAP explanations, and a Kaggle late submission."

gh pr merge --squash --delete-branch

git switch main
git pull
git tag -a v0.4.0-step4 -m "Step 4 complete: model training and experiment tracking"
git push origin v0.4.0-step4
```

`mlflow.db`, `mlruns/`, and `models/final_model.joblib` stay out of Git. They are already in `.gitignore` from Step 1. The metadata JSON is small and text, so that one is committed: Step 6 needs the feature list.

---

## 18. Where to deploy in Step 6

This answers Q-04. Your requirements were free, not complicated, standard for this level, and good for showing hiring managers.

**Recommendation: Hugging Face Spaces, using the Docker SDK.** That is D-44.

### 18.1 Why

**It is genuinely free.** No credit card, no trial that expires, no surprise bill. Some competitors advertise a free tier that turns out to be trial credits.

**It runs your actual Docker image.** This matters more than it sounds. Step 6 builds a Dockerfile. On a platform that only accepts a Python file, that Dockerfile becomes decoration: you built it, and then deployed something else. On Spaces the container you tested locally is the container that runs, which makes Step 6 real work rather than an exercise.

**Hiring managers can click the link.** A public URL that loads without a login, without a signup, without a fifty second cold start. Render's free tier spins down after fifteen minutes of inactivity, and a link that shows a blank page for a minute is a link nobody waits for.

**It is the recognised venue.** Spaces is where ML people put demos. A reviewer seeing a Spaces link knows immediately what they are looking at.

**FastAPI gets you Swagger for free.** FastAPI generates interactive API documentation at `/docs`. A hiring manager can send a test transaction through a web form and watch a fraud score come back, without installing anything. That is a genuinely persuasive thirty seconds.

### 18.2 The shape of it

Two Spaces:

1. **The API.** Docker, FastAPI, loading `feature_engineer.joblib` and `final_model.joblib`. Endpoints `/health`, `/predict`, and `/docs`.
2. **The dashboard.** Streamlit, calling the API for live scoring and reading precomputed artifacts for everything else.

Two Spaces rather than one, because a real service boundary is worth showing. It also means the dashboard can go down without taking the API with it.

### 18.3 The one thing to plan for

Your artifacts are large: the transformer is 28 MB and the model will add more. Spaces repositories handle this through Git LFS, which is supported but needs setting up before the first push.

The alternative is putting the artifacts on the Hugging Face Model Hub and having the container download them at startup. That is closer to how real deployments work, since the model is versioned separately from the code. Step 6 will use the Model Hub route and explain why.

**Backup option:** Render's free web service tier also runs Docker. It is a fine second choice if Spaces does not suit, with the cold start being the real cost. Step 6 will note what changes.

---

## 19. Who the dashboard is for

This answers Q-12. You said portfolio and hiring managers, which is a clearer brief than it might seem, and it rules several things out. That is D-45.

**The reader.** Somebody technical but not necessarily an ML specialist, looking at your project for maybe two minutes, probably alongside several others, possibly on a phone.

**What follows from that:**

**It has to load fast.** Under three seconds. A dashboard that shows a spinner has already lost. Every chart reads from a small precomputed file, never from the 590,540 row table. This is why D-33 in Step 3 specified precomputed artifacts.

**It has to explain itself.** No jargon without a one-line gloss. "PR-AUC 0.72" means nothing on its own. "Catches 58% of fraud while reviewing 1% of transactions" needs no explanation at all.

**It should lead with the money.** The cost model output is the most memorable thing you have. The annual savings figure, with its assumptions visible, is what a reader repeats to a colleague.

**It should be interactive somewhere.** A single-transaction scorer with the SHAP breakdown is the thing people actually try. Reading a chart is passive; clicking a button and getting an answer is not.

**It should show the engineering.** Model version, drift status, when it was last trained, the CI badge. That is what separates you from the many portfolios that stop at a notebook with a confusion matrix.

**What to leave out.** The full EDA. Every chart you have made. Anything needing a paragraph of setup. The dashboard is not the report; the report is in `reports/`, and the dashboard links to it.

Planned layout, five sections top to bottom: the headline result and business impact, how the model performs with two charts, a live transaction scorer, drift monitoring, and how it was built. Step 7 will build exactly that.

---

## 20. Reading your results

### 20.1 Sanity checks

| Check | Expected | If it is off |
|-------|----------|--------------|
| Dummy PR-AUC | Exactly 0.0344, the validation fraud rate | The metric code is wrong. Stop and tell me. |
| Dummy ROC-AUC | Exactly 0.5 | Same. |
| Logistic regression PR-AUC | Well above the dummy, well below the boosters | If it beats the boosters, something is badly wrong. |
| Best PR-AUC | Comfortably above 0.5 | Below 0.3 suggests a problem in the features or the split. |
| CV spread | Small relative to the mean | A large spread means performance depends heavily on the period, which matters for retraining frequency. |
| Chosen review rate | Somewhere between 0.5% and 2% | At exactly 2% the capacity constraint is binding, which is itself worth reporting. |

### 20.2 The three things worth studying

**The uid ablation result.** Whichever way it went, it is the most interesting thing in the run. If dropping six features cost almost nothing, that tells you those features were doing less work than their prominence suggested. If it cost a lot, you have a real tension between validation performance and test robustness, and that is a genuinely hard trade-off worth talking about.

**The SHAP ranking.** Check where `has_identity` lands. Back in Step 3 I predicted it would rank low, because the apparent 3.75x signal was mostly a `ProductCD` effect. This is the test of that analysis. If it ranks low, the reasoning held. If it ranks high, the reasoning was wrong and I would want to see the chart.

Also check whether any of the V columns rescued in Step 3 appear near the top. V111, with its 46% fraud rate on 1,370 rows, is the one to look for. If it ranks high, the rescue rule saved a genuinely important feature from deletion, which is a good story.

**The gap between the two operating points.** If the unconstrained optimum wants a 6% review rate and capacity allows 2%, the difference in savings is the price of being short-staffed. That number is a business case for another analyst, expressed in the only terms that matter.

---

## 21. Verification checklist

**Setup**
- [ ] Branch `step-04-training` created
- [ ] `MLFLOW_TRACKING_URI` uses forward slashes
- [ ] Step 4 config block added, the check prints `25.0` and the markers
- [ ] `src/features/engineer.py` concat fix applied
- [ ] `src/models/` package created
- [ ] All four new files created
- [ ] `run.py` updated with `training`, `--quick`, `--models`
- [ ] Import check prints `imports OK`

**Quick pass**
- [ ] `--quick` run completed end to end with no errors

**Real run**
- [ ] Full run completed
- [ ] Dummy scored PR-AUC 0.0344 and ROC-AUC 0.5000 exactly
- [ ] Every candidate trained
- [ ] The ablation ran and printed a decision
- [ ] Four CV folds completed
- [ ] Cost analysis printed dollar figures
- [ ] SHAP saved three charts
- [ ] `models/final_model.joblib` and the metadata JSON exist
- [ ] `data/processed/kaggle_submission.csv` has 506,691 rows

**MLflow**
- [ ] `mlflow ui` opens and shows the runs
- [ ] Comparing two runs works
- [ ] The registered model appears under Models with the `candidate` alias

**Kaggle**
- [ ] Submission uploaded
- [ ] Public and private scores recorded

**Git**
- [ ] No `.joblib`, `mlflow.db`, or `mlruns/` in `git status`
- [ ] README results filled in with real numbers
- [ ] Branch merged, tag `v0.4.0-step4` pushed
- [ ] Submission tracked with DVC and pushed

---

## 22. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sqlalchemy.exc.ArgumentError` on the tracking URI | Backslashes in the database URL | The `.as_posix()` fix in Section 6.1 |
| `mlflow.exceptions.MlflowException: ... registry` | Registry needs a database backend | Confirm the URI starts `sqlite:///`, not `file:` |
| `TypeError: log_model() got an unexpected keyword argument 'name'` | Older MLflow than the shim expects | The shim handles it. If it still fails, send me `python -c "import mlflow; print(mlflow.__version__)"` |
| XGBoost: `Must have at least 1 validation dataset for early stopping` | Refitting with early stopping still on | `rebuild_for_refit` clears it. Check you used that helper. |
| CatBoost writes a `catboost_info` folder | Default logging | `allow_writing_files=False` is set. It is also gitignored. |
| LightGBM `best_iteration_` is None | Early stopping never triggered | The model used every round. Fine, but consider raising `MAX_BOOSTING_ROUNDS`. |
| `MemoryError` during logistic regression | Imputer and scaler make dense float64 copies | Run without it: `--models dummy lightgbm xgboost catboost` |
| SHAP is very slow | Sample too large | Lower `SHAP_SAMPLE_SIZE` to 2000 |
| SHAP `AssertionError` on shape | Explanation returned per-class values | The code handles `ndim == 3`. Send me the traceback if it still fails. |
| `mlflow ui` shows no runs | Wrong backend store | Run it from the project root, with the URI exactly as in Section 14 |
| Kaggle submit returns 404 | Wrong competition slug | It is `ieee-fraud-detection`, with no "cis" |
| Kaggle submit rejects the file | Wrong columns | Must be exactly `TransactionID,isFraud` |
| Training takes over 2 hours | CatBoost on CPU | Drop it with `--models` and tell me the timing |

---

## 23. What to send me before Step 5

1. **The full terminal output** of `python run.py --step training`
2. **`reports/training_summary.md`** contents
3. **`reports/model_comparison.csv`** as an attachment
4. **`reports/feature_importance.csv`** as an attachment. This tells us whether `has_identity` ranked low as predicted and whether the rescued V columns earned their place.
5. **`reports/cv_results.csv`** as an attachment
6. **The uid ablation decision** and the two PR-AUC numbers
7. **Your Kaggle public and private scores**
8. **A screenshot of the MLflow runs table**, if easy. Useful for the Step 7 dashboard and the portfolio write-up.
9. **Any checklist item that did not tick**, with the error text

---

## 24. What Step 5 covers

- A pytest suite: unit tests for the metrics and the cost model, tests that the transformer round-trips through joblib unchanged, and a test that catches leakage if anyone reintroduces it
- GitHub Actions running the tests, ruff, and black on every pull request, with a badge for the README
- `pre-commit` so problems are caught before they reach a commit
- The MLflow model registry used properly: promoting from `candidate` to `production` by moving an alias, and what has to be true before a promotion is allowed
- Drift monitoring built on the real shift already found: identity coverage moving from 24.4% to 28.0%, and the uid family collapsing from 11% to 82% missing. Both are genuine, both are measurable, and neither had to be manufactured.
- Population Stability Index and Kolmogorov-Smirnov tests, explained from scratch, with thresholds that mean something
- Scoring the test set month by month to show performance decaying, or not, as the data moves further from training
- A retraining trigger: what condition should fire it, and what should happen when it does
- Monitoring outputs written in the shape the Step 7 dashboard needs, per D-33

---

*End of Step 4. `PROJECT_STATE.md` follows as a separate document.*
