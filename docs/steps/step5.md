# Step 5: The MLOps Layer
### Tests, continuous integration, drift monitoring, and promotion gates

**Project:** IEEE-CIS Fraud Detection
**Repository:** https://github.com/Dee-ui/ieee-cis-fraud-detection
**Local path:** `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`
**Platform:** Windows, VS Code, PowerShell, Python 3.11.9
**Estimated time:** 4 to 5 hours, of which about 10 minutes is the machine running
**Step 5 of 7**

---

## 0. How to use this document

Sections 1 and 2 read your Step 4 results. Two of my predictions were tested by your data: one held up exactly, and one was measured with the wrong tool, which I correct in Section 2.7. Section 2.8 contains a finding I derived from your numbers that changes how the model should be described.

Section 3 is a short MLflow guide, since you asked for one, written against your actual screenshot.

Section 4 fixes four things. Sections 6 to 8 explain the new concepts before any code appears, because you said this step feels unfamiliar. Sections 9 to 15 are the code.

Section 22 has the checklist.

---

## 1. Where Step 4 left you

Total runtime 21 minutes 32 seconds.

**Candidate comparison, validation period 2018-04-20 to 2018-05-31:**

| Model | PR-AUC | Lift | ROC-AUC | Rounds | Minutes |
|-------|--------|------|---------|--------|---------|
| **lightgbm** | **0.60682** | 17.6x | 0.92751 | 617 | 0.71 |
| xgboost | 0.59907 | 17.4x | **0.93079** | 1,193 | 4.35 |
| catboost | 0.52819 | 15.4x | 0.89368 | 1,500 | 7.14 |
| logistic_regression | 0.18309 | 5.3x | 0.82095 | n/a | 1.07 |
| dummy | 0.03441 | 1.0x | 0.50000 | n/a | 0.03 |

**Cross-validation:** mean PR-AUC 0.63340, spread 0.02800.

**The uid ablation:** with 0.60682, without 0.59393, difference +0.01289. That is above the 0.005 tolerance fixed in advance, so the decision was **keep**.

**Cost analysis:** doing nothing costs $711,534 over 42 days. At the 2% review capacity the model saves $202,013 over that window, which annualises to **$1,760,894**. The chosen threshold is 0.4222, catching 44.6% of fraud.

**Kaggle:** public 0.944058, private **0.914018**.

**Registry:** `ieee-cis-fraud-detector` version 2, alias `candidate`. MLflow run `68850ae7c1264e80ba87229fa54ed899`. Final model 5.3 MB.

---

## 2. Reading your Step 4 results

### 2.1 The dummy check passed exactly

PR-AUC 0.03441 and ROC-AUC 0.50000. The validation fraud rate is 4,064 of 118,108, which is 0.034409.

This is the check I said mattered most, and it is worth understanding why. The dummy model predicts the same thing for every row, so it has learned nothing. On a metric that works correctly, PR-AUC for such a model must equal the fraud rate and ROC-AUC must equal exactly 0.5. Both came out right.

That means every other number in the run was measured with code that is behaving correctly. Without that check, a subtle bug in the metric code would have made every comparison meaningless, and you would have had no way to notice.

### 2.2 LightGBM won on PR-AUC, XGBoost won on ROC-AUC

Look carefully:

- LightGBM: PR-AUC 0.60682, ROC-AUC 0.92751
- XGBoost: PR-AUC 0.59907, ROC-AUC 0.93079

The two metrics disagree about which model is better. That is not a mistake, and it is worth understanding because it is exactly why we chose a primary metric back in Step 2 rather than reporting several and picking whichever looks best.

The two measure different things. ROC-AUC asks how well the model separates fraud from legitimate transactions across the whole range of scores, including the enormous bulk of very low-risk transactions. PR-AUC concentrates on the top of the ranking, where the transactions you would actually review live.

XGBoost is slightly better at the overall separation. LightGBM is slightly better at the part we care about, which is the riskiest one or two percent. Since the whole system is built around a review queue, PR-AUC is the right tie-breaker, and decision D-20 said so before we had any numbers.

**One honest consequence.** Kaggle scores on ROC-AUC. XGBoost would very likely have scored a little higher on the leaderboard than the 0.914018 you got. We optimised for the business problem rather than the leaderboard, which was the right call for this project, but it is worth saying plainly rather than letting the Kaggle number look like the target we were aiming at.

LightGBM also trained in 43 seconds against XGBoost's 4 minutes 21 seconds, six times faster for a marginally better result on the metric that matters. That is a real operational advantage when Step 5 starts retraining on a schedule.

### 2.3 CatBoost never early stopped, so its comparison is not clean

Read the CatBoost log again:

```
1499:   learn: 0.7919840   test: 0.5281428   best: 0.5281428 (1499)
bestTest = 0.528142846
bestIteration = 1499
```

The best iteration was 1,499 out of a maximum of 1,500. CatBoost was still improving when it ran out of rounds. It did not early stop, it hit the ceiling.

So CatBoost's 0.52819 is not its real score, it is the score of a CatBoost that was cut off. Given more rounds it would have gone higher. Whether it would have caught LightGBM's 0.60682 is unknown, and a gap of 0.079 is large, so probably not, but we cannot say that from this run.

**How to report it honestly:** "CatBoost reached 0.528 within a 1,500 round budget and had not converged, so this comparison understates it." Not "CatBoost was worse".

If you want a clean answer, that is one command and about fifteen minutes:

```powershell
# Raise MAX_BOOSTING_ROUNDS to 4000 in config/config.py first, then:
python run.py --step training --models catboost
```

That produces a fair CatBoost number without disturbing the LightGBM result already recorded. It is optional. I would do it, because "we gave every candidate the same fair chance" is a stronger claim than "we gave them all the same budget and one of them needed more".

Note that LightGBM stopped at 617 of 1,500 and XGBoost at 1,193 of 1,500, so both genuinely converged. Only CatBoost hit the wall.

### 2.4 The cross-validation shape is the most important thing in the run

| Fold | Training rows | Validation period | PR-AUC |
|------|---------------|-------------------|--------|
| 1 | 118,108 | 2017-12-26 to 2018-02-02 | 0.61833 |
| 2 | 236,216 | 2018-02-02 to 2018-03-11 | 0.63763 |
| 3 | 354,324 | 2018-03-11 to 2018-04-20 | **0.67082** |
| 4 | 472,432 | 2018-04-20 to 2018-05-31 | **0.60682** |

Read that column carefully. Each fold trains on more data than the last. Performance climbs from 0.618 to 0.638 to 0.671, exactly as more data should produce. Then fold 4, with the most training data of all, drops to 0.607, the second worst score in the set.

More data made it worse. That cannot be a data-volume story. It is a **period-difficulty** story: something about April to May 2018 is harder to predict than March was.

That is the single best motivation for everything in this step. The model is not static in quality. It varies with the period being scored, and it got worse in the most recent period we can measure. The test set runs from July to December, further away still, with no labels at all.

Which means: you cannot know how the model is doing in production by looking at the number from training. You need monitoring that watches the inputs, because the outputs will not be checkable for weeks or months. That is what Sections 12 and 13 build.

### 2.5 The uid ablation kept them, and the model leans on them hard

The rule said drop them if the cost was under 0.005 PR-AUC. The cost was 0.01289, more than double the tolerance, so they stay. The rule decided, not hindsight, which is exactly how it was supposed to work.

But look at what your SHAP ranking says about how much the model depends on them:

| Rank | Feature | Mean absolute SHAP |
|------|---------|--------------------|
| 6 | `D15_std_by_uid` | 0.10702 |
| 7 | `D15_mean_by_uid` | 0.10418 |
| 8 | `uid_freq` | 0.10305 |
| 14 | `TransactionAmt_mean_by_uid` | 0.08845 |
| 24 | `TransactionAmt_std_by_uid` | 0.05925 |
| 27 | `TransactionAmt_ratio_to_uid_mean` | 0.05488 |
| 88 | `D15_ratio_to_uid_mean` | 0.01858 |

Four of the top twenty features are uid features. The family accounts for **9.9% of the model's total explanatory weight**.

And those features are blank on roughly 82% of test rows.

So the shipped model leans meaningfully on a group of features that largely stop working on the data it will actually score. That is now the number one production risk in this project, and it is why the drift monitoring in Section 13 is built around watching them specifically.

**One correction to Step 4.** I described this as six features. Your run found seven, because the marker rule also caught `uid_freq`, which I missed when I analysed the manifest. My manifest analysis only looked at features whose *missingness* jumps, and `uid_freq` does not go missing. When a uid is unseen, the frequency lookup returns 0.0, not a blank.

That distinction matters more than it sounds. A feature that goes blank is visibly degraded, and a tree model routes blanks down a learned branch. A feature that collapses to the same constant value for 82% of rows looks perfectly healthy in a missingness check while carrying almost no information. It is the quieter and more dangerous failure, and a plain missing-value check will never catch it.

That is precisely why Section 8 introduces the Population Stability Index, which compares whole distributions rather than just counting blanks.

### 2.6 `has_identity` landed exactly where D-31 predicted

Back in Step 3 I argued the headline "fraud is 3.75x more likely with an identity record" was mostly a `ProductCD` effect, and predicted the flag would rank low.

**It ranked 270th of 284, with a mean absolute SHAP of exactly 0.0.**

The model used it zero times. Not "barely", not "a little": the gradient boosting never found a split on that column worth making, because everything it carried was already available through `ProductCD`.

The analysis was right, and it is satisfying that the prediction was specific enough to be wrong. This is a good thing to be able to talk through on the PM track: a headline number, a suspicion about why it was misleading, arithmetic that quantified the suspicion, a prediction, and a result that confirmed it.

### 2.7 The rescued V columns ranked low, and I told you to measure it the wrong way

In Step 4 I said to check whether V111 appeared near the top of the SHAP ranking, and that if it did, the rescue rule had saved something important.

Here is what your data actually shows:

| Feature | SHAP rank | Mean absolute SHAP |
|---------|-----------|--------------------|
| C3 | 108 | 0.010951 |
| id_07 | 206 | 0.001884 |
| V108 | 211 | 0.001701 |
| V111 | **259** | **0.000091** |
| V121, V120 | 267, 268 | **0.000000** |

V111, the column with a 46% fraud rate on its rare rows and a 13.2x lift, ranks 259th of 284.

**That looks like the rescue rule was pointless. It is not. I gave you the wrong measuring stick.**

Here is the problem. Mean absolute SHAP is an **average across all rows**. V111 holds the value 1.0 on 99.71% of rows. On those rows it contributes nothing, correctly, because it says nothing. It only speaks on the 0.29% of rows where it is something else.

SHAP was computed on a 5,000 row sample. At 0.29%, that sample contains roughly **14 rows** where V111 is not 1.0. Even if V111 completely determines the prediction on those 14 rows, averaging a huge effect over 14 rows and nothing over 4,986 gives a number very close to zero.

So the ranking is telling you "V111 rarely matters", which is true. It is not telling you "V111 does not matter when it fires", which is the question the rescue rule was actually about.

**The right measure for a rare-but-decisive feature is the maximum absolute SHAP, or the average taken only over the rows where the feature holds its rare value.** Section 4.4 patches the training code to record both from now on.

This is a genuinely useful thing to have learned, and it generalises well beyond this project. Mean importance systematically undervalues features that matter enormously but rarely, which in fraud detection is a whole category of the most valuable signals. Any "top 20 features" chart built on mean importance quietly hides them.

For what it is worth, C3 at rank 108 with a real non-zero contribution is the rescue rule visibly paying off in a way that mean importance *can* see, because C3's rare value covers 0.4% of rows and drives the prediction strongly downward.

### 2.8 The model catches cheap fraud and misses expensive fraud

This one I derived from your own reported numbers, and it changes how the model should be described.

Your cost figures: doing nothing costs $711,534 across 4,064 frauds. Subtracting the chargeback fees of 4,064 × $25 gives $609,934 of actual fraud value, so the average fraud is $150.08. At the 2% review rate the model catches 44.6% of frauds, which is 1,812 of them, and saves $202,013.

Working backwards through the cost formula, the only value of caught fraud consistent with those savings is about $190,268.

| Measure | Value |
|---------|-------|
| Recall by **count** | 44.6% |
| Recall by **value** | **31.2%** |
| Average amount of a caught fraud | **$105** |
| Average amount of a missed fraud | **$186** |

The model catches almost half of fraudulent transactions but under a third of fraudulent money. **Missed frauds are on average 77% larger than caught ones.**

That is not a bug, and it is not surprising once you think about it. A large fraudulent purchase looks a lot like a large legitimate purchase. Small-value fraud tends to come in patterns, from repeated testing behaviour or bursts on a compromised card, and patterns are exactly what a model finds.

**Why it matters.** Every headline about this model should say "44.6% of fraud cases" and not "44.6% of fraud". Someone reading the recall figure and multiplying it by total fraud losses will overstate the benefit by about 43%. The cost model already accounts for this correctly, which is why the annualised saving is $1.76M and not the $2.5M a count-based estimate would suggest. But the recall number on its own is misleading if quoted without the value figure beside it.

**What to do about it.** Three options, ordered by effort:

1. **Report both numbers.** Free, honest, and it goes in the README today.
2. **Add a value-based rule alongside the model.** Any transaction over some amount goes to review regardless of score. Crude, but high-value fraud is where the money is, and a flat rule catches what the model misses.
3. **Train with amount weighting.** Weight each training example by its transaction amount so the model is penalised more for missing expensive fraud. This would probably lower PR-AUC by count while raising recall by value.

Option 3 is genuinely interesting and would make a strong addition, but it is a modelling change and Step 5 is about the engineering layer. Section 21 records it as a candidate for future work. Option 1 is done in the README in Section 18.

**Verify it yourself** rather than trusting my algebra:

```powershell
python -c "import pandas as pd, joblib, numpy as np; from config.config import *; d=pd.read_parquet(FEATURES_TRAIN_FILE); v=d[d[SPLIT_COLUMN]=='valid']; m=joblib.load(SELECTION_MODEL_FILE if SELECTION_MODEL_FILE.exists() else FINAL_MODEL_FILE); import json; f=json.loads(MODEL_METADATA_FILE.read_text())['feature_names']; s=m.predict_proba(v[f])[:,1]; k=int(len(v)*0.02); top=np.argsort(-s)[:k]; y=v[TARGET_COLUMN].to_numpy(); a=v['TransactionAmt'].to_numpy(); print('recall by count:', y[top].sum()/y.sum()); print('recall by value:', (y[top]*a[top]).sum()/(y*a).sum())"
```

Run this after Section 13 creates `selection_model.joblib`. If those two numbers come out near 0.446 and 0.312, the finding is confirmed.

### 2.9 The Kaggle gap

Public 0.944058, private 0.914018, a gap of 3 points.

Some gap between public and private is normal, since they are different slices of the test set. Three points is on the larger side. It is consistent with the model doing less well on data further from training, which is the same story fold 4 told, but the competition never published enough detail about how the two slices were chosen for me to claim that as the cause. Treat it as suggestive, not proven.

Your private score of 0.914 is a solid result for a single model with no ensembling, no test-set leakage, and production constraints applied throughout. The top of that leaderboard used heavy ensembling and aggressive customer-identity reconstruction, much of which does not survive contact with a real deployment.

---

## 3. The MLflow guide you asked for

Your screenshot shows the runs table with the placeholder search text still in the box. Here is how to actually use it.

Start the server from the project root, in a second terminal with the environment active:

```powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000
```

### 3.1 Reading your runs table

Your table currently shows fifteen runs, and they tell a story if you know how to read them:

- **`candidate_dummy` and `candidate_logistic_regression` from 12 hours ago**, one with a red X. That red X is the skops failure from Section 4.1. MLflow recorded the run as failed rather than losing it, which is the point of tracking: even the broken attempt is on the record.
- **A cluster from 1 hour ago ending in `final_lightgbm` with `ieee-cis-fraud-detector v1`.** That is your `--quick` run. It trained 150 rounds and **registered itself as version 1 of the production model**.
- **A second cluster ending in `final_lightgbm` with `v2`.** That is the real run.

That first point is a genuine problem, and Section 4.4 fixes it. A throwaway 150-round test model is sitting in your model registry as version 1. Nothing currently stops someone deploying it.

### 3.2 Adding the columns that matter

By default the table shows Run Name, Created, Duration, Source, Models. Your metrics are hidden behind **"Show more columns (58 total)"**.

Click **Columns** in the toolbar and tick:

- `valid_pr_auc`
- `valid_roc_auc`
- `valid_recall_at_1pct`
- `best_round`
- `fit_seconds`

Now click the `valid_pr_auc` header to sort. You have the comparison table from `model_comparison.csv`, except live, sortable, and including runs from every session rather than just the last one.

### 3.3 The search box

The grey text `metrics.rmse < 1 and params.model = "tree"` is an example, not a filter. It does nothing until you replace it.

The syntax is `field.name operator value`, where the field is one of `metrics`, `params`, `tags`, or `attributes`. String values need quotes; numbers do not.

Queries that work on your project right now:

```
tags.phase = 'candidate_comparison'
```
Hides the ablation and final runs, showing only the five model candidates.

```
metrics.valid_pr_auc > 0.5
```
The runs that actually performed. This hides the dummy, logistic regression, and every quick-mode run in one go.

```
tags.model_family = 'lightgbm' and metrics.valid_pr_auc > 0.55
```
Combining conditions with `and`.

```
attributes.status = 'FAILED'
```
Just the failed runs. Useful when something broke and you want to see what it had logged before it died.

```
tags.phase = 'final' and metrics.selection_pr_auc > 0.6
```
Final models good enough to consider promoting.

After Section 4.4 adds the `run_mode` tag, this becomes the most useful query you have:

```
tags.run_mode = 'full'
```
Everything real, with every throwaway test run hidden.

Note `attributes.status` rather than `tags.status`. Status, run name, and start time are attributes of the run itself, not things you logged.

### 3.4 Comparing runs

Tick the checkbox next to `candidate_lightgbm` and `candidate_xgboost`, then click **Compare**.

You get parameters side by side with the differing ones highlighted, and metrics side by side. This is the view that answers "what was actually different" without you having to remember.

Try it on your two `final_lightgbm` runs, the v1 quick one and the v2 real one. The `n_estimators` difference, 150 against 771, is visible immediately, and that is the whole reason one of them should never be deployed.

### 3.5 The chart view

The three icons at the top left of the table switch views. The middle one, the line-chart icon, is the chart view.

For a single boosted run it shows the metric over training rounds if you logged per-round metrics. We only log final values, so the more useful mode is selecting several runs and adding a bar chart of `valid_pr_auc`. That is a shareable version of your model comparison chart, generated without writing any plotting code.

### 3.6 The registry

Click **Models** in the top navigation, then `ieee-cis-fraud-detector`.

You have two versions. Version 2 carries the alias `candidate`. Version 1 carries nothing and is the quick-run model.

Two things to do about that, and both are one command.

**First, tag the good run so the promotion gates in Section 14 can recognise it.** Version 2's run was created before we started tagging `run_mode`, so the gate would reject it for lack of the tag. Tags can be set after the fact:

```powershell
python -c "import mlflow; from config.config import MLFLOW_TRACKING_URI; mlflow.set_tracking_uri(MLFLOW_TRACKING_URI); mlflow.MlflowClient().set_tag('68850ae7c1264e80ba87229fa54ed899', 'run_mode', 'full'); print('tagged')"
```

This is worth understanding rather than just running. Parameters and metrics are the record of what happened and should never be edited. Tags are labels *about* the run, and correcting or adding them later is normal and expected.

**Second, mark version 1 so nobody deploys it.** You can delete it, but tagging it is better: the record of the mistake stays, which is more useful than pretending it never happened.

```powershell
python -c "import mlflow; from config.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME; mlflow.set_tracking_uri(MLFLOW_TRACKING_URI); c=mlflow.MlflowClient(); c.set_model_version_tag(REGISTERED_MODEL_NAME, '1', 'do_not_deploy', 'quick-mode test run, 150 rounds'); print('tagged v1')"
```

### 3.7 What this is worth showing

When you demonstrate this project, the model is the least surprising part. Everyone has a model.

Opening this page, filtering to `tags.run_mode = 'full'`, sorting by `valid_pr_auc`, selecting two runs and comparing them, then pointing at the registry and saying "this alias is what production loads, and here are the gates a model has to pass before the alias moves" is a different kind of conversation. That is showing a process rather than a result.

---

## 4. Four fixes before we start

### 4.1 The skops error you already fixed

You hit this on the logistic regression:

```
UntrustedTypesFoundException: Untrusted types found in the file: ['numpy.dtype']
```

**Your fix is correct.** For anyone reading this later, here is what was happening.

MLflow 3 changed how it saves scikit-learn models. It used to use pickle, which can execute arbitrary code when loaded, so a malicious model file is a genuine security risk. MLflow now uses a library called skops, which saves models in a format that can be inspected before loading, with an allow-list of types it will accept.

The allow-list is conservative. A fitted `LogisticRegression` inside a `Pipeline` contains a `numpy.dtype` object, which is entirely harmless but is not on the default list. So MLflow saved the model, immediately tried to reload it to verify, and refused its own file.

Passing `skops_trusted_types=["numpy.dtype"]` says "I know what is in this file, I trained it myself, load it". Which is true.

Two things about your fix I want to call out because they were good judgement:

You guarded it with `if flavor == "sklearn" and "skops_trusted_types" in parameters`, so the argument is only passed to the flavor that needs it and only if the installed MLflow accepts it. That follows the same "check what is actually there" pattern as the `name` versus `artifact_path` handling, which is the right instinct.

You wrote a comment explaining what the list is for and how to extend it. Six months from now, when a different model trips over a different type, that comment saves someone twenty minutes.

This was a real gap in the code I gave you. Keep your version.

### 4.2 The LightGBM `eval_set` deprecation

```
LGBMDeprecationWarning: The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y' instead.
```

LightGBM 4.7 is moving away from the list-of-tuples format. Today it warns. In a future version it will stop accepting it.

Rather than guess which form your exact version wants, check. Open `src/models/candidates.py` and replace `_fit_lightgbm` with:

```python
def _fit_lightgbm(model, X_train, y_train, X_valid, y_valid):
    """
    LightGBM takes early stopping as a callback.

    eval_metric "average_precision" is PR-AUC, so training stops when the
    metric we actually care about stops improving, rather than when log loss
    does. Those are not the same point on an imbalanced problem.

    LightGBM 4.7 deprecated the old eval_set argument in favour of separate
    eval_X and eval_y. Rather than guess which form the installed version
    wants, we look at what fit actually accepts. The same approach is used
    for the MLflow log_model change in src/utils/mlflow_utils.py.
    """
    import inspect

    import lightgbm as lgb

    fit_parameters = inspect.signature(model.fit).parameters
    if "eval_X" in fit_parameters:
        evaluation = {"eval_X": X_valid, "eval_y": y_valid}
    else:
        evaluation = {"eval_set": [(X_valid, y_valid)]}

    model.fit(
        X_train,
        y_train,
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=200),
        ],
        **evaluation,
    )
    return model, int(model.best_iteration_)
```

### 4.3 The MLflow integer schema warning

This one appeared on every single model log:

```
UserWarning: Inferred schema contains integer column(s). Integer columns in Python
cannot represent missing values...
```

**Why it matters, and it is not cosmetic.** Your feature table has integer columns: the 38 `category_code` columns are `int32` and `has_identity` is `int8`. MLflow recorded the model's expected input schema from those types.

In Step 6, a transaction arrives at the API as JSON. JSON has one number type. When pandas reads it, every number becomes a float. MLflow's schema enforcement then sees a float where the schema demands an integer, and rejects the request.

You would find this out in Step 6, in a container, with an error message about schema enforcement that does not obviously point back here.

The fix is the one the warning recommends: declare those columns as floats in the schema. Open `src/pipelines/training.py` and change both places where the signature is built.

In `_train_candidates`:

```python
            # Cast the schema sample to float64 on purpose. The category code
            # columns are integers, but a transaction arriving as JSON in
            # Step 6 will have every number read back as a float, and MLflow's
            # schema enforcement would reject it. Declaring them as floats now
            # avoids an error that would otherwise surface inside a container.
            signature = infer_signature(
                X_valid.head(50).astype("float64"), scores[:50]
            )
```

And in the final run block:

```python
        signature = infer_signature(
            all_X.head(50).astype("float64"), _score(final_model, all_X.head(50))
        )
```

### 4.4 Quick runs must never reach the registry

Your registry has a 150-round test model as version 1. That happened because `--quick` changes how the model trains but nothing about what happens afterwards.

Two changes to `src/pipelines/training.py`.

**First, tag every run with its mode.** In `_train_candidates`, add a `run_mode` argument and set the tag. Change the signature and the tag block:

```python
def _train_candidates(
    candidates, X_train, y_train, X_valid, y_valid, amounts_valid, max_rounds,
    run_mode="full",
):
    ...
        with mlflow.start_run(run_name=f"candidate_{candidate.name}"):
            mlflow.set_tag("phase", "candidate_comparison")
            mlflow.set_tag("model_family", candidate.name)
            mlflow.set_tag("run_mode", run_mode)
```

Update the call site in `run_training`:

```python
    run_mode = "quick" if quick else "full"
    comparison, score_sets, fitted = _train_candidates(
        candidates, X_train, y_train, X_valid, y_valid, amounts_valid,
        max_rounds, run_mode=run_mode,
    )
```

Add the same tag to the ablation run and the final run:

```python
        mlflow.set_tag("phase", "final")
        mlflow.set_tag("model_family", winner_name)
        mlflow.set_tag("run_mode", run_mode)
```

**Second, refuse to register from a quick run.** Replace the registration block:

```python
    registered_version = None
    if quick:
        print("  QUICK MODE: skipping model registration.")
        print("  A model trained on a reduced round budget must never enter")
        print("  the registry, because nothing downstream can tell the")
        print("  difference between it and a real one.")
    else:
        try:
            registered = mlflow.register_model(
                model_info.model_uri, REGISTERED_MODEL_NAME
            )
            registered_version = registered.version
            mlflow.MlflowClient().set_registered_model_alias(
                REGISTERED_MODEL_NAME, MODEL_ALIAS_CANDIDATE, registered_version
            )
            print(f"  Registered as {REGISTERED_MODEL_NAME} version "
                  f"{registered_version}, alias '{MODEL_ALIAS_CANDIDATE}'")
        except Exception as error:  # noqa: BLE001
            print(f"  Registry step failed: {error}")
            print("  The model file and the MLflow run are still saved.")
```

**Third, save the selection model.** The monitoring stage needs a model that has never seen the validation period, and the final model has seen everything. Add this just after the winner is chosen, before the ablation section:

```python
    # Save the model that was trained on the training portion only. The final
    # model is retrained on every labelled row, so it cannot be used to score
    # the validation period honestly. Step 5 monitoring needs one that can.
    joblib.dump(winner_model, SELECTION_MODEL_FILE)
    print(f"  Saved {SELECTION_MODEL_FILE.name} for monitoring")
```

Add `SELECTION_MODEL_FILE` to the imports at the top of the file. Section 11 defines it.

If you would rather not re-run training, the monitoring stage in Section 13 trains one itself when the file is missing. It takes about a minute.

### 4.5 Record max SHAP as well as mean

Section 2.7 showed that mean absolute SHAP hides rare-but-decisive features. Fix it for future runs. In `src/pipelines/training.py`, in `_explain`, replace the importance table:

```python
    # Two different questions, two different measures.
    #
    # mean_abs_shap answers "how much does this feature move predictions on
    # average". It is the standard importance number and it is what most
    # charts show.
    #
    # max_abs_shap answers "when this feature does speak, how loudly". A
    # column like V111 holds the same value on 99.7% of rows and says nothing
    # on those rows, so its average is near zero even though it can dominate
    # a prediction on the rows where it differs. Averaging hides exactly the
    # kind of rare, decisive signal that matters most in fraud detection.
    values_array = np.abs(values.values)
    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": values_array.mean(axis=0),
                "max_abs_shap": values_array.max(axis=0),
                "rows_with_influence": (values_array > 0.01).sum(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["max_rank"] = importance["max_abs_shap"].rank(
        ascending=False
    ).astype(int)
```

Next time you run training, sort `feature_importance.csv` by `max_abs_shap` and look at where V111 lands. I expect it to be far higher than 259th. That would confirm the rescue rule saved something real and that the earlier measurement was simply the wrong tool.

---

## 5. Decisions made in this step

| ID | Decision | Why |
|----|----------|-----|
| D-47 | Every model run is tagged `run_mode`, and quick runs cannot be registered | Version 1 of the registry is a 150-round test model. Nothing downstream could tell it apart from a real one. |
| D-48 | Model schemas declare integer columns as floats | JSON has one number type, so Step 6's API would send floats where the schema demands integers and be rejected inside a container. |
| D-49 | Feature importance records max absolute SHAP alongside the mean | Mean importance systematically hides features that matter enormously but rarely, which in fraud is a whole category of the best signals. See Section 2.7. |
| D-50 | Tests run on synthetic data only, never on the real dataset | The dataset is 1.3 GB and is not in the repository. A test suite that needs it cannot run in CI, and a test suite that does not run in CI does not get run. |
| D-51 | A row-independence test is the primary leakage guard | Transforming one row must give the same answer as transforming a batch containing it. If anyone reintroduces a groupby inside `transform`, this fails immediately. It is also exactly what Step 6 needs to be true. |
| D-52 | CI installs a light dependency set, not the full environment | The full environment is about 2.5 GB and takes minutes to install. Tests only need pandas, numpy, scipy, scikit-learn, and pyarrow. |
| D-53 | Drift is measured with PSI as the primary signal, KS as a secondary, and a missingness comparison alongside | PSI catches distribution collapse, which a missingness check misses entirely. `uid_freq` is the worked example: it never goes blank, it just becomes 0.0 for 82% of rows. |
| D-54 | The KS statistic is used, never the KS p-value | With 100,000 rows every difference is statistically significant, so the p-value is always tiny and tells you nothing. The statistic measures how big the difference is, which is the question. |
| D-55 | Feature drift is weighted by SHAP importance before being turned into an overall verdict | 284 features will always contain a few that have drifted. Drift in a feature the model ignores is not a problem. Drift in the top ten is. |
| D-56 | Promotion from `candidate` to `production` runs through six gates and is a separate deliberate command | Training produces a candidate. Deciding it is fit to serve is a different decision with different evidence, and it should not happen automatically as a side effect of a training run. |
| D-57 | The monitoring stage writes a small `dashboard_data.json` | Per D-45, the Step 7 dashboard must load in under three seconds. It cannot compute anything from the 590,540 row table on page load. |
| D-58 | Recall is always reported by count **and** by value | The model catches 44.6% of fraud cases but only 31.2% of fraud money. Quoting the count alone overstates the benefit by about 43%. See Section 2.8. |

---

## 6. Concepts: testing

You said this step feels unfamiliar, so this section and the next two explain the ideas before any code appears.

### 6.1 What a test actually is

A test is a small piece of code that runs your real code with a known input and checks the answer is what it should be. That is all.

```python
def test_addition():
    assert 2 + 2 == 4
```

If the assertion holds, the test passes silently. If it fails, pytest tells you which one, what it expected, and what it got.

### 6.2 Why an ML project needs them more than most

In ordinary software, a bug usually announces itself. The page does not load, the button does nothing, an error appears.

In machine learning, a bug usually produces **a number**. A slightly wrong number. Everything runs, nothing errors, and the answer is quietly incorrect.

You have already seen this twice in this project. If `optimise_dtypes` had cast `TransactionDT` to `float32`, every time calculation would have been off by one second and nothing would have complained. If frequency encoding had been fitted on the whole training file including validation, every validation score would have been inflated and nothing would have complained.

Tests are how you catch the failures that do not announce themselves.

### 6.3 What we are going to test

Four groups, and each one exists because of something specific in this project.

**The metrics.** The cost model is the number the business case rests on. If it is wrong, the $1.76M figure is wrong. So we compute a four-row example by hand, on paper, and assert the code produces the same answer.

**The transformer round-trip.** Step 6 loads `feature_engineer.joblib` inside a container. If saving and loading changes anything, predictions in production differ from predictions in training, and nothing errors. We save, load, transform, and assert the output is identical.

**Row independence.** This is the important one. `transform` on a single row must produce the same result as `transform` on a batch containing that row. It holds today because every transformation is either row-wise or a lookup into stored state. If someone later adds a `groupby` inside `transform`, thinking it harmless, this test fails immediately. It is also exactly the property Step 6 depends on, since the API scores one transaction at a time.

**Leakage.** The split must be time-ordered. A category that only appears in the validation portion must map to -1, proving the transformer never saw it. `TransactionDT` must not be a feature.

### 6.4 Why the tests use fake data

The real dataset is 1.3 GB and is not in the repository. If tests needed it, they could not run in CI, and tests that do not run automatically are tests nobody runs.

So every test builds a small synthetic table with the right shape and the right column names, a few thousand rows, generated fresh each time. It runs in seconds and needs nothing but the code. That is D-50.

---

## 7. Concepts: continuous integration

### 7.1 What it is

Continuous integration means: every time you push code, a computer somewhere automatically checks it.

You push. GitHub notices. It spins up a fresh Linux machine, installs Python, installs your dependencies, runs your tests and your linters, and reports pass or fail on the pull request.

### 7.2 Why it beats running tests yourself

Three reasons.

**It runs on a clean machine.** Your laptop has things installed that you have forgotten about. CI starts from nothing every time, so "works on my machine" is caught immediately.

**It cannot be skipped.** Running tests before committing requires remembering, and at 11pm on a Friday nobody remembers.

**It is visible.** The badge on your README goes red the moment something breaks. A reviewer sees it before they see your code.

### 7.3 The three checks we run

**ruff** is a linter. It reads the code without running it and flags unused imports, undefined names, shadowed variables, and import ordering. Very fast.

**black** is a formatter. It has exactly one opinion about how Python should look and applies it everywhere. `black --check` fails if any file is not already formatted that way. The value is not that black's style is best, it is that nobody ever argues about it again.

**pytest** runs the tests.

### 7.4 pre-commit

CI catches problems after you push. `pre-commit` catches them before you commit, by running the same checks locally when you type `git commit`.

Same checks, earlier. If black reformats a file, the commit stops, you review the change, and commit again. Slightly annoying the first few times, then invisible.

---

## 8. Concepts: drift

### 8.1 The problem, in one sentence

Your model learnt what fraud looked like between December 2017 and May 2018. The world does not hold still, and by December 2018 fraud may look different, but you will not have labels for weeks or months so you cannot simply measure whether it still works.

### 8.2 Three kinds, and which we can detect

**Data drift.** The inputs change. Mobile transactions rise from 10% to 30%. Detectable immediately, because inputs arrive with every transaction.

**Concept drift.** The relationship changes. Something that used to indicate fraud stops indicating it. Only detectable once labels arrive, which can be months.

**Label drift.** The fraud rate itself moves. Also needs labels.

In production you nearly always find out about data drift long before you can confirm concept drift. So monitoring watches the inputs and treats a large shift as a warning that the outputs may be about to go wrong.

Your project has two real, already-measured examples, which is why this is not a theoretical exercise:

- Identity coverage moved from 24.4% in training to 28.0% in test.
- The uid family goes from 11% blank in training to 82% blank in test, and the model gives that family 9.9% of its explanatory weight.

Neither was manufactured. Both are in the data.

### 8.3 PSI, explained from scratch

Population Stability Index measures how far a distribution has moved.

Take a feature, say `TransactionAmt`. In the training data, cut it into ten equal-sized buckets, so each holds 10% of training rows. Bucket 1 is the cheapest tenth of transactions, bucket 10 the most expensive.

Now take December's transactions and drop them into those same buckets. If December looks like training, roughly 10% lands in each. If December has far more expensive transactions, the top buckets fill up and the bottom ones empty out.

PSI turns that comparison into one number:

```
PSI = sum over buckets of  (new% - old%) x ln(new% / old%)
```

The multiplication is what makes it useful. A bucket that moved from 10% to 12% contributes a little. A bucket that moved from 10% to 40% contributes a lot. A bucket that emptied from 10% to 0.5% contributes a lot too, because the logarithm punishes proportional collapse.

The conventional reading, which we use:

| PSI | Meaning |
|-----|---------|
| Under 0.10 | Stable, no action |
| 0.10 to 0.25 | Moderate shift, worth watching |
| Over 0.25 | Significant shift, investigate |

**Why PSI and not just a missingness check.** `uid_freq` never goes blank. When a uid is unseen the lookup returns 0.0. So a missingness check sees a perfectly healthy feature while 82% of rows collapse onto a single value. PSI sees that immediately, because eight of the ten buckets empty out. That is D-53, and `uid_freq` is why it exists.

### 8.4 KS, explained from scratch

The Kolmogorov-Smirnov statistic is a second opinion, built differently.

Draw the cumulative curve for the training data: for each value, what share of rows is at or below it. Draw the same curve for December. The KS statistic is the **largest vertical gap** between those two curves.

It ranges from 0, identical, to 1, no overlap at all. It needs no buckets, so it cannot be fooled by an unlucky bucket choice, and it is more sensitive to a shift in the middle of a distribution than PSI is.

**One critical practical note, which is D-54.** KS comes with a p-value, and every tutorial reports it. On 100,000 rows, the p-value is essentially always near zero, because with that much data even a difference of no practical consequence is statistically detectable. Reporting it would mean flagging all 284 features every month.

Use the statistic, which measures **how big** the difference is. Ignore the p-value, which only tells you the difference is real, and at this sample size it always is.

### 8.5 Why drift is weighted by importance

284 features will always contain a handful that have drifted. Most of them will be features the model barely uses.

So a raw count of drifted features is a bad alarm: it fires constantly and gets ignored, which is worse than no alarm.

Instead we weight each feature's PSI by its SHAP importance and add them up. Drift in `C13`, the model's top feature, dominates. Drift in `has_identity`, which the model never uses at all, contributes exactly nothing. That is D-55, and it is what makes the alarm worth listening to.

### 8.6 Promotion gates

Right now, training a model automatically points the `candidate` alias at it. That is fine, because "candidate" means "this exists".

`production` should be different. Before a model is allowed to serve real traffic, it should have to pass explicit checks:

1. It came from a full run, not a quick test
2. Its validation PR-AUC clears a floor
3. Its cross-validation spread is not wildly unstable
4. It is not worse than the model currently in production
5. It has a chosen operating threshold, not a default 0.5
6. Its feature list matches what the transformer produces

Gate 1 alone would have stopped your version 1. Gate 6 catches the case where somebody retrains the feature engineer, changes the feature count, and the model silently expects columns the transformer no longer produces.

Promotion is a separate, deliberate command. Training says "here is a candidate". Promotion says "this is fit to serve". Different decisions, different evidence. That is D-56.

---

## 9. What Step 5 produces

**New code:**

| File | Purpose |
|------|---------|
| `pyproject.toml` | ruff, black, pytest, and coverage configuration in one place |
| `requirements-ci.txt` | The light dependency set CI installs |
| `tests/conftest.py` | Synthetic data fixtures shared by every test |
| `tests/test_metrics.py` | The cost model checked against hand arithmetic |
| `tests/test_feature_engineer.py` | Round-trip and row-independence |
| `tests/test_leakage.py` | The guards against reintroducing leakage |
| `tests/test_drift.py` | PSI and KS behave as expected |
| `src/monitoring/drift.py` | PSI, KS, missingness comparison, verdicts |
| `src/monitoring/promotion.py` | The six gates |
| `src/utils/monitoring_plots.py` | Four monitoring charts |
| `src/pipelines/monitoring.py` | The monitoring stage |
| `scripts/promote_model.py` | The promotion command |
| `.github/workflows/ci.yml` | The CI pipeline |
| `.pre-commit-config.yaml` | Local pre-commit hooks |

**New outputs:**

| File | Contents |
|------|----------|
| `reports/monitoring/feature_drift.csv` | PSI, KS, and missingness per feature per month |
| `reports/monitoring/period_metrics.csv` | Weekly PR-AUC on the labelled held-out period |
| `reports/monitoring/score_drift.csv` | Score distribution and alert rate per month |
| `reports/monitoring/drift_summary.md` | The written findings and the verdict |
| `reports/monitoring/dashboard_data.json` | Precomputed headline numbers for Step 7 |
| `reports/figures/16` to `19` | Four monitoring charts |
| `models/selection_model.joblib` | The model trained on the training portion only |

---

## 10. Create `pyproject.toml` and `requirements-ci.txt`

### 10.1 `pyproject.toml`

Create this in the project root.

```toml
# Tool configuration for the whole project, in one file.
#
# Keeping ruff, black, and pytest settings here rather than in separate
# config files means there is one place to look, and every tool picks it
# up automatically with no extra flags.

[tool.black]
line-length = 88
target-version = ["py311"]

[tool.ruff]
line-length = 88
target-version = "py311"
# Nothing in these folders is hand-written source code.
exclude = [
    ".venv",
    ".dvc",
    "data",
    "mlruns",
    "reports",
    "notebooks",
]

[tool.ruff.lint]
# E, W  pycodestyle errors and warnings
# F     pyflakes: undefined names, unused imports
# I     import sorting
# UP    modern Python syntax
# B     common bug patterns
select = ["E", "W", "F", "I", "UP", "B"]

# E501 is line length. black already handles that, and having two tools
# argue about the same thing produces noise rather than quality.
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::FutureWarning",
]

[tool.coverage.run]
source = ["src", "config"]
omit = ["*/__init__.py"]
```

### 10.2 `requirements-ci.txt`

The full environment is about 2.5 GB and would take several minutes to install on every push. The tests only touch metrics, the feature engineer, column selection, and drift, so CI only needs what those import. That is D-52.

```text
# Light dependency set for continuous integration.
#
# The tests run on synthetic data and never import mlflow, lightgbm,
# xgboost, catboost, shap, streamlit, or fastapi. Installing the full
# environment would add several minutes to every push for no benefit.
#
# If a new test imports something not listed here, CI will fail with a
# clear ImportError, which is the right way to find out.

numpy>=1.26
pandas>=2.2
scipy>=1.13
scikit-learn>=1.5
pyarrow>=16.0
joblib>=1.4

pytest>=8.3
pytest-cov>=5.0
ruff>=0.6
black>=24.8
```

---

## 11. Update `config/config.py`

Append this before `ensure_directories`.

```python
# =========================================================
# STEP 5: MONITORING, TESTING, AND PROMOTION
# =========================================================

# ---------------------------------------------------------
# Output locations
# ---------------------------------------------------------

MONITORING_DIR = REPORTS_DIR / "monitoring"

FEATURE_DRIFT_FILE = MONITORING_DIR / "feature_drift.csv"
PERIOD_METRICS_FILE = MONITORING_DIR / "period_metrics.csv"
SCORE_DRIFT_FILE = MONITORING_DIR / "score_drift.csv"
DRIFT_SUMMARY_FILE = MONITORING_DIR / "drift_summary.md"

# Small precomputed file the Step 7 dashboard reads. Per D-45 the dashboard
# must load in under three seconds, so it cannot compute anything from the
# 590,540 row table on page load.
DASHBOARD_DATA_FILE = MONITORING_DIR / "dashboard_data.json"

# The model trained on the training portion only. The final model has seen
# every labelled row, so it cannot score the validation period honestly.
SELECTION_MODEL_FILE = MODELS_DIR / "selection_model.joblib"


# ---------------------------------------------------------
# Drift settings
# ---------------------------------------------------------

# Ten buckets is the convention. More buckets makes PSI jumpier on small
# samples; fewer makes it blind to shifts inside a bucket.
PSI_BINS = 10

PSI_STABLE = 0.10          # below this, no action
PSI_SIGNIFICANT = 0.25     # above this, investigate

# The KS test is slow on very large samples and gains nothing past a point,
# so both sides are subsampled to this size.
KS_SAMPLE_SIZE = 50_000

# How many of the model's most important features get watched closely.
DRIFT_TOP_FEATURES = 20

# A feature needs at least this many usable values in a period before its
# drift number means anything.
DRIFT_MIN_ROWS = 500

# The overall verdict fires on importance-weighted PSI rather than a raw
# count of drifted features. With 284 features, a few will always have
# drifted, and drift in a feature the model ignores does not matter. D-55.
RETRAIN_WEIGHTED_PSI = 0.15
WATCH_WEIGHTED_PSI = 0.05

# How far the alert rate may move from the expected review rate before it
# counts as a problem, as a fraction of the expected rate.
ALERT_RATE_TOLERANCE = 0.50


# ---------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------

MODEL_ALIAS_PRODUCTION = "production"

PROMOTION_MIN_PR_AUC = 0.50
PROMOTION_MAX_CV_SPREAD = 0.05
PROMOTION_REGRESSION_TOLERANCE = 0.01
```

Then add `MONITORING_DIR` to the list inside `ensure_directories`:

```python
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        EXPLAINABILITY_DIR,
        MONITORING_DIR,
    ]
```

---

## 12. The test suite

### 12.1 Create `tests/conftest.py`

`conftest.py` is a file pytest finds automatically. Anything defined in it is available to every test without importing.

A **fixture** is a function that builds something a test needs. Mark it with `@pytest.fixture`, then any test that names it as an argument gets a fresh copy.

```python
"""
Shared fixtures for the test suite.

Every test runs on synthetic data built here. The real dataset is 1.3 GB and
is not in the repository, so tests that needed it could not run in CI, and
tests that do not run automatically do not get run. That is decision D-50.

The synthetic frame mirrors the shape of the joined table: the same column
names, the same dtypes, the same mixture of numeric, text, and blank values.
It has none of the real data's signal, which is fine, because these tests
check that the machinery is correct rather than that the model is good.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.config import SECONDS_PER_DAY

N_ROWS = 3000
SEED = 7


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded random generator, so every test run is identical."""
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_joined(rng) -> pd.DataFrame:
    """
    A small stand-in for data/interim/train_joined.parquet.

    Includes every column the feature engineer needs to build its derived
    features: the time and amount columns, the uid sources (card1, addr1,
    D1), text columns for the category and email handling, id_31 and id_33
    for the device features, M columns for the match features, and a couple
    of small V blocks so the V reduction has something to reduce.
    """
    n = N_ROWS

    # Time runs forward across 120 days so a time split has something to cut.
    time_seconds = np.sort(rng.integers(SECONDS_PER_DAY, SECONDS_PER_DAY * 120, n))

    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(1_000_000, 1_000_000 + n, dtype="int32"),
            "TransactionDT": time_seconds.astype("int32"),
            "isFraud": rng.binomial(1, 0.05, n).astype("int8"),
            "TransactionAmt": np.round(rng.gamma(2.0, 60.0, n), 2),
            "has_identity": rng.binomial(1, 0.25, n).astype("int8"),
            # Identifier-style numbers: many repeats, so frequency encoding
            # and group aggregates have something to work with.
            "card1": rng.integers(1000, 1100, n).astype("int16"),
            "card2": rng.integers(100, 160, n).astype("float32"),
            "addr1": rng.integers(200, 240, n).astype("float32"),
            "D1": rng.integers(0, 90, n).astype("float32"),
            "D15": rng.integers(0, 200, n).astype("float32"),
            "C1": rng.integers(0, 20, n).astype("float32"),
            "dist1": rng.integers(0, 500, n).astype("float32"),
        }
    )

    # Text columns, stored as category exactly as the real pipeline does.
    frame["ProductCD"] = pd.Series(
        rng.choice(["W", "C", "R", "H", "S"], n), dtype="category"
    )
    frame["card4"] = pd.Series(
        rng.choice(["visa", "mastercard", "discover"], n), dtype="category"
    )
    frame["card6"] = pd.Series(rng.choice(["credit", "debit"], n), dtype="category")
    frame["P_emaildomain"] = pd.Series(
        rng.choice(["gmail.com", "yahoo.com", "hotmail.co.uk"], n), dtype="category"
    )
    frame["R_emaildomain"] = pd.Series(
        rng.choice(["gmail.com", "aol.com"], n), dtype="category"
    )
    frame["DeviceInfo"] = pd.Series(
        rng.choice(["SAMSUNG SM-G892A Build/NRD90M", "Windows", "iOS Device"], n),
        dtype="category",
    )
    frame["id_31"] = pd.Series(
        rng.choice(["chrome 62.0", "chrome 63.0", "safari generic"], n),
        dtype="category",
    )
    frame["id_33"] = pd.Series(
        rng.choice(["1920x1080", "1334x750", "2208x1242"], n), dtype="category"
    )
    frame["M1"] = pd.Series(rng.choice(["T", "F"], n), dtype="category")
    frame["M4"] = pd.Series(rng.choice(["M0", "M1", "M2"], n), dtype="category")

    # Two small V blocks. Inside each block the columns are correlated, so
    # the correlation clustering has near-duplicates to collapse.
    base_a = rng.normal(0, 1, n)
    base_b = rng.normal(0, 1, n)
    for index in range(1, 4):
        frame[f"V{index}"] = (base_a + rng.normal(0, 0.05, n)).astype("float32")
    for index in range(4, 7):
        frame[f"V{index}"] = (base_b + rng.normal(0, 0.05, n)).astype("float32")

    # Blanks in the same block-wise pattern the real V columns show.
    blank_a = rng.random(n) < 0.20
    frame.loc[blank_a, ["V1", "V2", "V3"]] = np.nan

    return frame


@pytest.fixture
def synthetic_v_groups() -> list[list[str]]:
    """The V block structure for the synthetic frame."""
    return [["V1", "V2", "V3"], ["V4", "V5", "V6"]]


@pytest.fixture
def fitted_engineer(synthetic_joined, synthetic_v_groups):
    """
    A feature engineer fitted on the first 80% of the synthetic frame.

    Fitted on the earlier portion only, exactly as the real pipeline does,
    so the leakage tests have something honest to check against.
    """
    from src.features.engineer import FraudFeatureEngineer

    cut = int(len(synthetic_joined) * 0.8)
    train_part = synthetic_joined.iloc[:cut]

    engineer = FraudFeatureEngineer(
        v_groups=synthetic_v_groups,
        verbose=False,
    )
    engineer.fit(train_part, train_part["isFraud"])
    return engineer
```

### 12.2 Create `tests/test_metrics.py`

```python
"""
Tests for the metrics and the cost model.

The cost model is the number the entire business case rests on. If it is
wrong, the annualised savings figure is wrong, and nothing about the code
running successfully would reveal that. So the central test computes a
four-row example by hand and asserts the code agrees.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import (
    best_operating_point,
    cost_curve,
    ranking_metrics,
    review_rate_metrics,
)

COST_SETTINGS = {
    "review_cost": 4.0,
    "chargeback_fee": 25.0,
    "friction_cost": 1.0,
    "recovery_rate": 0.90,
}


def test_perfect_ranking_scores_one():
    """A model that orders every fraud above every legitimate row is perfect."""
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = ranking_metrics(y, scores)

    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_constant_scores_hit_the_known_floor():
    """
    A model that predicts the same value for everyone has learnt nothing.

    On correct metrics that must give ROC-AUC of exactly 0.5 and PR-AUC of
    exactly the fraud rate. This is the same check the dummy model passed in
    the real training run, and it is the one that proves the metric code
    itself is sound.
    """
    y = np.array([0, 0, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])

    result = ranking_metrics(y, scores)

    assert result["roc_auc"] == pytest.approx(0.5)
    assert result["pr_auc"] == pytest.approx(0.25)
    assert result["pr_auc_baseline"] == pytest.approx(0.25)


def test_cost_curve_matches_hand_arithmetic():
    """
    The cost model, worked out on paper.

    Four transactions, sorted by score:
        score 0.9, fraud, $100
        score 0.8, legit, $50
        score 0.7, fraud, $200
        score 0.1, legit, $10

    Total fraud value $300, two frauds, so the baseline cost of doing
    nothing is 300 + 2 x 25 = $350.

    Flagging the top one:
        missed:      1 fraud worth $200  ->  200 + 25       = $225.00
        caught:      1 fraud worth $100  ->  4 + 0.1x(125)  =  $16.50
        false alarm: none                                    =   $0.00
        total                                                = $241.50
        savings      350 - 241.50                            = $108.50
    """
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    amounts = np.array([100.0, 50.0, 200.0, 10.0])

    curve = cost_curve(y, scores, amounts, **COST_SETTINGS)

    # Flagging nothing costs the full baseline and saves nothing.
    assert curve.loc[0, "total_cost"] == pytest.approx(350.0)
    assert curve.loc[0, "savings"] == pytest.approx(0.0)

    # Flagging the top one.
    assert curve.loc[1, "total_cost"] == pytest.approx(241.50)
    assert curve.loc[1, "savings"] == pytest.approx(108.50)
    assert curve.loc[1, "frauds_caught"] == pytest.approx(1.0)
    assert curve.loc[1, "precision"] == pytest.approx(1.0)


def test_cost_curve_catches_every_fraud_at_the_end():
    """Flagging everything must catch every fraud and leave none missed."""
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    amounts = np.array([100.0, 50.0, 200.0, 10.0])

    curve = cost_curve(y, scores, amounts, **COST_SETTINGS)
    last = curve.iloc[-1]

    assert last["frauds_caught"] == pytest.approx(2.0)
    assert last["frauds_missed"] == pytest.approx(0.0)
    assert last["recall"] == pytest.approx(1.0)
    assert last["review_rate"] == pytest.approx(1.0)


def test_capacity_constraint_is_respected():
    """The constrained optimum must never exceed the review budget."""
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.binomial(1, 0.05, n)
    scores = np.clip(y * 0.4 + rng.random(n) * 0.6, 0, 1)
    amounts = rng.gamma(2.0, 60.0, n)

    curve = cost_curve(y, scores, amounts, **COST_SETTINGS)

    unconstrained = best_operating_point(curve, capacity_rate=None)
    constrained = best_operating_point(curve, capacity_rate=0.02)

    assert constrained["review_rate"] <= 0.02 + 1e-9
    # A constraint can never help, so the unconstrained answer is at least
    # as good. This catches a sign error in the minimisation.
    assert unconstrained["savings"] >= constrained["savings"] - 1e-6


def test_review_rate_metrics_pick_the_top_scores():
    """At a 50% review rate on four rows, the top two are reviewed."""
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])

    result = review_rate_metrics(y, scores, 0.5)

    assert result["n_reviewed"] == 2
    assert result["frauds_caught"] == 1
    assert result["recall"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
```

### 12.3 Create `tests/test_feature_engineer.py`

```python
"""
Tests for the feature engineer.

Two of these guard things that would otherwise fail silently in production:
the joblib round-trip, because Step 6 loads the transformer inside a
container, and row independence, because the API scores one transaction at
a time.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from config.config import TARGET_COLUMN, TIME_COLUMN, UNSEEN_CATEGORY_CODE


def test_transform_produces_the_fitted_feature_list(fitted_engineer, synthetic_joined):
    """Output columns must match the list fixed during fit, in the same order."""
    output = fitted_engineer.transform(synthetic_joined)

    assert list(output.columns) == fitted_engineer.feature_names_
    assert len(output) == len(synthetic_joined)


def test_target_and_passthrough_never_become_features(fitted_engineer):
    """
    The answer must not be in the features, and neither must the time column.

    TransactionDT is excluded because test values sit entirely above training
    values, so a tree cannot split on it usefully. That is decision D-26.
    """
    names = fitted_engineer.feature_names_

    assert TARGET_COLUMN not in names
    assert TIME_COLUMN not in names
    assert "TransactionID" not in names
    assert "uid" not in names  # grouping only, never a feature. D-29.


def test_joblib_round_trip_changes_nothing(fitted_engineer, synthetic_joined, tmp_path):
    """
    Saving and reloading the transformer must not change a single value.

    Step 6 loads this file inside a container. If the round trip altered
    anything, production predictions would differ from training predictions
    and nothing would error. This is the test that makes deployment safe.

    tmp_path is a pytest fixture giving a fresh temporary folder that is
    cleaned up afterwards, so the test leaves nothing behind.
    """
    before = fitted_engineer.transform(synthetic_joined)

    path = tmp_path / "engineer.joblib"
    joblib.dump(fitted_engineer, path)
    reloaded = joblib.load(path)

    after = reloaded.transform(synthetic_joined)

    assert list(after.columns) == list(before.columns)
    pd.testing.assert_frame_equal(before, after)


def test_transform_is_row_independent(fitted_engineer, synthetic_joined):
    """
    Transforming one row must give the same answer as transforming a batch.

    This is the leakage guard that matters most. Every transformation in the
    engineer is either row-wise or a lookup into state stored during fit, so
    a row's features cannot depend on which other rows travel with it.

    If someone later adds a groupby inside transform, thinking it harmless,
    this test fails immediately. It is also exactly the property the Step 6
    API depends on, because it scores one transaction at a time.
    """
    batch = fitted_engineer.transform(synthetic_joined)

    for position in (0, 17, len(synthetic_joined) - 1):
        single = fitted_engineer.transform(synthetic_joined.iloc[[position]])

        assert list(single.columns) == list(batch.columns)
        np.testing.assert_allclose(
            single.to_numpy(dtype="float64"),
            batch.iloc[[position]].to_numpy(dtype="float64"),
            rtol=1e-9,
            equal_nan=True,
        )


def test_unseen_category_maps_to_the_reserved_code(
    fitted_engineer, synthetic_joined
):
    """
    A value the transformer never saw during fit must map to -1.

    This proves two things at once: that the mapping is fixed at fit time
    rather than rebuilt on each call, and that a new value at prediction time
    degrades gracefully instead of raising.
    """
    row = synthetic_joined.iloc[[0]].copy()
    row["ProductCD"] = pd.Series(
        ["A_PRODUCT_THAT_NEVER_EXISTED"], dtype="object", index=row.index
    )

    output = fitted_engineer.transform(row)

    assert output["ProductCD_code"].iloc[0] == UNSEEN_CATEGORY_CODE


def test_unseen_value_gets_zero_frequency(fitted_engineer, synthetic_joined):
    """
    A card number never seen in training must get a frequency of zero.

    Zero is the truthful answer: as far as the training data knows, this
    value does not exist. It is also the behaviour that makes uid_freq
    collapse on the real test set, which is why Section 8.3 exists.
    """
    row = synthetic_joined.iloc[[0]].copy()
    row["card1"] = 999_999

    output = fitted_engineer.transform(row)

    assert output["card1_freq"].iloc[0] == pytest.approx(0.0)


def test_fit_requires_the_target(synthetic_joined, synthetic_v_groups):
    """
    Fitting without labels must fail loudly.

    The near-constant rescue rule compares fraud rates, so it cannot run
    without the target. Failing clearly is better than silently skipping
    the rescue and quietly dropping useful columns.
    """
    from src.features.engineer import FraudFeatureEngineer

    engineer = FraudFeatureEngineer(v_groups=synthetic_v_groups, verbose=False)

    with pytest.raises(ValueError):
        engineer.fit(synthetic_joined)
```

### 12.4 Create `tests/test_leakage.py`

```python
"""
Guards against reintroducing leakage.

Leakage does not raise an error. It produces a validation score that is too
good, which looks like success. These tests assert the structural properties
that make leakage impossible, so that if someone changes the pipeline in a
way that breaks them, the build fails rather than the score improving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import TIME_COLUMN, UNSEEN_CATEGORY_CODE, VALIDATION_FRACTION


def test_time_split_puts_every_training_row_before_every_validation_row():
    """
    The split must cut on time, not at random.

    If any training row happened after any validation row, the model would be
    learning from the future. This asserts the boundary holds with no overlap.
    """
    times = np.sort(np.random.default_rng(3).integers(86_400, 10_000_000, 5000))
    frame = pd.DataFrame({TIME_COLUMN: times})

    boundary = float(frame[TIME_COLUMN].quantile(1 - VALIDATION_FRACTION))
    train = frame[frame[TIME_COLUMN] <= boundary]
    valid = frame[frame[TIME_COLUMN] > boundary]

    assert len(train) > 0 and len(valid) > 0
    assert train[TIME_COLUMN].max() <= valid[TIME_COLUMN].min()

    # Roughly the requested share, allowing for ties on the boundary value.
    assert abs(len(valid) / len(frame) - VALIDATION_FRACTION) < 0.02


def test_engineer_never_learned_anything_from_the_validation_period(
    synthetic_joined, synthetic_v_groups
):
    """
    A category present only in the validation portion must be unknown.

    This proves the transformer was fitted on the earlier rows alone. If it
    had seen the whole file, this value would have its own code instead of
    the reserved unseen code.
    """
    from src.features.engineer import FraudFeatureEngineer

    frame = synthetic_joined.copy()
    cut = int(len(frame) * 0.8)

    # Plant a value that exists only after the split boundary.
    frame["ProductCD"] = frame["ProductCD"].astype("object")
    frame.iloc[cut + 5, frame.columns.get_loc("ProductCD")] = "ONLY_IN_VALID"
    frame["ProductCD"] = frame["ProductCD"].astype("category")

    train_part = frame.iloc[:cut]
    engineer = FraudFeatureEngineer(v_groups=synthetic_v_groups, verbose=False)
    engineer.fit(train_part, train_part["isFraud"])

    planted = engineer.transform(frame.iloc[[cut + 5]])

    assert planted["ProductCD_code"].iloc[0] == UNSEEN_CATEGORY_CODE


def test_no_feature_is_a_disguised_time_index(fitted_engineer, synthetic_joined):
    """
    No feature may track the raw clock.

    Trees cannot split outside the value range they were trained on, so a
    feature that rises monotonically with time is useless at prediction time
    and worse than useless during training, because it looks helpful.

    Hour and day of week are fine, and expected to correlate weakly, so the
    bar is set at a near-perfect correlation rather than at zero.
    """
    output = fitted_engineer.transform(synthetic_joined)
    clock = synthetic_joined[TIME_COLUMN].to_numpy(dtype="float64")

    suspicious = []
    for column in output.columns:
        values = output[column].to_numpy(dtype="float64")
        usable = np.isfinite(values)
        if usable.sum() < 100 or np.nanstd(values[usable]) == 0:
            continue
        correlation = abs(np.corrcoef(clock[usable], values[usable])[0, 1])
        if correlation > 0.98:
            suspicious.append((column, round(float(correlation), 4)))

    assert not suspicious, f"features tracking the clock: {suspicious}"
```

### 12.5 Create `tests/test_drift.py`

```python
"""
Tests for the drift detectors.

These matter because a broken drift detector fails in the worst possible
way: it reports that everything is fine.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.config import PSI_SIGNIFICANT, PSI_STABLE
from src.monitoring.drift import (
    kolmogorov_smirnov,
    missing_rate,
    population_stability_index,
)


def test_psi_of_a_distribution_against_itself_is_near_zero():
    """No change must produce no signal."""
    rng = np.random.default_rng(11)
    sample = rng.normal(0, 1, 20_000)

    psi = population_stability_index(sample, sample.copy())

    assert psi < 0.01


def test_psi_flags_a_clear_shift():
    """A distribution moved by two standard deviations must trip the alarm."""
    rng = np.random.default_rng(12)
    reference = rng.normal(0, 1, 20_000)
    current = rng.normal(2, 1, 20_000)

    psi = population_stability_index(reference, current)

    assert psi > PSI_SIGNIFICANT


def test_psi_catches_a_collapse_onto_one_value():
    """
    The uid_freq failure, in miniature.

    When most rows collapse onto a single value, nothing goes blank, so a
    missingness check sees a healthy feature. PSI must catch it, because
    that is the whole reason PSI is the primary signal. Decision D-53.
    """
    rng = np.random.default_rng(13)
    reference = rng.uniform(0, 1, 20_000)

    current = rng.uniform(0, 1, 20_000)
    current[: int(0.82 * len(current))] = 0.0  # 82% collapse onto zero

    assert missing_rate(current) == pytest.approx(0.0)  # nothing is blank
    assert population_stability_index(reference, current) > PSI_SIGNIFICANT


def test_psi_tolerates_a_small_wobble():
    """Random sampling noise must not look like drift."""
    rng = np.random.default_rng(14)
    reference = rng.normal(0, 1, 20_000)
    current = rng.normal(0.02, 1, 20_000)

    assert population_stability_index(reference, current) < PSI_STABLE


def test_ks_ranges_from_zero_to_one():
    """Identical samples give roughly zero, disjoint samples give roughly one."""
    rng = np.random.default_rng(15)
    sample = rng.normal(0, 1, 5000)

    assert kolmogorov_smirnov(sample, sample.copy()) < 0.05
    assert kolmogorov_smirnov(sample, sample + 100) > 0.95


def test_psi_returns_nan_when_there_is_not_enough_data():
    """A handful of rows cannot support a distribution comparison."""
    assert np.isnan(population_stability_index(np.array([1.0, 2.0]), np.array([1.0])))


def test_missing_rate_counts_blanks():
    values = np.array([1.0, np.nan, 3.0, np.nan])
    assert missing_rate(values) == pytest.approx(0.5)
```

### 12.6 Run them

```powershell
pytest
```

Expect around 20 tests passing in a few seconds. If any fail, send me the output before continuing: a failing test here means something in Steps 3 or 4 is not behaving as documented, and that is worth knowing.

With coverage:

```powershell
pytest --cov=src --cov=config --cov-report=term-missing
```

Coverage is the share of your code lines the tests actually execute. Do not chase a high number. A test suite that touches every line while asserting nothing useful is worse than a small suite that checks the things that matter, because it creates false confidence.

---

## 13. Drift monitoring

### 13.1 Create `src/monitoring/drift.py`

```python
"""
Drift detection.

Three measures, each catching something the others miss:

  PSI          how far a distribution has moved, bucket by bucket. The
               primary signal, because it catches a collapse onto one value
               that a missingness check cannot see.
  KS           the largest gap between two cumulative curves. A second
               opinion that needs no buckets.
  missingness  the share of blanks, and how much it changed.

Only the KS statistic is used, never its p-value. On 100,000 rows every
difference is statistically significant, so the p-value would flag all 284
features every month and tell you nothing. Decision D-54.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config.config import (
    DRIFT_MIN_ROWS,
    KS_SAMPLE_SIZE,
    PSI_BINS,
    PSI_SIGNIFICANT,
    PSI_STABLE,
    RANDOM_SEED,
)

# Stops a bucket that emptied completely from producing infinity.
EPSILON = 1e-6


def _usable(values) -> np.ndarray:
    """Drop blanks and infinities, returning a plain float array."""
    array = np.asarray(values, dtype="float64")
    return array[np.isfinite(array)]


def missing_rate(values) -> float:
    """Share of values that are blank or infinite."""
    array = np.asarray(values, dtype="float64")
    if array.size == 0:
        return float("nan")
    return float((~np.isfinite(array)).mean())


def population_stability_index(reference, current, bins: int = PSI_BINS) -> float:
    """
    How far has this distribution moved?

    Cut the reference into equal-sized buckets, then see what share of the
    current data lands in each. If nothing changed, each bucket still holds
    about the same share and the answer is near zero.

        PSI = sum over buckets of (new share - old share) x ln(new / old)

    The multiplication is what gives it teeth. A bucket that moved from 10%
    to 12% barely registers. One that emptied from 10% to 0.5% contributes
    heavily, because the logarithm punishes proportional collapse. That is
    exactly the failure mode we need to catch: a feature that stops varying
    without ever going blank.

    Reading it:
        under 0.10   stable
        0.10 to 0.25 moderate, worth watching
        over 0.25    significant, investigate
    """
    reference_values = _usable(reference)
    current_values = _usable(current)

    if len(reference_values) < DRIFT_MIN_ROWS or len(current_values) < DRIFT_MIN_ROWS:
        return float("nan")

    # Bucket edges come from the reference, so the reference is 10% per
    # bucket by construction and the current data is what moves.
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))

    # A column with only one or two distinct values cannot be bucketed.
    if len(edges) < 3:
        return float("nan")

    # Open the outer edges so values beyond the training range are counted
    # rather than dropped. Those are exactly the ones worth noticing.
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_share = np.histogram(reference_values, bins=edges)[0] / len(
        reference_values
    )
    current_share = np.histogram(current_values, bins=edges)[0] / len(current_values)

    reference_share = np.clip(reference_share, EPSILON, None)
    current_share = np.clip(current_share, EPSILON, None)

    return float(
        np.sum(
            (current_share - reference_share)
            * np.log(current_share / reference_share)
        )
    )


def kolmogorov_smirnov(reference, current) -> float:
    """
    The largest vertical gap between two cumulative distribution curves.

    Runs from 0, identical, to 1, no overlap at all. Needs no buckets, so it
    cannot be fooled by an unlucky bucket choice, and it is more sensitive
    than PSI to a shift in the middle of a distribution.

    Both sides are subsampled, because the statistic settles down long before
    100,000 rows and the test is slow on large inputs.
    """
    reference_values = _usable(reference)
    current_values = _usable(current)

    if len(reference_values) < DRIFT_MIN_ROWS or len(current_values) < DRIFT_MIN_ROWS:
        return float("nan")

    rng = np.random.default_rng(RANDOM_SEED)
    if len(reference_values) > KS_SAMPLE_SIZE:
        reference_values = rng.choice(reference_values, KS_SAMPLE_SIZE, replace=False)
    if len(current_values) > KS_SAMPLE_SIZE:
        current_values = rng.choice(current_values, KS_SAMPLE_SIZE, replace=False)

    # .statistic only. The p-value is deliberately ignored, see D-54.
    return float(ks_2samp(reference_values, current_values).statistic)


def drift_band(psi: float) -> str:
    """Turn a PSI number into a word a human can act on."""
    if not np.isfinite(psi):
        return "unknown"
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def compare_features(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    period_label: str,
) -> pd.DataFrame:
    """Run all three measures on every feature, for one period."""
    records = []

    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue

        reference_values = reference[feature].to_numpy(dtype="float64")
        current_values = current[feature].to_numpy(dtype="float64")

        psi = population_stability_index(reference_values, current_values)
        missing_reference = missing_rate(reference_values)
        missing_current = missing_rate(current_values)

        records.append(
            {
                "period": period_label,
                "feature": feature,
                "psi": psi,
                "ks": kolmogorov_smirnov(reference_values, current_values),
                "band": drift_band(psi),
                "missing_reference": missing_reference,
                "missing_current": missing_current,
                "missing_change": missing_current - missing_reference,
                "mean_reference": float(np.nanmean(reference_values))
                if np.isfinite(reference_values).any()
                else float("nan"),
                "mean_current": float(np.nanmean(current_values))
                if np.isfinite(current_values).any()
                else float("nan"),
            }
        )

    return pd.DataFrame(records)


def weighted_drift_score(
    drift: pd.DataFrame, importance: pd.DataFrame
) -> float:
    """
    One number for the whole period, weighted by how much the model cares.

    With 284 features a few will always have drifted. A raw count fires every
    month and gets ignored, which is worse than no alarm at all. Weighting by
    SHAP importance means drift in C13, the model's top feature, dominates,
    while drift in has_identity, which the model never uses, contributes
    nothing. Decision D-55.
    """
    weights = importance.set_index("feature")["mean_abs_shap"]
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float("nan")

    merged = drift.copy()
    merged["weight"] = merged["feature"].map(weights).fillna(0.0)
    merged = merged[np.isfinite(merged["psi"])]

    if merged.empty:
        return float("nan")

    return float((merged["psi"] * merged["weight"]).sum() / total_weight)
```

### 13.2 Create `src/utils/monitoring_plots.py`

```python
"""Charts for the monitoring stage."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from config.config import PSI_SIGNIFICANT, PSI_STABLE  # noqa: E402

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
    PR-AUC week by week on labelled data the model never trained on.

    This is the only honest performance measurement available, because the
    test period has no labels. A downward slope here is the clearest possible
    signal that the model needs retraining.
    """
    figure, axis = plt.subplots(figsize=(11, 5))

    axis.plot(
        period_metrics["period"],
        period_metrics["pr_auc"],
        marker="o",
        color=ACCENT_COLOUR,
        linewidth=1.8,
    )
    mean_value = period_metrics["pr_auc"].mean()
    axis.axhline(
        mean_value, color=NEUTRAL_COLOUR, linestyle="--", label=f"mean {mean_value:.4f}"
    )

    for _, row in period_metrics.iterrows():
        axis.annotate(
            f"{row['pr_auc']:.3f}",
            (row["period"], row["pr_auc"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
        )

    axis.set_xlabel("Week of the held-out validation period")
    axis.set_ylabel("PR-AUC")
    axis.set_title("Model performance week by week, on data it never trained on")
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

    for bar, value in zip(bars, score_drift["alert_rate"]):
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
```

### 13.3 Create `src/pipelines/monitoring.py`

```python
"""
Monitoring stage.

Answers two questions that production has to answer without labels.

  1. Is the model still working?
     Only measurable on labelled data, so we score the held-out validation
     period week by week using a model that never saw it.

  2. Has the data changed?
     Measurable immediately, so we compare every month of the unlabelled
     test period against the training distribution.

Input:  data/processed/train_features.parquet
        data/processed/test_features.parquet
        models/selection_model.joblib   (built here if missing)
        reports/feature_importance.csv
Output: reports/monitoring/*
        reports/figures/16 to 19

Run with:
    python run.py --step monitoring
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from config.config import (
    ALERT_RATE_TOLERANCE,
    DASHBOARD_DATA_FILE,
    DRIFT_SUMMARY_FILE,
    DRIFT_TOP_FEATURES,
    FEATURE_DRIFT_FILE,
    FEATURE_IMPORTANCE_FILE,
    FEATURES_TEST_FILE,
    FEATURES_TRAIN_FILE,
    FIGURES_DIR,
    MODEL_METADATA_FILE,
    PERIOD_METRICS_FILE,
    PSI_SIGNIFICANT,
    REFERENCE_DATETIME,
    RETRAIN_WEIGHTED_PSI,
    SCORE_DRIFT_FILE,
    SELECTION_MODEL_FILE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_SPLIT_LABEL,
    VALID_SPLIT_LABEL,
    WATCH_WEIGHTED_PSI,
    ensure_directories,
)
from src.monitoring.drift import compare_features, weighted_drift_score
from src.utils.metrics import ranking_metrics
from src.utils.monitoring_plots import (
    plot_alert_rate,
    plot_feature_drift,
    plot_performance_over_time,
    plot_score_drift,
)


def _timestamps(seconds: np.ndarray) -> pd.Series:
    """Turn the seconds counter into readable dates, for grouping only."""
    reference = pd.Timestamp(REFERENCE_DATETIME)
    return reference + pd.to_timedelta(pd.Series(seconds), unit="s")


def _load_metadata() -> dict:
    if not MODEL_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"{MODEL_METADATA_FILE} not found.\n"
            f"Run  python run.py --step training  first."
        )
    return json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))


def _get_selection_model(train_frame: pd.DataFrame, metadata: dict):
    """
    Load the model trained on the training portion only, or build one.

    The final model has seen every labelled row, including the validation
    period, so scoring that period with it would be meaningless. We need one
    that genuinely never saw those weeks.
    """
    if SELECTION_MODEL_FILE.exists():
        print(f"  Loading {SELECTION_MODEL_FILE.name} ...")
        return joblib.load(SELECTION_MODEL_FILE)

    print("  No selection model found. Training one on the train portion ...")
    print("  (this takes about a minute and is saved for next time)")

    from src.models.candidates import build_candidates, rebuild_for_refit

    family = metadata.get("model_family", "lightgbm")
    rounds = max(50, int(metadata.get("n_estimators", 600) / 1.25))

    candidates = build_candidates(rounds, include=[family])
    if not candidates:
        raise ValueError(f"cannot rebuild model family '{family}'")

    features = metadata["feature_names"]
    train_rows = train_frame[train_frame[SPLIT_COLUMN] == TRAIN_SPLIT_LABEL]

    model = rebuild_for_refit(candidates[0], rounds)
    model.fit(train_rows[features], train_rows[TARGET_COLUMN].to_numpy())

    joblib.dump(model, SELECTION_MODEL_FILE)
    print(f"  Saved {SELECTION_MODEL_FILE.name}")
    return model


def _weekly_performance(
    valid_frame: pd.DataFrame, model, features: list[str]
) -> pd.DataFrame:
    """
    PR-AUC week by week on the held-out validation period.

    The only honest performance number available, because this is the last
    labelled data the model did not train on.
    """
    scores = model.predict_proba(valid_frame[features])[:, 1]
    labels = valid_frame[TARGET_COLUMN].to_numpy()
    weeks = _timestamps(valid_frame[TIME_COLUMN].to_numpy()).dt.to_period("W")

    records = []
    for week, index in pd.Series(range(len(weeks))).groupby(weeks.to_numpy()):
        positions = index.to_numpy()
        week_labels = labels[positions]

        # A week with no fraud at all cannot produce a PR-AUC.
        if week_labels.sum() < 10 or len(positions) < 500:
            continue

        metrics = ranking_metrics(week_labels, scores[positions])
        records.append(
            {
                "period": str(week),
                "rows": len(positions),
                "frauds": int(week_labels.sum()),
                "fraud_rate": float(week_labels.mean()),
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    return pd.DataFrame(records)


def run_monitoring() -> dict:
    print("=" * 60)
    print("STAGE: MONITORING")
    print("=" * 60)

    ensure_directories()
    metadata = _load_metadata()
    features = metadata["feature_names"]
    threshold = float(metadata["chosen_threshold"])
    expected_rate = float(metadata["chosen_review_rate"])

    print(f"  Model: {metadata['model_family']}, {len(features)} features")
    print(f"  Operating threshold: {threshold:.4f} "
          f"(expected alert rate {expected_rate:.2%})")

    # --- load ----------------------------------------------------------
    print(f"\n  Loading {FEATURES_TRAIN_FILE.name} ...")
    train_frame = pd.read_parquet(FEATURES_TRAIN_FILE)
    reference = train_frame[train_frame[SPLIT_COLUMN] == TRAIN_SPLIT_LABEL]
    valid_frame = train_frame[train_frame[SPLIT_COLUMN] == VALID_SPLIT_LABEL]
    print(f"    reference (train portion): {len(reference):,} rows")

    model = _get_selection_model(train_frame, metadata)

    # --- 1. performance on labelled held-out data -----------------------
    print("\n  Measuring performance week by week on held-out labelled data ...")
    period_metrics = _weekly_performance(valid_frame, model, features)
    period_metrics.to_csv(PERIOD_METRICS_FILE, index=False)
    for _, row in period_metrics.iterrows():
        print(f"    {row['period']}  rows {int(row['rows']):>6,}  "
              f"frauds {int(row['frauds']):>4,}  PR-AUC {row['pr_auc']:.4f}")

    # --- 2. drift on the unlabelled test period --------------------------
    print(f"\n  Loading {FEATURES_TEST_FILE.name} ...")
    test_frame = pd.read_parquet(FEATURES_TEST_FILE)
    test_months = _timestamps(test_frame[TIME_COLUMN].to_numpy()).dt.to_period("M")
    print(f"    {len(test_frame):,} rows across {test_months.nunique()} months")

    importance = (
        pd.read_csv(FEATURE_IMPORTANCE_FILE)
        if FEATURE_IMPORTANCE_FILE.exists()
        else pd.DataFrame({"feature": features, "mean_abs_shap": 1.0})
    )
    top_features = importance.nlargest(DRIFT_TOP_FEATURES, "mean_abs_shap")[
        "feature"
    ].tolist()

    print("\n  Comparing each month against the training distribution ...")
    drift_frames = []
    score_records = []

    for month in sorted(test_months.unique()):
        month_rows = test_frame[(test_months == month).to_numpy()]
        if len(month_rows) < 1000:
            continue

        label = str(month)
        drift = compare_features(reference, month_rows, features, label)
        drift_frames.append(drift)

        scores = model.predict_proba(month_rows[features])[:, 1]
        alert_rate = float((scores >= threshold).mean())
        weighted = weighted_drift_score(drift, importance)

        significant = int((drift["psi"] > PSI_SIGNIFICANT).sum())
        significant_top = int(
            (
                drift[drift["feature"].isin(top_features)]["psi"] > PSI_SIGNIFICANT
            ).sum()
        )

        score_records.append(
            {
                "period": label,
                "rows": len(month_rows),
                "score_mean": float(scores.mean()),
                "score_p50": float(np.percentile(scores, 50)),
                "score_p90": float(np.percentile(scores, 90)),
                "score_p99": float(np.percentile(scores, 99)),
                "alert_rate": alert_rate,
                "alert_rate_ratio": alert_rate / expected_rate
                if expected_rate
                else float("nan"),
                "weighted_psi": weighted,
                "features_significant": significant,
                "top_features_significant": significant_top,
            }
        )

        print(
            f"    {label}  rows {len(month_rows):>7,}  "
            f"alerts {alert_rate:>6.2%}  "
            f"weighted PSI {weighted:>6.3f}  "
            f"drifted {significant:>3}/{len(features)} "
            f"(top20: {significant_top})"
        )

    feature_drift = pd.concat(drift_frames, ignore_index=True)
    feature_drift.to_csv(FEATURE_DRIFT_FILE, index=False)

    score_drift = pd.DataFrame(score_records)
    score_drift.to_csv(SCORE_DRIFT_FILE, index=False)

    # --- 3. the verdict ---------------------------------------------------
    latest = score_drift.iloc[-1]
    worst_weighted = float(score_drift["weighted_psi"].max())
    alert_ratio = float(latest["alert_rate_ratio"])
    alert_off = abs(alert_ratio - 1.0) > ALERT_RATE_TOLERANCE

    if worst_weighted > RETRAIN_WEIGHTED_PSI or int(latest["top_features_significant"]) >= 3:
        verdict = "RETRAIN"
    elif worst_weighted > WATCH_WEIGHTED_PSI or alert_off:
        verdict = "WATCH"
    else:
        verdict = "OK"

    print(f"\n  Verdict: {verdict}")
    print(f"    worst weighted PSI       : {worst_weighted:.4f} "
          f"(retrain above {RETRAIN_WEIGHTED_PSI})")
    print(f"    latest alert rate        : {latest['alert_rate']:.2%} "
          f"against an expected {expected_rate:.2%}")
    print(f"    top-20 features drifted  : {int(latest['top_features_significant'])}")

    # --- 4. charts ----------------------------------------------------------
    print("\n  Generating charts ...")
    if not period_metrics.empty:
        plot_performance_over_time(period_metrics, FIGURES_DIR)
    plot_feature_drift(feature_drift, top_features, FIGURES_DIR)
    plot_score_drift(score_drift, FIGURES_DIR)
    plot_alert_rate(score_drift, expected_rate, FIGURES_DIR)

    # --- 5. the dashboard file ------------------------------------------------
    # Small on purpose. Per D-45 the Step 7 dashboard must load in under three
    # seconds, so everything it shows has to be precomputed into a file this
    # size rather than derived from the feature tables at page load.
    dashboard = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": metadata["model_family"],
        "registered_version": metadata.get("registered_version"),
        "n_features": len(features),
        "threshold": threshold,
        "expected_alert_rate": expected_rate,
        "selection_pr_auc": metadata.get("selection_pr_auc"),
        "cv_pr_auc_mean": metadata.get("cv_pr_auc_mean"),
        "verdict": verdict,
        "worst_weighted_psi": worst_weighted,
        "weekly_performance": period_metrics.to_dict(orient="records"),
        "monthly_drift": score_drift.to_dict(orient="records"),
        "top_drifted_features": feature_drift.nlargest(15, "psi")[
            ["period", "feature", "psi", "missing_reference", "missing_current"]
        ].to_dict(orient="records"),
    }
    DASHBOARD_DATA_FILE.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print(f"  Wrote {DASHBOARD_DATA_FILE.name} "
          f"({DASHBOARD_DATA_FILE.stat().st_size / 1024:.0f} KB)")

    results = {
        "period_metrics": period_metrics,
        "feature_drift": feature_drift,
        "score_drift": score_drift,
        "top_features": top_features,
        "verdict": verdict,
        "worst_weighted_psi": worst_weighted,
        "expected_rate": expected_rate,
        "threshold": threshold,
        "metadata": metadata,
    }
    _write_summary(results)

    print("\n" + "=" * 60)
    print("MONITORING HEADLINES")
    print("=" * 60)
    print(f"  Verdict              : {verdict}")
    print(f"  Worst weighted PSI   : {worst_weighted:.4f}")
    print(f"  Months monitored     : {len(score_drift)}")
    print(f"  Weeks measured       : {len(period_metrics)}")
    print(f"\n  Full report: {DRIFT_SUMMARY_FILE}")

    return results


def _write_summary(results: dict) -> None:
    lines: list[str] = []
    add = lines.append

    score_drift = results["score_drift"]
    feature_drift = results["feature_drift"]
    period_metrics = results["period_metrics"]

    add("# Monitoring Summary")
    add("")
    add("Generated automatically by `src/pipelines/monitoring.py`. "
        "Do not edit by hand, it is overwritten on every run.")
    add("")

    add(f"## Verdict: {results['verdict']}")
    add("")
    add(f"- Worst importance-weighted PSI across all periods: "
        f"**{results['worst_weighted_psi']:.4f}**")
    add(f"- Retrain threshold: {RETRAIN_WEIGHTED_PSI}")
    add(f"- Operating threshold: {results['threshold']:.4f}, expected alert rate "
        f"{results['expected_rate']:.2%}")
    add("")

    add("## 1. Performance on labelled held-out data")
    add("")
    if period_metrics.empty:
        add("Not enough labelled rows per week to measure.")
    else:
        add(period_metrics.round(5).to_markdown(index=False))
        add("")
        add("These weeks are the last labelled data the model never trained on. "
            "This is the only honest performance measurement available, because "
            "the test period has no labels at all. In production you would be "
            "in the same position: weeks or months of scoring before you learn "
            "whether the scores were any good.")
    add("")

    add("## 2. Data drift, month by month")
    add("")
    add(score_drift[
        ["period", "rows", "alert_rate", "weighted_psi",
         "features_significant", "top_features_significant"]
    ].round(4).to_markdown(index=False))
    add("")
    add("`weighted_psi` weights each feature's drift by how much the model "
        "actually relies on it. With 284 features a few will always have "
        "drifted, and drift in a feature the model ignores is not a problem.")
    add("")

    add("## 3. The features that moved most")
    add("")
    add(feature_drift.nlargest(20, "psi")[
        ["period", "feature", "psi", "band", "missing_reference", "missing_current"]
    ].round(4).to_markdown(index=False))
    add("")

    add("## 4. Alert volume")
    add("")
    add("The threshold is fixed, so any change in alert volume comes entirely "
        "from the data moving. This is the number an operations manager feels "
        "directly, because it is how much work lands in the review queue.")
    add("")
    add(score_drift[["period", "alert_rate", "alert_rate_ratio"]]
        .round(4).to_markdown(index=False))
    add("")

    add("## 5. What happens next")
    add("")
    if results["verdict"] == "RETRAIN":
        add("The drift is large enough to act on. Retrain on data that includes "
            "the recent period, then run the promotion gates before the new "
            "model is allowed to serve:")
        add("")
        add("```powershell")
        add("python run.py --step features")
        add("python run.py --step training")
        add("python scripts/promote_model.py --version <new version> --dry-run")
        add("```")
    elif results["verdict"] == "WATCH":
        add("Drift is present but below the retraining threshold. Keep "
            "monitoring, and look at whether the trend is worsening month "
            "on month or holding steady.")
    else:
        add("No action needed. The data still resembles what the model was "
            "trained on.")
    add("")

    DRIFT_SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {DRIFT_SUMMARY_FILE.name}")
```

---

## 14. Promotion gates

### 14.1 Create `src/monitoring/promotion.py`

```python
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
                detail=f"candidate {pr_auc:.5f} against production {current:.5f}"
                if pr_auc and current
                else "cannot compare",
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
            detail=f"threshold = {threshold}"
            if threshold is not None
            else "no threshold recorded",
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
```

### 14.2 Create `scripts/promote_model.py`

```python
"""
Promote a registered model version to production.

Usage:
  python scripts/promote_model.py --version 2 --dry-run
  python scripts/promote_model.py --version 2

Promotion moves the 'production' alias. Step 6 loads whatever that alias
points at, so deploying a new model means moving a pointer rather than
editing and redeploying code.

--dry-run runs every gate and reports, without moving anything. Use it first,
always.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import mlflow  # noqa: E402

from config.config import (  # noqa: E402
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS_PRODUCTION,
    MODEL_METADATA_FILE,
    PREPROCESSOR_FILE,
    REGISTERED_MODEL_NAME,
)
from src.monitoring.promotion import (  # noqa: E402
    all_passed,
    evaluate_gates,
    format_gates,
)


def _production_metrics(client) -> dict | None:
    """Metrics of whatever is currently in production, if anything is."""
    try:
        version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME, MODEL_ALIAS_PRODUCTION
        )
    except Exception:  # noqa: BLE001
        return None
    return client.get_run(version.run_id).data.metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model to production.")
    parser.add_argument("--version", required=True, help="Registry version number.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check every gate and report, without moving the alias.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    print("=" * 60)
    print(f"PROMOTION CHECK: {REGISTERED_MODEL_NAME} version {args.version}")
    print("=" * 60)

    version = client.get_model_version(REGISTERED_MODEL_NAME, args.version)
    run = client.get_run(version.run_id)

    metadata = None
    if MODEL_METADATA_FILE.exists():
        candidate_metadata = json.loads(
            MODEL_METADATA_FILE.read_text(encoding="utf-8")
        )
        # The metadata file describes the most recent training run. Only use
        # it if it really is the version being promoted, otherwise the gates
        # would be checking the wrong model.
        if str(candidate_metadata.get("registered_version")) == str(args.version):
            metadata = candidate_metadata
        else:
            print(f"  Note: {MODEL_METADATA_FILE.name} describes version "
                  f"{candidate_metadata.get('registered_version')}, "
                  f"not {args.version}. Gates needing it will fail.")

    transformer_features = None
    if PREPROCESSOR_FILE.exists():
        transformer_features = list(joblib.load(PREPROCESSOR_FILE).feature_names_)

    gates = evaluate_gates(
        run_tags=run.data.tags,
        run_metrics=run.data.metrics,
        metadata=metadata,
        transformer_features=transformer_features,
        production_metrics=_production_metrics(client),
    )

    print(f"\n  run id: {version.run_id}\n")
    print(format_gates(gates))

    if not all_passed(gates):
        print("\n  RESULT: promotion refused. Fix the failures above.")
        sys.exit(1)

    if args.dry_run:
        print("\n  RESULT: every gate passed. Re-run without --dry-run to promote.")
        return

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, MODEL_ALIAS_PRODUCTION, args.version
    )
    print(f"\n  RESULT: version {args.version} is now "
          f"'{MODEL_ALIAS_PRODUCTION}'.")
    print("  Step 6 loads whatever this alias points at.")


if __name__ == "__main__":
    main()
```

---

## 15. Continuous integration

### 15.1 Create `.github/workflows/ci.yml`

```yaml
# Runs on a fresh Linux machine every time code is pushed or a pull request
# is opened. A clean machine is the point: it has none of the things your
# laptop has quietly accumulated, so "works on my machine" is caught here.

name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  quality:
    runs-on: ubuntu-latest

    steps:
      - name: Check out the repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      # The light dependency set, not the full 2.5 GB environment. Tests run
      # on synthetic data and never import mlflow, the boosting libraries,
      # shap, streamlit, or fastapi. Decision D-52.
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-ci.txt

      - name: Lint with ruff
        run: ruff check .

      - name: Check formatting with black
        run: black --check .

      - name: Run tests
        run: pytest --cov=src --cov=config --cov-report=term-missing
```

### 15.2 Create `.pre-commit-config.yaml`

```yaml
# The same checks CI runs, but on your machine when you type git commit.
# Same checks, caught earlier.
#
# After creating this file, run:
#     pre-commit install
#     pre-commit autoupdate
#
# autoupdate pins each hook to the latest release, which avoids guessing
# version tags that may not exist.

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      # Stops a Parquet file or a model binary being committed by accident.
      - id: check-added-large-files
        args: ["--maxkb=1000"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black

  # Strips notebook output before committing, so diffs stay readable.
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
```

### 15.3 Install and run the first pass

```powershell
pre-commit install
pre-commit autoupdate
```

Now the important part. This is the **first** time ruff and black have seen this codebase, so the first run will change a lot of files. Import ordering, blank lines, quote styles.

Do it deliberately, in one commit, before anything else:

```powershell
# See what ruff objects to, without changing anything
ruff check .

# Fix what can be fixed automatically, mostly import ordering
ruff check . --fix

# Reformat everything
black .

# Review carefully before committing. This will be a large diff.
git diff --stat
```

Then **run the tests again**, because a reformat should change nothing about behaviour and you want to prove it:

```powershell
pytest
```

If the tests still pass, commit it on its own:

```powershell
git add -A
git commit -m "style: apply ruff and black across the codebase"
```

Keeping the formatting change in its own commit matters. Mixed into a feature commit, it would hide the real change in hundreds of lines of whitespace.

If ruff flags something it cannot fix automatically, send me the output. Most will be unused imports, which are safe to delete, but I would rather look than have you guess.

---

## 16. Update `run.py`

**Add the stage function:**

```python
def run_monitoring_stage(args: argparse.Namespace) -> dict:
    from src.pipelines.monitoring import run_monitoring

    return run_monitoring()
```

**Update the choices:**

```python
        choices=["ingestion", "eda", "features", "training", "monitoring", "all"],
```

**Update the dispatch:**

```python
    elif args.step == "monitoring":
        run_monitoring_stage(args)
    elif args.step == "all":
        run_ingestion_stage(args)
        run_eda_stage(args)
        run_features_stage(args)
        run_training_stage(args)
        run_monitoring_stage(args)
```

**Update the docstring** to list the new stage.

---

## 17. Run it

### 17.1 Branch and tag the existing run

```powershell
git switch main
git pull
git switch -c step-05-mlops
```

Tag the real training run so the promotion gates recognise it (explained in Section 3.6):

```powershell
python -c "import mlflow; from config.config import MLFLOW_TRACKING_URI; mlflow.set_tracking_uri(MLFLOW_TRACKING_URI); mlflow.MlflowClient().set_tag('68850ae7c1264e80ba87229fa54ed899', 'run_mode', 'full'); print('tagged')"
```

### 17.2 Tests first

```powershell
pytest
```

### 17.3 Monitoring

```powershell
python run.py --step monitoring
```

Expect 3 to 6 minutes, or 5 to 8 if it has to train the selection model first.

**What to look for:**

- Six weeks of labelled performance, one row each
- Six months of drift, July through December
- The uid features should dominate the "most drifted" table. If they do not, tell me, because that would contradict everything Section 2.5 predicts.
- A verdict of `RETRAIN` or `WATCH` is expected and is a good outcome here, not a failure. It means the monitoring is detecting the shift we already know exists. A verdict of `OK` would be the surprising result.

### 17.4 The promotion gates

```powershell
python scripts/promote_model.py --version 2 --dry-run
```

Then try version 1, the quick-mode model, and watch it get refused:

```powershell
python scripts/promote_model.py --version 1 --dry-run
```

That second command is worth running even though you know the answer. Watching the gate fire on a real bad model is the moment the whole idea stops being abstract.

If version 2 passes every gate:

```powershell
python scripts/promote_model.py --version 2
```

### 17.5 Push and watch CI run

```powershell
git add -A
git commit -m "feat: add tests, ci, drift monitoring, and promotion gates"
git push -u origin step-05-mlops
```

Open the repository on GitHub, click **Actions**, and watch it run. First time takes 2 to 4 minutes.

If it fails, read the log from the bottom. The failing step is named, and the error is usually the last few lines. Common first-time failures: black finding an unformatted file that you formatted locally but did not commit, or ruff finding something in a file you forgot to fix.

---

## 18. The updated README

Your current README has four things to fix, one of which is a real error.

**The error.** It says the model is *"worth roughly $202,013 a year"*. That figure is the saving over the **42 day validation window**, not a year. The annual figure is $1,760,894. Understating your result by a factor of nine is an unusual direction to get it wrong in, but it is still wrong.

**The others:** a typo, `TStability`; two TBD values that your run filled in; and the roadmap still showing Step 4 unticked.

I have also added the count-versus-value distinction from Section 2.8, the model comparison table, the CI badge, and a short section on how the project is kept honest, which is the part that distinguishes this from a notebook.

**Replace the entire contents of `README.md`:**

````markdown
# IEEE-CIS Fraud Detection

[![CI](https://github.com/Dee-ui/ieee-cis-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Dee-ui/ieee-cis-fraud-detection/actions/workflows/ci.yml)

An end-to-end machine learning and MLOps project that detects fraudulent card
transactions, covering the full lifecycle from raw data to a monitored,
containerised, deployed service with an interactive dashboard.

> Status: in progress. Steps 1 to 5 of 7 complete.

---

## Headline result

A LightGBM model that catches **44.6% of fraudulent transactions while
reviewing 2% of all traffic**, worth roughly **$1.76M a year** in prevented
losses under a documented cost model.

Scored **0.914 ROC-AUC on the Kaggle private leaderboard** as a single model,
with no ensembling and no test-set leakage.

---

## Problem

Card fraud is rare and expensive.

Rare, because about 3.5% of transactions in this dataset are fraudulent. A model
that predicts "never fraud" is 96.5% accurate and completely useless, which is
why accuracy is not used anywhere in this project.

Expensive, because the two ways of being wrong cost different things. A missed
fraud is a direct loss. A false alarm blocks a real customer's payment. Any
useful system has to be tuned against a real review capacity rather than
optimised in the abstract.

## Dataset

IEEE-CIS Fraud Detection (Kaggle competition, data provided by Vesta
Corporation).

| Table | Rows | Columns | Contents |
|-------|------|---------|----------|
| `train_transaction` | 590,540 | 394 | Transaction level, carries the `isFraud` label |
| `train_identity` | 144,233 | 41 | Device and network signals, only for some transactions |
| `test_transaction` | 506,691 | 393 | No label |
| `test_identity` | 141,907 | 41 | No label |

The two tables join on `TransactionID`. Most columns are anonymised: 339 are
engineered features supplied by Vesta with no published meaning.

Data is not stored in this repository. See Quickstart to download it.

## What the data shows

Full findings in [`reports/eda_summary.md`](reports/eda_summary.md).

![Class balance](reports/figures/01_class_balance.png)

**The imbalance.** 20,663 frauds out of 590,540, a rate of 3.4990%, roughly one
in twenty-nine.

**Time matters more than anything else.** Training covers 2017-12-01 to
2018-05-31; the test set covers 2018-07-01 to 2018-12-30, with a deliberate 30
day gap. The test set is entirely in the future, so validation is a time-based
split, never a random one.

**Fraud concentrates in identifiable places.** Product code C runs at 11.69%
against W at 2.04%. Credit cards run at 6.68% against debit at 2.43%. Mobile
devices run at 10.17% against desktop at 6.52%.

**Identity records are almost decided by product type.** The raw figures suggest
fraud is 3.75x more likely when an identity record exists. That comparison is
confounded: product W never produces one and also has the lowest fraud rate.
Restricted to products where the flag varies, the difference is 1.39x. The
model later confirmed this, ranking the flag 270th of 284 features with a SHAP
value of exactly zero.

**The 339 anonymous V columns have hidden structure.** They fall into 15 blocks
that go blank on identical rows. Eight of the fifteen interleave through each
other's number ranges, so the structure is invisible unless you compare the
actual missing patterns. Correlation clustering inside each block reduced 337
surviving columns to 137.

## Approach

- **Metric.** PR-AUC is primary, baseline 0.035. ROC-AUC secondary. Recall at a
  fixed review rate is the business headline. Accuracy is never reported.
- **Validation.** Time-based: the last 20% of the training period.
- **Missing values are left missing.** Boosted trees learn a direction for
  blanks at every split. Filling them would assert something untrue.
- **No leakage by construction.** Every learned transformation is fitted on the
  training portion only and saved as an object, so training and serving cannot
  drift apart. A test asserts that transforming one row gives the same answer as
  transforming a batch containing it.

## Results

Validation is the last 20% of the training period by time: 2018-04-20 to
2018-05-31, 118,108 transactions containing 4,064 frauds, never seen in training.

| Model | PR-AUC | ROC-AUC | Fit time |
|-------|--------|---------|----------|
| **LightGBM** | **0.6068** | 0.9275 | 43s |
| XGBoost | 0.5991 | **0.9308** | 4m 21s |
| CatBoost | 0.5282 | 0.8937 | 7m 08s |
| Logistic regression | 0.1831 | 0.8210 | 1m 04s |
| Random baseline | 0.0344 | 0.5000 | - |

CatBoost had not converged within its 1,500 round budget, so that figure
understates it.

| Metric | Baseline | This model |
|--------|----------|------------|
| PR-AUC | 0.0344 | **0.6068** (17.6x) |
| ROC-AUC | 0.500 | **0.9275** |
| Recall at 1% review rate | 1.0% | **26.6%** |
| Recall at 2% review rate | 2.0% | **44.6%** |
| Kaggle private leaderboard | 0.500 | **0.9140** |

Stability across four expanding time windows: PR-AUC **0.6334**, spread
**0.0280**.

### What it is worth

Under a cost model with five stated assumptions, documented in
[`docs/steps/step4.md`](docs/steps/step4.md), running at a 2% manual review
capacity saves **$202,013 over the 42 day validation window**, which annualises
to roughly **$1.76M a year**.

The assumptions: $4.00 per analyst review, $25.00 chargeback fee per missed
fraud, $1.00 friction per false alarm, 90% of flagged fraud actually prevented,
and a team able to review 2% of transactions. All five live in
`config/config.py`. Change one, re-run, and the figure updates.

**One caveat that matters.** The model catches 44.6% of fraud **by count** but
only 31.2% **by value**. Missed frauds average $186 against $105 for caught
ones, because a large fraudulent purchase looks much like a large legitimate
one. The cost model accounts for this correctly, but any estimate built by
multiplying the recall figure by total fraud losses would overstate the benefit
by about 43%.

These are assumptions rather than figures from a business, and the savings
estimate should be read as an order of magnitude rather than a forecast.

## How it is kept honest

- **A test suite** covering the cost model against hand arithmetic, a
  joblib round-trip of the feature transformer, and structural guards against
  leakage. Tests run on synthetic data, so they work anywhere with no dataset.
- **CI on every push**: ruff, black, and pytest on a clean machine.
- **Drift monitoring** comparing every month of the unlabelled test period
  against the training distribution, using PSI weighted by feature importance,
  because drift in a feature the model ignores is not a problem.
- **Promotion gates.** A model reaches production only by passing six checks.
  One of them exists because a quick-mode test model once registered itself.

## Architecture

_Diagram added in Step 6._

## Quickstart

```bash
git clone https://github.com/Dee-ui/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS or Linux
pip install -r requirements.lock.txt

# Data (requires a Kaggle account that has joined the competition)
kaggle auth login
python scripts/download_data.py
python scripts/verify_data.py

# Build everything
python run.py --step all
```

## Pipeline

| Stage | Command | Output |
|-------|---------|--------|
| Ingestion | `python run.py --step ingestion` | `data/interim/*_joined.parquet` |
| EDA | `python run.py --step eda` | `reports/eda_summary.md`, 10 charts |
| Features | `python run.py --step features` | 284 features, `models/feature_engineer.joblib` |
| Training | `python run.py --step training` | `models/final_model.joblib`, MLflow runs |
| Monitoring | `python run.py --step monitoring` | `reports/monitoring/*`, 4 charts |
| Promotion | `python scripts/promote_model.py --version N` | Moves the production alias |

Every stage reads a file and writes a file, so any one can be run on its own.

## Project structure

See [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for the annotated
structure, the full decision log, and current status.

## Roadmap

- [x] Step 1: Dataset acquisition, scaffold, repo, environment
- [x] Step 2: Exploratory data analysis and data understanding
- [x] Step 3: Feature engineering and preprocessing pipeline
- [x] Step 4: Model training with MLflow experiment tracking
- [x] Step 5: Testing, CI, drift monitoring, and promotion gates
- [ ] Step 6: Dockerisation and deployment
- [ ] Step 7: Dashboard and portfolio packaging

## Tech stack

Python 3.11, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, MLflow, SHAP,
DVC, pytest, ruff, GitHub Actions, FastAPI, Docker, Streamlit.

## Licence

MIT. See [`LICENSE`](LICENSE).
````

---

## 19. Commit, merge, tag

```powershell
git add pyproject.toml requirements-ci.txt
git commit -m "build: add tool configuration and a light ci dependency set"

git add tests/
git commit -m "test: add metrics, transformer, leakage, and drift tests"

git add src/monitoring/ src/utils/monitoring_plots.py src/pipelines/monitoring.py
git commit -m "feat: add drift monitoring with importance-weighted psi"

git add scripts/promote_model.py
git commit -m "feat: add promotion gates for the model registry"

git add .github/ .pre-commit-config.yaml
git commit -m "ci: run ruff, black, and pytest on every push"

git add config/config.py src/models/candidates.py src/pipelines/training.py src/utils/mlflow_utils.py
git commit -m "fix: tag run mode, block quick runs from the registry, and record max shap"

git add run.py README.md docs/ reports/
git commit -m "docs: add step 5 guide, fix the annualised savings figure in the readme"

git push -u origin step-05-mlops

gh pr create --base main --head step-05-mlops `
  --title "Step 5: tests, CI, drift monitoring, and promotion gates" `
  --body "Synthetic-data test suite including a row-independence leakage guard. GitHub Actions running ruff, black, and pytest. Drift monitoring with importance-weighted PSI. Six promotion gates. Fixes: quick runs can no longer register, model schemas declare integers as floats, max SHAP recorded alongside mean."
```

Wait for CI to go green on the pull request before merging. That is the whole point of having it.

```powershell
gh pr merge --squash --delete-branch

git switch main
git pull
git tag -a v0.5.0-step5 -m "Step 5 complete: the MLOps layer"
git push origin v0.5.0-step5
```

---

## 20. Reading your results

### 20.1 The checks

| Check | Expected | If it is off |
|-------|----------|--------------|
| Tests | All pass in a few seconds | Send me the failure. It means Step 3 or 4 is not behaving as documented. |
| Weekly PR-AUC | Six rows, values somewhere near 0.6 | Wild swings between weeks mean the weekly sample is too small to be reliable. |
| Verdict | `WATCH` or `RETRAIN` | `OK` would be surprising, given what we already know moved. |
| Most drifted features | uid family at the top | If not, tell me. It contradicts Section 2.5. |
| Version 1 promotion | Refused at gate 1 | If it passes, the `run_mode` tag is not being read. |
| CI | Green | Read the log from the bottom. |

### 20.2 What to look at closely

**The weekly performance chart.** Six weeks is not many, but the shape matters. Flat is reassuring. A downward slope across the validation period would tell you the model was already degrading before the test period even started, which would change how often it needs retraining.

**The alert rate chart.** The threshold is fixed at 0.4222, so any movement is the data moving underneath it. If December's alert rate is double July's, the review team's workload doubles without anyone changing anything. That is the operational reality a fraud manager cares about most, and it is invisible from any accuracy metric.

**The drift grid.** Read down a column for how one month compares with training. Read across a row for whether one feature keeps getting worse. A row that climbs steadily is a feature going stale.

---

## 21. Recorded for later

Two things worth doing that fall outside this step.

**Amount-weighted training.** Section 2.8 showed the model catches cheap fraud and misses expensive fraud. Weighting training examples by transaction amount would push it towards the money. It would probably lower PR-AUC by count while raising recall by value, which is a trade worth measuring. Recorded as an open question, Q-16.

**A fair CatBoost comparison.** Section 2.3 showed CatBoost hit its round ceiling. Raising `MAX_BOOSTING_ROUNDS` to 4,000 and running `python run.py --step training --models catboost` would give it a fair chance. Optional, about fifteen minutes.

---

## 22. Verification checklist

**Fixes**
- [ ] `_fit_lightgbm` uses the inspect-based eval argument
- [ ] Both signatures cast to float64
- [ ] `run_mode` tag added to all three run types
- [ ] Quick runs skip registration
- [ ] Selection model saved during training
- [ ] `_explain` records max absolute SHAP
- [ ] Run `68850ae7c1264e80ba87229fa54ed899` tagged `run_mode = full`
- [ ] Version 1 tagged `do_not_deploy`

**Setup**
- [ ] `pyproject.toml` and `requirements-ci.txt` created
- [ ] Config extended, `MONITORING_DIR` added to `ensure_directories`
- [ ] Five test files created
- [ ] `src/monitoring/drift.py` and `promotion.py` created
- [ ] `src/utils/monitoring_plots.py` and `src/pipelines/monitoring.py` created
- [ ] `scripts/promote_model.py` created
- [ ] `run.py` updated with `monitoring`

**Running**
- [ ] `pytest` passes
- [ ] `ruff check .` clean after `--fix`
- [ ] `black .` applied and committed separately
- [ ] `pytest` still passes after reformatting
- [ ] `python run.py --step monitoring` completed
- [ ] Six months monitored, weekly performance measured
- [ ] `dashboard_data.json` under about 200 KB
- [ ] Version 2 passes the gates; version 1 is refused

**CI**
- [ ] Workflow created, `pre-commit install` run
- [ ] CI green on the pull request
- [ ] Badge showing in the README

**Git**
- [ ] README replaced, the $202,013 error fixed
- [ ] Merged, tagged `v0.5.0-step5`

---

## 23. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'src'` when running pytest | pytest run from the wrong folder | Run from the project root. `pyproject.toml` sets `testpaths`. |
| `test_transform_is_row_independent` fails | Something in `transform` depends on other rows | Send me the failure. This is the guard working. |
| `ruff check .` reports dozens of errors | First run on an unlinted codebase | `ruff check . --fix`, review, re-run the tests. |
| `black --check` fails in CI but passes locally | Formatted locally but not committed | `git add -A` after running black. |
| `pre-commit autoupdate` fails | No internet, or a rev tag does not exist | The versions given are known-good. Skip autoupdate if it fails. |
| Monitoring: `FileNotFoundError` on the metadata | Training has not run since Step 4 | `python run.py --step training` |
| Monitoring is slow the first time | It is training the selection model | Expected, about a minute, saved for next time. |
| PSI is `nan` for many features | Too few distinct values to bucket | Expected on binary and near-constant columns. Not an error. |
| Promotion: `run_mode tag missing` on version 2 | The run predates the tag | The `set_tag` command in Section 17.1. |
| Promotion: `feature list matches` fails | The metadata describes a different version | Only the most recent training run has a metadata file. Re-run training, or promote the version it describes. |
| CI fails installing scipy or pyarrow | Network hiccup on the runner | Re-run the job from the Actions page. |
| `check-added-large-files` blocks a commit | A Parquet or model file got staged | `git reset HEAD <file>`. The hook is doing its job. |

---

## 24. What to send me before Step 6

1. **The full terminal output** of `python run.py --step monitoring`
2. **`reports/monitoring/drift_summary.md`** contents
3. **`reports/monitoring/feature_drift.csv`** and **`score_drift.csv`** as attachments
4. **`reports/monitoring/period_metrics.csv`** as an attachment
5. **The output of both promotion commands**, version 2 and version 1
6. **The pytest output**, including how many tests ran
7. **Whether CI went green**, and the log if it did not
8. **Q-15:** do you have a Hugging Face account? It is free at huggingface.co. Step 6 needs one, plus a token from Settings, Access Tokens, with write permission.
9. **Whether you re-ran CatBoost** with a larger budget, and what it scored

---

## 25. What Step 6 covers

- A Dockerfile that builds the service, and why the layer order matters for build speed
- A FastAPI service with `/health`, `/predict`, `/predict/batch`, and the automatic `/docs` page
- Loading the transformer and the model from the registry alias, so deploying means moving a pointer rather than editing code
- Request validation with Pydantic: what a transaction must contain and what happens when it does not
- The single-row scoring path, which the row-independence test already proved is safe
- `docker compose` for running the service and MLflow together locally
- Publishing the artifacts to the Hugging Face Model Hub, so the container downloads them at startup rather than baking a 33 MB payload into the image
- Deploying to Hugging Face Spaces with the Docker SDK, per D-44
- Extending CI to build the image on every push, so a broken Dockerfile is caught before deployment
- A response time budget, and what to do when a single prediction takes too long

---

*End of Step 5. `PROJECT_STATE.md` follows as a separate document.*
