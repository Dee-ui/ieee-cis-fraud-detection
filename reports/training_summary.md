# Model Training Summary

Generated automatically by `src/pipelines/training.py`. Do not edit by hand, it is overwritten on every run.

## 1. Candidate comparison

| model    |   pr_auc |   pr_auc_lift |   roc_auc |   best_round |   fit_minutes |
|:---------|---------:|--------------:|----------:|-------------:|--------------:|
| catboost |   0.5291 |       15.3768 |   0.89397 |         1532 |          5.98 |

Winner: **catboost**, validation PR-AUC **0.52910**.

## 2. The uid ablation

Six uid features are blank on about 82% of test rows, so the winner was retrained without them. The decision rule was fixed in advance: drop them if the cost is under 0.005 PR-AUC.

| Model | Validation PR-AUC |
|-------|-------------------|
| with uid features | 0.52910 |
| without uid features | 0.57198 |
| difference | -0.04287 |

**Decision: dropped.** Final feature count 277.

## 3. Stability across time

|   fold |   train_rows |   valid_rows | valid_start   | valid_end   |   pr_auc |   roc_auc |
|-------:|-------------:|-------------:|:--------------|:------------|---------:|----------:|
|      1 |       118108 |       118108 | 2017-12-26    | 2018-02-02  |  0.58515 |   0.87518 |
|      2 |       236216 |       118108 | 2018-02-02    | 2018-03-11  |  0.57329 |   0.89305 |
|      3 |       354324 |       118108 | 2018-03-11    | 2018-04-20  |  0.62487 |   0.92152 |
|      4 |       472432 |       118108 | 2018-04-20    | 2018-05-31  |  0.57198 |   0.91259 |

Mean PR-AUC **0.58882**, spread **0.02475**. Each fold trains on more history than the last and is scored on the period straight after, which is the same shape as the real problem.

## 4. What it is worth

Costs use the assumptions in `config/config.py`. They are stated assumptions, not figures supplied by a business. See step4.md section 3.

| Assumption | Value |
|------------|-------|
| Analyst review | $4.00 per case |
| Chargeback fee | $25.00 per missed fraud |
| False alarm friction | $1.00 |
| Fraud recovered when caught | 90% |
| Review capacity | 2% of transactions |

Over the 42 day validation period, doing nothing costs **$711,534** in fraud losses and chargeback fees.

| Operating point | Review rate | Recall | Savings |
|-----------------|-------------|--------|---------|
| Cheapest overall | 16.44% | 82.1% | $438,616 |
| Cheapest within capacity | 2.00% | 43.4% | $218,263 |

**Annualised, at the within-capacity operating point: $1,902,351 a year.**

The chosen threshold is **0.3609**.

Recall and cost at each headline review rate:

|   review_rate |   n_reviewed |   threshold |   recall |   precision |   savings |
|--------------:|-------------:|------------:|---------:|------------:|----------:|
|         0.005 |          591 |     0.97677 |  0.13484 |     0.92724 |     55708 |
|         0.01  |         1181 |     0.81392 |  0.25886 |     0.89077 |    117967 |
|         0.02  |         2362 |     0.35941 |  0.43406 |     0.74682 |    218238 |
|         0.05  |         5905 |     0.08856 |  0.61983 |     0.42659 |    345719 |

## 5. What drives the model

| feature                            |   mean_abs_shap |
|:-----------------------------------|----------------:|
| C13                                |         0.34196 |
| C1                                 |         0.24257 |
| C14                                |         0.18802 |
| M4_code                            |         0.18033 |
| TransactionAmt_mean_by_card1       |         0.14448 |
| card6_code                         |         0.14097 |
| C2                                 |         0.12233 |
| C11                                |         0.11811 |
| TransactionAmt_ratio_to_addr1_mean |         0.11045 |
| card1_freq                         |         0.10996 |
| D1                                 |         0.09953 |
| M5_code                            |         0.0978  |
| M6_code                            |         0.09463 |
| V308                               |         0.09117 |
| card1_addr1_freq                   |         0.08926 |
| D15_mean_by_card1                  |         0.08764 |
| C8                                 |         0.08359 |
| card2_freq                         |         0.08343 |
| ProductCD_code                     |         0.07525 |
| D4                                 |         0.07436 |

Charts in `reports/explainability/`.

## 6. Carried into Step 5

1. Registered model `ieee-cis-fraud-detector` version 3, alias `candidate`.
2. MLflow run id `45253e4849de42b5bd70dc27741bc138`.
3. Operating threshold 0.3609, chosen by cost within review capacity, not left at 0.5.
4. Watch the uid family in drift monitoring, whether or not it was dropped. It was the clearest train-to-test shift in the data.
5. `models/final_model_metadata.json` holds the exact feature list the service must supply.
