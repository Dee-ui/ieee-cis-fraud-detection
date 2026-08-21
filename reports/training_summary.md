# Model Training Summary

Generated automatically by `src/pipelines/training.py`. Do not edit by hand, it is overwritten on every run.

## 1. Candidate comparison

| model               |   pr_auc |   pr_auc_lift |   roc_auc |   best_round |   fit_minutes |
|:--------------------|---------:|--------------:|----------:|-------------:|--------------:|
| lightgbm            |  0.60682 |      17.6353  |   0.92751 |          617 |          0.71 |
| xgboost             |  0.59907 |      17.4103  |   0.93079 |         1193 |          4.35 |
| catboost            |  0.52819 |      15.3502  |   0.89368 |         1500 |          7.14 |
| logistic_regression |  0.18309 |       5.32094 |   0.82095 |          nan |          1.07 |
| dummy               |  0.03441 |       1       |   0.5     |          nan |          0.03 |

Winner: **lightgbm**, validation PR-AUC **0.60682**.

## 2. The uid ablation

Six uid features are blank on about 82% of test rows, so the winner was retrained without them. The decision rule was fixed in advance: drop them if the cost is under 0.005 PR-AUC.

| Model | Validation PR-AUC |
|-------|-------------------|
| with uid features | 0.60682 |
| without uid features | 0.59393 |
| difference | +0.01289 |

**Decision: kept.** Final feature count 284.

## 3. Stability across time

|   fold |   train_rows |   valid_rows | valid_start   | valid_end   |   pr_auc |   roc_auc |
|-------:|-------------:|-------------:|:--------------|:------------|---------:|----------:|
|      1 |       118108 |       118108 | 2017-12-26    | 2018-02-02  |  0.61833 |   0.90256 |
|      2 |       236216 |       118108 | 2018-02-02    | 2018-03-11  |  0.63763 |   0.91974 |
|      3 |       354324 |       118108 | 2018-03-11    | 2018-04-20  |  0.67082 |   0.9413  |
|      4 |       472432 |       118108 | 2018-04-20    | 2018-05-31  |  0.60682 |   0.92751 |

Mean PR-AUC **0.63340**, spread **0.02800**. Each fold trains on more history than the last and is scored on the period straight after, which is the same shape as the real problem.

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
| Cheapest overall | 18.86% | 86.0% | $444,996 |
| Cheapest within capacity | 2.00% | 44.6% | $202,033 |

**Annualised, at the within-capacity operating point: $1,760,894 a year.**

The chosen threshold is **0.4222**.

Recall and cost at each headline review rate:

|   review_rate |   n_reviewed |   threshold |   recall |   precision |   savings |
|--------------:|-------------:|------------:|---------:|------------:|----------:|
|         0.005 |          591 |     0.95653 |  0.13755 |     0.94585 |   57413.8 |
|         0.01  |         1181 |     0.83433 |  0.26599 |     0.91533 |  114501   |
|         0.02  |         2362 |     0.42142 |  0.44587 |     0.76715 |  202013   |
|         0.05  |         5905 |     0.09626 |  0.64296 |     0.44251 |  339362   |

## 5. What drives the model

| feature                            |   mean_abs_shap |
|:-----------------------------------|----------------:|
| C13                                |         0.28983 |
| C14                                |         0.13333 |
| TransactionAmt_ratio_to_addr1_mean |         0.12076 |
| C1                                 |         0.11747 |
| V70                                |         0.11473 |
| D15_std_by_uid                     |         0.10702 |
| D15_mean_by_uid                    |         0.10418 |
| uid_freq                           |         0.10305 |
| card1_freq                         |         0.09954 |
| TransactionAmt_mean_by_card1       |         0.09683 |
| card6_code                         |         0.09609 |
| V91                                |         0.09234 |
| M5_code                            |         0.09101 |
| TransactionAmt_mean_by_uid         |         0.08845 |
| C11                                |         0.0856  |
| D1                                 |         0.08501 |
| D2                                 |         0.08174 |
| dist1                              |         0.07479 |
| card2_freq                         |         0.0742  |
| C2                                 |         0.0741  |

Charts in `reports/explainability/`.

## 6. Carried into Step 5

1. Registered model `ieee-cis-fraud-detector` version 2, alias `candidate`.
2. MLflow run id `68850ae7c1264e80ba87229fa54ed899`.
3. Operating threshold 0.4222, chosen by cost within review capacity, not left at 0.5.
4. Watch the uid family in drift monitoring, whether or not it was dropped. It was the clearest train-to-test shift in the data.
5. `models/final_model_metadata.json` holds the exact feature list the service must supply.
