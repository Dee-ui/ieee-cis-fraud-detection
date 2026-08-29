# Monitoring Summary

Generated automatically by `src/pipelines/monitoring.py`. Do not edit by hand, it is overwritten on every run.

## Verdict: WATCH

- Worst importance-weighted PSI across all periods: **0.1247**
- Retrain threshold: 0.15
- Operating threshold: 0.4222, expected alert rate 2.00%

## 1. Performance on labelled held-out data

| period                |   rows |   frauds |   fraud_rate |   pr_auc |   pr_auc_lift |   roc_auc | is_full_week   |
|:----------------------|-------:|---------:|-------------:|---------:|--------------:|----------:|:---------------|
| 2018-04-16/2018-04-22 |   7029 |      296 |       0.0421 |   0.7059 |       16.7626 |    0.9458 | False          |
| 2018-04-23/2018-04-29 |  18652 |      555 |       0.0298 |   0.5931 |       19.9333 |    0.9294 | True           |
| 2018-04-30/2018-05-06 |  22071 |      681 |       0.0309 |   0.6819 |       22.1007 |    0.9386 | True           |
| 2018-05-07/2018-05-13 |  20726 |      650 |       0.0314 |   0.5689 |       18.1403 |    0.9137 | True           |
| 2018-05-14/2018-05-20 |  20332 |      717 |       0.0353 |   0.5414 |       15.3537 |    0.916  | True           |
| 2018-05-21/2018-05-27 |  19010 |      760 |       0.04   |   0.6497 |       16.2522 |    0.9318 | True           |
| 2018-05-28/2018-06-03 |  10288 |      405 |       0.0394 |   0.5461 |       13.8715 |    0.9227 | False          |

`pr_auc_lift` is the PR-AUC divided by that week's own fraud rate. It is the column to read for a trend. Raw PR-AUC sits on a floor equal to the fraud rate, and that rate moves week to week, so raw scores from different weeks are not directly comparable.

These weeks are the last labelled data the model never trained on. This is the only honest performance measurement available, because the test period has no labels at all. In production you would be in the same position: weeks or months of scoring before you learn whether the scores were any good.

## 2. Data drift, month by month

| period   |   rows |   alert_rate |   weighted_psi |   features_significant |   top_features_significant |
|:---------|-------:|-------------:|---------------:|-----------------------:|---------------------------:|
| 2018-07  |  78430 |       0.0283 |         0.0816 |                      7 |                          1 |
| 2018-08  |  77094 |       0.0237 |         0.0896 |                      9 |                          1 |
| 2018-09  |  71288 |       0.0237 |         0.0905 |                      7 |                          1 |
| 2018-10  |  80677 |       0.0199 |         0.0955 |                      7 |                          1 |
| 2018-11  |  82804 |       0.0162 |         0.0922 |                      6 |                          1 |
| 2018-12  | 116398 |       0.016  |         0.1247 |                     13 |                          1 |

`weighted_psi` weights each feature's drift by how much the model actually relies on it. With 284 features a few will always have drifted, and drift in a feature the model ignores is not a problem.

## 3. The features that moved most

| period   | feature               |    psi | band        |   rows_current |   missing_reference |   missing_current |
|:---------|:----------------------|-------:|:------------|---------------:|--------------------:|------------------:|
| 2018-11  | D15_ratio_to_uid_mean | 4.1352 | significant |          10291 |              0.3403 |            0.8757 |
| 2018-12  | D15_ratio_to_uid_mean | 3.9773 | significant |          10435 |              0.3403 |            0.9104 |
| 2018-09  | D15_ratio_to_uid_mean | 3.4915 | significant |          12816 |              0.3403 |            0.8202 |
| 2018-10  | D15_ratio_to_uid_mean | 3.4774 | significant |          12213 |              0.3403 |            0.8486 |
| 2018-08  | D15_ratio_to_uid_mean | 3.2968 | significant |          16135 |              0.3403 |            0.7907 |
| 2018-07  | D15_ratio_to_uid_mean | 3.0883 | significant |          19123 |              0.3403 |            0.7562 |
| 2018-10  | id_13                 | 3.0284 | significant |          19812 |              0.7774 |            0.7544 |
| 2018-11  | id_13                 | 2.8667 | significant |          21216 |              0.7774 |            0.7438 |
| 2018-09  | id_13                 | 2.8101 | significant |          12883 |              0.7774 |            0.8193 |
| 2018-07  | id_13                 | 2.7603 | significant |          13693 |              0.7774 |            0.8254 |
| 2018-12  | id_13                 | 2.6953 | significant |          49668 |              0.7774 |            0.5733 |
| 2018-08  | id_13                 | 2.5982 | significant |          13014 |              0.7774 |            0.8312 |
| 2018-12  | uid_freq              | 2.4622 | significant |         116398 |              0      |            0      |
| 2018-11  | uid_freq              | 2.1611 | significant |          82804 |              0      |            0      |
| 2018-10  | uid_freq              | 1.9328 | significant |          80677 |              0      |            0      |
| 2018-09  | uid_freq              | 1.7306 | significant |          71288 |              0      |            0      |
| 2018-08  | uid_freq              | 1.5439 | significant |          77094 |              0      |            0      |
| 2018-07  | uid_freq              | 1.3536 | significant |          78430 |              0      |            0      |
| 2018-10  | V160                  | 1.2139 | significant |           7158 |              0.8525 |            0.9113 |
| 2018-08  | V160                  | 1.1871 | significant |           7276 |              0.8525 |            0.9056 |

46 feature-period combinations were measured on fewer than 5,000 usable values and are excluded from this table. A PSI computed on a few hundred rows swings wildly for reasons that have nothing to do with drift. They are still in `feature_drift.csv`, flagged in the `low_confidence` column.
## 4. Alert volume

The threshold is fixed, so any change in alert volume comes entirely from the data moving. This is the number an operations manager feels directly, because it is how much work lands in the review queue.

| period   |   alert_rate |   alert_rate_ratio |
|:---------|-------------:|-------------------:|
| 2018-07  |       0.0283 |             1.4165 |
| 2018-08  |       0.0237 |             1.1851 |
| 2018-09  |       0.0237 |             1.1853 |
| 2018-10  |       0.0199 |             0.9965 |
| 2018-11  |       0.0162 |             0.81   |
| 2018-12  |       0.016  |             0.8013 |

## 5. What happens next

Drift is present but below the retraining threshold. Keep monitoring, and look at whether the trend is worsening month on month or holding steady.
