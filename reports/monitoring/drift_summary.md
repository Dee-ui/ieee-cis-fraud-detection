# Monitoring Summary

Generated automatically by `src/pipelines/monitoring.py`. Do not edit by hand, it is overwritten on every run.

## Verdict: WATCH

- Worst importance-weighted PSI across all periods: **0.0668**
- Retrain threshold: 0.15
- Operating threshold: 0.3609, expected alert rate 2.00%

## 1. Performance on labelled held-out data

| period                |   rows |   frauds |   fraud_rate |   pr_auc |   roc_auc |
|:----------------------|-------:|---------:|-------------:|---------:|----------:|
| 2018-04-16/2018-04-22 |   7029 |      296 |      0.04211 |  0.66972 |   0.92124 |
| 2018-04-23/2018-04-29 |  18652 |      555 |      0.02976 |  0.53673 |   0.90407 |
| 2018-04-30/2018-05-06 |  22071 |      681 |      0.03085 |  0.64035 |   0.92004 |
| 2018-05-07/2018-05-13 |  20726 |      650 |      0.03136 |  0.5517  |   0.90863 |
| 2018-05-14/2018-05-20 |  20332 |      717 |      0.03526 |  0.49947 |   0.90045 |
| 2018-05-21/2018-05-27 |  19010 |      760 |      0.03998 |  0.6275  |   0.92359 |
| 2018-05-28/2018-06-03 |  10288 |      405 |      0.03937 |  0.50684 |   0.91    |

These weeks are the last labelled data the model never trained on. This is the only honest performance measurement available, because the test period has no labels at all. In production you would be in the same position: weeks or months of scoring before you learn whether the scores were any good.

## 2. Data drift, month by month

| period   |   rows |   alert_rate |   weighted_psi |   features_significant |   top_features_significant |
|:---------|-------:|-------------:|---------------:|-----------------------:|---------------------------:|
| 2018-07  |  78430 |       0.0292 |         0.0386 |                      4 |                          0 |
| 2018-08  |  77094 |       0.0257 |         0.0434 |                      6 |                          0 |
| 2018-09  |  71288 |       0.0246 |         0.0385 |                      4 |                          0 |
| 2018-10  |  80677 |       0.0202 |         0.0419 |                      6 |                          0 |
| 2018-11  |  82804 |       0.0167 |         0.0319 |                      4 |                          0 |
| 2018-12  | 116398 |       0.02   |         0.0668 |                     10 |                          0 |

`weighted_psi` weights each feature's drift by how much the model actually relies on it. With 284 features a few will always have drifted, and drift in a feature the model ignores is not a problem.

## 3. The features that moved most

| period   | feature    |    psi | band        |   missing_reference |   missing_current |
|:---------|:-----------|-------:|:------------|--------------------:|------------------:|
| 2018-11  | id_21      | 7.1474 | significant |              0.991  |            0.9913 |
| 2018-12  | id_21      | 3.7989 | significant |              0.991  |            0.9814 |
| 2018-10  | id_13      | 3.0284 | significant |              0.7774 |            0.7544 |
| 2018-11  | id_13      | 2.8667 | significant |              0.7774 |            0.7438 |
| 2018-09  | id_13      | 2.8101 | significant |              0.7774 |            0.8193 |
| 2018-07  | id_13      | 2.7603 | significant |              0.7774 |            0.8254 |
| 2018-12  | id_13      | 2.6953 | significant |              0.7774 |            0.5733 |
| 2018-08  | id_13      | 2.5982 | significant |              0.7774 |            0.8312 |
| 2018-10  | V160       | 1.2139 | significant |              0.8525 |            0.9113 |
| 2018-08  | V160       | 1.1871 | significant |              0.8525 |            0.9056 |
| 2018-07  | V160       | 1.183  | significant |              0.8525 |            0.9118 |
| 2018-09  | V160       | 1.1752 | significant |              0.8525 |            0.9052 |
| 2018-12  | id_31_freq | 0.6906 | significant |              0      |            0      |
| 2018-12  | V60        | 0.5839 | significant |              0.1416 |            0.0008 |
| 2018-12  | V17        | 0.5701 | significant |              0.1406 |            0.003  |
| 2018-12  | V80        | 0.5424 | significant |              0.1647 |            0.0009 |
| 2018-12  | V40        | 0.5409 | significant |              0.2959 |            0.1047 |
| 2018-10  | id_21      | 0.5168 | significant |              0.991  |            0.9928 |
| 2018-10  | id_31_code | 0.4914 | significant |              0      |            0      |
| 2018-11  | id_31_code | 0.4801 | significant |              0      |            0      |

## 4. Alert volume

The threshold is fixed, so any change in alert volume comes entirely from the data moving. This is the number an operations manager feels directly, because it is how much work lands in the review queue.

| period   |   alert_rate |   alert_rate_ratio |
|:---------|-------------:|-------------------:|
| 2018-07  |       0.0292 |             1.4657 |
| 2018-08  |       0.0257 |             1.2863 |
| 2018-09  |       0.0246 |             1.2322 |
| 2018-10  |       0.0202 |             1.0099 |
| 2018-11  |       0.0167 |             0.8345 |
| 2018-12  |       0.02   |             1.0018 |

## 5. What happens next

Drift is present but below the retraining threshold. Keep monitoring, and look at whether the trend is worsening month on month or holding steady.
