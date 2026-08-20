# Feature Engineering Summary

Generated automatically by `src/pipelines/features.py`. Do not edit by hand, it is overwritten on every run.

## 1. Column reduction

| Stage | Columns |
|-------|---------|
| Joined training table | 435 |
| Dropped: single value | 0 |
| Dropped: near-constant | 2 |
| Rescued from near-constant | 22 |
| V columns before reduction | 337 |
| V columns after reduction | 137 |
| **Final feature count** | **284** |

Every dropped column, with the evidence behind the decision, is in `reports/dropped_columns.csv`. The V column mapping is in `reports/v_column_reduction.csv`.

## 2. Feature types

| kind           |   features |
|:---------------|-----------:|
| base_numeric   |        199 |
| category_code  |         38 |
| aggregate      |         18 |
| frequency      |         18 |
| derived_amount |          3 |
| derived_screen |          3 |
| derived_match  |          2 |
| derived_time   |          2 |
| derived_email  |          1 |

## 3. The time split

| Portion | Rows | Frauds | Fraud rate | First | Last |
|---------|------|--------|------------|-------|------|
| train | 472,432 | 16,599 | 3.5135% | 2017-12-01 | 2018-04-20 |
| valid | 118,108 | 4,064 | 3.4409% | 2018-04-20 | 2018-05-31 |

The boundary sits at TransactionDT 12,192,854, which is 2018-04-20.

The transformer was fitted on the `train` portion only. The `valid` portion and the test set were transformed using what was learned there, and contributed nothing to it. Any frequency count or group average attached to a validation row was computed without that row.

## 4. Test set

- Rows: **506,691**
- Features: **284**, identical to training and in the same order
- Values never seen during training, across all counted columns: **6.81%** of lookups returned zero

## 5. Verification

- The target is not present in the feature table
- Training and test features match exactly, in name and order
- No feature column is blank on every row
- No feature column contains infinity

## 6. Carried into Step 4

1. Read the `split` column rather than recomputing the split, so every experiment is scored on exactly the same rows.
2. `TransactionID` and `TransactionDT` are present in the files but are not features. Drop them before training.
3. Load `models/feature_engineer.joblib` for scoring, never rebuild the transformations by hand.
4. PR-AUC is primary, baseline 0.035. ROC-AUC secondary. Recall at a 1% review rate is the business headline.
