# EDA Summary: IEEE-CIS Fraud Detection

Generated automatically by `src/pipelines/eda.py`. Do not edit by hand, it is overwritten on every run.

## 1. Dataset shape

- Joined training table: **590,540 rows x 435 columns**
- In-memory size after type optimisation: **927.2 MB**

## 2. Class balance

- Total transactions: **590,540**
- Fraudulent: **20,663**
- Legitimate: **569,877**
- Fraud rate: **3.4990%**
- Roughly 1 fraud per **28** legitimate transactions

A model that predicted "never fraud" would score **96.50% accuracy** while being useless. Accuracy is not used as a metric on this project.

## 3. Identity coverage

| Group | Transactions | Fraud rate |
|-------|--------------|------------|
| No identity record | 446,307 | 2.0939% |
| Has identity record | 144,233 | 7.8470% |

Fraud is **3.75x** as likely among transactions that have an identity record. Read that figure carefully. The table below shows that identity coverage is almost entirely decided by `ProductCD`: product W never has an identity record, and every other product almost always does. Since W also has the lowest fraud rate and makes up most of the data, the bulk of this gap is a product effect rather than an identity effect. Restricted to the non-W products, where the flag actually varies, the lift is closer to 1.4x. `has_identity` is kept as a feature, but it is expected to rank low.

Identity coverage by product code, as a percentage of each product's transactions:

| ProductCD   |   no_identity_share |   has_identity_share |
|:------------|--------------------:|---------------------:|
| C           |                 9.2 |                 90.8 |
| H           |                 0.4 |                 99.6 |
| R           |                 0.4 |                 99.6 |
| S           |                 0.4 |                 99.6 |
| W           |               100   |                  0   |

## 4. Time coverage

| Split | First | Last | Span (days) |
|-------|-------|------|-------------|
| train | 2017-12-01 | 2018-05-31 | 182.0 |
| test | 2018-07-01 | 2018-12-30 | 183.0 |

There is a gap of **30.0 days** between the last training transaction and the first test transaction. The test set is entirely in the future relative to training.

**Consequence:** validation must be a time-based split, never a random one. A random split would let the model learn from transactions that happened after the ones it is validated on, producing a validation score that cannot be reproduced in production.

## 5. Feature families

| family      |   columns |   mean_missing_pct |   max_missing_pct |
|:------------|----------:|-------------------:|------------------:|
| vesta_V     |       339 |              43.04 |             86.12 |
| identity_id |        38 |              84.82 |             99.2  |
| timedelta_D |        15 |              58.15 |             93.41 |
| counting_C  |        14 |               0    |              0    |
| match_M     |         9 |              49.92 |             59.35 |
| card        |         6 |               0.51 |              1.51 |
| address     |         2 |              11.13 |             11.13 |
| device      |         2 |              78.04 |             79.91 |
| distance    |         2 |              76.64 |             93.63 |
| email       |         2 |              46.37 |             76.75 |
| engineered  |         1 |               0    |              0    |
| amount      |         1 |               0    |              0    |
| product     |         1 |               0    |              0    |
| identifier  |         1 |               0    |              0    |
| time        |         1 |               0    |              0    |
| target      |         1 |               0    |              0    |

## 6. Missing data

- Columns with no missing values at all: **53**
- Columns missing more than 90% of their values: **12**

The 25 emptiest columns:

| column   | family      | dtype    |   missing_count |   missing_pct |
|:---------|:------------|:---------|----------------:|--------------:|
| id_24    | identity_id | float32  |          585793 |         99.2  |
| id_08    | identity_id | float32  |          585385 |         99.13 |
| id_21    | identity_id | float32  |          585381 |         99.13 |
| id_07    | identity_id | float32  |          585385 |         99.13 |
| id_26    | identity_id | float32  |          585377 |         99.13 |
| id_25    | identity_id | float32  |          585408 |         99.13 |
| id_22    | identity_id | float32  |          585371 |         99.12 |
| id_27    | identity_id | category |          585371 |         99.12 |
| id_23    | identity_id | category |          585371 |         99.12 |
| dist2    | distance    | float32  |          552913 |         93.63 |
| D7       | timedelta_D | float32  |          551623 |         93.41 |
| id_18    | identity_id | float32  |          545427 |         92.36 |
| D13      | timedelta_D | float32  |          528588 |         89.51 |
| D14      | timedelta_D | float32  |          528353 |         89.47 |
| D12      | timedelta_D | float32  |          525823 |         89.04 |
| id_04    | identity_id | float32  |          524216 |         88.77 |
| id_03    | identity_id | float32  |          524216 |         88.77 |
| D6       | timedelta_D | float32  |          517353 |         87.61 |
| id_33    | identity_id | category |          517251 |         87.59 |
| id_09    | identity_id | float32  |          515614 |         87.31 |
| id_10    | identity_id | float32  |          515614 |         87.31 |
| D9       | timedelta_D | float32  |          515614 |         87.31 |
| D8       | timedelta_D | float32  |          515614 |         87.31 |
| id_30    | identity_id | category |          512975 |         86.87 |
| id_32    | identity_id | float32  |          512954 |         86.86 |

## 7. V column structure

The 339 V columns fall into **15 blocks** that share an identical missing value pattern.

Vesta engineered these features in batches from shared source data. When a source was unavailable for a transaction, every feature derived from it went blank together. Columns inside one block are therefore usually closely related, which gives Step 3 a principled way to reduce 339 columns to a manageable number: keep a representative from each block instead of dropping columns arbitrarily.

The ten largest blocks:

|   group_id |   n_columns |   missing_pct |
|-----------:|------------:|--------------:|
|          1 |          46 |         77.91 |
|          2 |          43 |          0.05 |
|          3 |          32 |          0    |
|          4 |          31 |         76.36 |
|          5 |          23 |         12.88 |
|          6 |          22 |         13.06 |
|          7 |          20 |         15.1  |
|          8 |          19 |         76.32 |
|          9 |          18 |         28.61 |
|         10 |          18 |         86.12 |

Full detail in `reports/v_column_missing_groups.csv`.

## 8. Fraud rate by key categorical columns

### Product code (ProductCD)

| category   |   transactions |   frauds | fraud_rate   |
|:-----------|---------------:|---------:|:-------------|
| C          |          68519 |     8008 | 11.69%       |
| S          |          11628 |      686 | 5.90%        |
| H          |          33024 |     1574 | 4.77%        |
| R          |          37699 |     1426 | 3.78%        |
| W          |         439670 |     8969 | 2.04%        |

### Card network (card4)

| category         |   transactions |   frauds | fraud_rate   |
|:-----------------|---------------:|---------:|:-------------|
| discover         |           6651 |      514 | 7.73%        |
| visa             |         384767 |    13373 | 3.48%        |
| mastercard       |         189217 |     6496 | 3.43%        |
| american express |           8328 |      239 | 2.87%        |
| (missing)        |           1577 |       41 | 2.60%        |

### Card type (card6)

| category   |   transactions |   frauds | fraud_rate   |
|:-----------|---------------:|---------:|:-------------|
| credit     |         148986 |     9950 | 6.68%        |
| (missing)  |           1571 |       39 | 2.48%        |
| debit      |         439938 |    10674 | 2.43%        |

### Device type (DeviceType)

| category   |   transactions |   frauds | fraud_rate   |
|:-----------|---------------:|---------:|:-------------|
| mobile     |          55645 |     5657 | 10.17%       |
| desktop    |          85165 |     5554 | 6.52%        |
| (missing)  |         449730 |     9452 | 2.10%        |

### Purchaser email domain, top 15 by fraud rate

| category      |   transactions |   frauds | fraud_rate   |
|:--------------|---------------:|---------:|:-------------|
| mail.com      |            559 |      106 | 18.96%       |
| outlook.com   |           5096 |      482 | 9.46%        |
| live.com.mx   |            749 |       41 | 5.47%        |
| hotmail.com   |          45250 |     2396 | 5.30%        |
| gmail.com     |         228355 |     9943 | 4.35%        |
| icloud.com    |           6267 |      197 | 3.14%        |
| comcast.net   |           7888 |      246 | 3.12%        |
| charter.net   |            816 |       25 | 3.06%        |
| (missing)     |          94456 |     2790 | 2.95%        |
| bellsouth.net |           1909 |       53 | 2.78%        |
| live.com      |           3041 |       84 | 2.76%        |
| anonymous.com |          36998 |      859 | 2.32%        |
| yahoo.com     |         100934 |     2297 | 2.28%        |
| msn.com       |           4092 |       90 | 2.20%        |
| aol.com       |          28289 |      617 | 2.18%        |

## 9. Decisions carried into Step 3

1. **Primary metric is PR-AUC.** ROC-AUC is reported alongside it, since it was the competition metric. Accuracy is not used.
2. **Validation is time-based.** The last 20% of the training period by `TransactionDT` becomes the validation set. No random shuffling.
3. **Missing values stay missing.** LightGBM, XGBoost, and CatBoost all learn a direction for missing values at each split. Filling blanks with an average would assert something false.
4. **`has_identity` is kept** as an explicit feature.
5. **V columns are reduced using the block structure** identified above, rather than by an arbitrary correlation cutoff.
