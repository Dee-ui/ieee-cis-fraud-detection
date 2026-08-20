# Step 3: Feature Engineering and Preprocessing
### Column pruning, V-block reduction, engineered features, the time split, and a saved transformer

**Project:** IEEE-CIS Fraud Detection
**Repository:** https://github.com/Dee-ui/ieee-cis-fraud-detection
**Local path:** `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`
**Platform:** Windows, VS Code, PowerShell, Python 3.11.9
**Estimated time:** 3 to 4 hours, of which about 15 minutes is the machine running
**Step 3 of 7**

---

## 0. How to use this document

Same as before. Work top to bottom, do not skip.

Sections 1 and 2 review what your Step 2 run actually produced, including three places where I need to correct something I told you earlier. Section 5 explains the concepts before any code appears. Sections 6 to 11 are the code. Section 12 runs it.

Blocks labelled `powershell` go in the VS Code terminal. Blocks labelled `python` are file contents you create and paste.

Section 18 has the checklist. Do not start Step 4 until every box ticks.

---

## 1. Where Step 2 left you

Everything below is now confirmed by your own run.

**Housekeeping resolved.** The project moved to `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`, outside OneDrive, and the folder name now matches the repository. `.venv` was rebuilt from `requirements.lock.txt`. Verification still passes from the new location. That closes open questions Q-08 and Q-09.

**Python confirmed as 3.11.9.** That closes Q-10.

**Ingestion output:**

| Split | Rows | Columns | Memory before | Memory after | Reduction | Parquet |
|-------|------|---------|---------------|--------------|-----------|---------|
| train | 590,540 | 435 | 2,567.7 MB | 927.2 MB | 63.9% | 80.3 MB |
| test | 506,691 | 434 | 2,214.5 MB | 795.2 MB | 64.1% | 69.8 MB |

Both shapes matched their expected values exactly. Train renamed zero identity columns, test renamed 38. Total runtime 3 minutes 7 seconds.

**EDA output:** 15 V blocks, fraud rate 3.4990% matching Step 1 exactly, train 2017-12-01 to 2018-05-31, test 2018-07-01 to 2018-12-30, a 30 day gap between them, 12 columns above 90% missing, and no unmapped column warning. Total runtime 28 seconds.

---

## 2. Reading your Step 2 results properly

This is the analysis section. Some of it corrects things I said earlier, which I would rather do openly than quietly paper over.

### 2.1 Correction one: I was wrong about the Parquet file sizes

I predicted 250 to 400 MB per file. You got 80.3 MB and 69.8 MB, roughly four times smaller.

Why I was wrong: I estimated from the in-memory size and applied a typical compression ratio. But this table is unusually compressible for two reasons that stack on each other. First, 398 of the 435 columns are `float32` and a large share of their values are blank, and Parquet stores blanks as a separate compact bitmap rather than as data. Second, Parquet's snappy compression works down each column, and columns here hold long runs of repeated values, which compress extremely well.

The practical consequence is good news: reading these files is fast, and shipping them around later in Step 6 costs almost nothing.

### 2.2 Correction two: the memory reduction landed slightly below my range

I said 65 to 75%. You got 63.9% and 64.1%.

That is not a problem, and the arithmetic explains it. The dominant saving is converting 398 columns from `float64` to `float32`, which is exactly 50% on those columns and no more. The rest of the saving comes from the 31 text columns becoming categories. Because this table is overwhelmingly numeric rather than text, the total lands close to 64% rather than higher. My range was optimistic, not your run being wrong.

### 2.3 Correction three, and this one matters: the 3.75x identity finding is mostly a product effect

Your EDA report says fraud is 3.75 times as likely among transactions with an identity record: 7.8470% against 2.0939%.

That number is real, but it is misleading, and the cross-tabulation I built into Step 2 is exactly what exposes it. Look at it again:

| ProductCD | No identity | Has identity |
|-----------|-------------|--------------|
| C | 9.2% | 90.8% |
| H | 0.4% | 99.6% |
| R | 0.4% | 99.6% |
| S | 0.4% | 99.6% |
| W | 100% | 0% |

Product W **never** has an identity record. Every other product almost always does. So "has an identity record" is very nearly the same statement as "is not product W".

And product W has the lowest fraud rate of all five products, at 2.04%, while making up 439,670 of the 590,540 transactions. So the 3.75x gap is mostly measuring the difference between W and everything else, not the effect of an identity record.

Working it out from your own numbers, restricted to the non-W products where the flag actually varies:

| Group, non-W products only | Transactions | Frauds | Fraud rate |
|----------------------------|--------------|--------|------------|
| Has identity record | 144,233 | 11,318 | 7.85% |
| No identity record | 6,637 | 376 | 5.67% |

That is a lift of **1.39x**, not 3.75x.

**What this means for us.** The `has_identity` flag does carry a little independent signal, but far less than the headline suggested. We keep it, because it is one column and costs nothing, but you should expect it to rank low when we look at feature importance in Step 4. If it ranked high, that would be a warning sign that the model has simply found `ProductCD` by another route.

Section 14 has a small patch to the Step 2 report generator so the auto-generated text carries this caveat from now on.

This is worth internalising for the PM track, because it is the single most common analytical mistake in this kind of work: a difference between two groups that is really a difference in what those groups are made of. The technical name is a confounded comparison. The habit that catches it is to always ask what else differs between the two groups.

### 2.4 What the V blocks actually look like

Your run found 15 blocks covering all 339 V columns. Here they are, sorted by size:

| Block | Columns | Missing | Number range | Shape |
|-------|---------|---------|--------------|-------|
| 1 | 46 | 77.91% | V217 to V278 | interleaved |
| 2 | 43 | 0.05% | V95 to V137 | one solid run |
| 3 | 32 | 0.00% | V279 to V321 | interleaved |
| 4 | 31 | 76.36% | V167 to V216 | interleaved |
| 5 | 23 | 12.88% | V12 to V34 | one solid run |
| 6 | 22 | 13.06% | V53 to V74 | one solid run |
| 7 | 20 | 15.10% | V75 to V94 | one solid run |
| 8 | 19 | 76.32% | V169 to V210 | interleaved |
| 9 | 18 | 28.61% | V35 to V52 | one solid run |
| 10 | 18 | 86.12% | V138 to V163 | interleaved |
| 11 | 18 | 86.05% | V322 to V339 | one solid run |
| 12 | 16 | 76.05% | V220 to V272 | interleaved |
| 13 | 11 | 47.29% | V1 to V11 | one solid run |
| 14 | 11 | 86.12% | V143 to V166 | interleaved |
| 15 | 11 | 0.21% | V281 to V315 | interleaved |

Two things in that table are worth pausing on, because they validate the effort we spent in Step 2.

**Blocks 10 and 14 both sit at 86.12% missing, but they are different blocks.** They have the same number of blanks, in different rows. If Step 2 had grouped columns by their missing count, which is the obvious shortcut, those two would have been merged into one group of 29 columns and we would be treating unrelated features as interchangeable. The hashing approach compares the actual pattern of which rows are blank, not just how many, and that is why it separated them correctly.

**Eight of the fifteen blocks are interleaved.** That means their V numbers weave through each other rather than sitting in a clean run. Block 4 spans V167 to V216 and block 8 spans V169 to V210, so the two are threaded together across the same stretch of numbers. Block 10 (V138 to V163) and block 14 (V143 to V166) are similarly braided.

The consequence: **you cannot reduce the V columns by chopping them into number ranges.** That is the intuitive approach and it would cut straight across the real groupings. The missing pattern is the only reliable guide, and now you have it.

### 2.5 Other findings that shape Step 3

**Identity coverage differs between train and test.** Train is 24.4%, test is 28.0%. That is a 3.6 percentage point gap in the makeup of the data across a 30 day time gap. Nothing is broken, but it is a genuine, measurable shift between the period we train on and the period we score. Note it now, because it becomes a worked example in Step 5 when we build drift monitoring. It is much better to demonstrate drift detection on a shift that really happened than on one you had to manufacture.

**Fraud rates by category are strong and sensible.** Product C at 11.69% against W at 2.04%. Credit cards at 6.68% against debit at 2.43%. Mobile devices at 10.17% against desktop at 6.52%. mail.com at 18.96% against yahoo.com at 2.28%. These are real, usable separations, and they tell us that `ProductCD`, `card6`, `DeviceType`, and the email domains all deserve careful encoding rather than being lumped in with everything else.

**53 columns have no missing values at all, and 12 are more than 90% empty.** The nine emptiest are all identity columns sitting between 99.12% and 99.20% missing, meaning fewer than 5,200 rows out of 590,540 have any value at all. Those are the first candidates for removal in Section 5.4.

**The C columns have zero missing values across the board.** That makes them unusually clean and immediately usable.

---

## 3. Decisions made in this step

| ID | Decision | Why |
|----|----------|-----|
| D-23 | Feature engineering is a **fitted object** that gets saved to disk, not a script that edits data in place | The same transformations must apply at training time and at prediction time, months apart, in a different process, possibly inside a container. The only reliable way to guarantee that is to learn the transformations once, save the learned state, and load it later. A script cannot be loaded. |
| D-24 | The transformer is fitted **only on the first 80% of the training period**, never on the validation portion and never on test | If the frequency of a card value is counted using validation rows, the model has been given information about the validation set before being scored on it. Its score then looks better than it is. Fitting on the training portion only is the whole point of having a split. |
| D-25 | Encodings are learned from training rows only, never from training and test combined | Combining them is common in competition write-ups because the test set is sitting right there. It is not available in production, where you score one transaction at a time with no knowledge of future transactions. We build for the production case. |
| D-26 | `TransactionDT`, `TransactionID`, and any absolute day counter are **excluded from the features** | Explained fully in Section 5.7. In short: test values lie completely outside the range of training values, and tree models cannot extrapolate beyond what they have seen. |
| D-27 | Columns where one value covers 99% or more of rows are dropped, **with a rescue rule** for columns whose rare values are strongly linked to fraud | A column that is 99.2% blank carries almost nothing. But "almost" is not "nothing" when the thing you are predicting happens 3.5% of the time, so we check each candidate against the target before discarding it, and we write down every decision. |
| D-28 | V columns are reduced by **correlation clustering inside each of the 15 blocks**, keeping the column with the most distinct values from each cluster | The blocks tell us which columns came from a shared source. Correlation inside a block tells us which of those are near-duplicates. Keeping the one with the most distinct values keeps the most informative version. |
| D-29 | A `uid` approximating a customer is built from card and address, and used **only for aggregation and frequency counts**, never as a feature on its own | Explained in Section 5.6. Using it directly invites the model to memorise individual customers. |
| D-30 | Every text column becomes an integer with a **stored mapping**. Blank gets its own code. Values never seen in training map to -1 | The mapping has to be stored, or the same word gets a different number next month and the model silently breaks. |
| D-31 | `has_identity` is kept, with the confound from Section 2.3 recorded in the documentation | One column, no cost, some residual signal. But we write down what we know so nobody later mistakes it for a strong feature. |
| D-32 | DVC is introduced now, with a **local folder remote** at `C:\Users\Dauda Agbonoga\dvcstore`, outside the project | Closes Q-03. A local folder needs no account, no internet, and no cost, which suits a machine that is now deliberately offline. Section 15 shows how to swap it for cloud storage later without changing anything else. |
| D-33 | The Streamlit dashboard will draw from **two different data sources**, not one, and its charts will be built from small precomputed artifacts rather than the raw tables | Answering the question you raised. Explained in Section 20 and carried into Step 7. |

---

## 4. What Step 3 produces

**New code:**

| File | Purpose |
|------|---------|
| `config/config.py` | Extended again: Step 3 paths, pruning thresholds, the lists of columns to encode and aggregate |
| `src/utils/column_selection.py` | Finds useless columns and reduces the V columns using the block structure |
| `src/utils/feature_utils.py` | Small, self-contained functions: time features, amount decomposition, email splitting, screen parsing, uid construction |
| `src/features/__init__.py` | New package |
| `src/features/engineer.py` | `FraudFeatureEngineer`, the fitted transformer that holds everything learned from training data |
| `src/pipelines/features.py` | The stage: load, split by time, fit, transform, verify, save |
| `run.py` | Updated with a `features` step |

**New outputs:**

| File | Contents |
|------|----------|
| `data/processed/train_features.parquet` | Model-ready training table, with a `split` column marking train and valid |
| `data/processed/test_features.parquet` | Model-ready test table, same feature columns in the same order |
| `models/feature_engineer.joblib` | The fitted transformer, so Step 4 and Step 6 apply identical transformations |
| `reports/feature_manifest.csv` | Every feature: what kind it is, where it came from, how much is missing |
| `reports/dropped_columns.csv` | Every dropped column with the reason and the evidence |
| `reports/v_column_reduction.csv` | Which V columns were kept, and which ones each represents |
| `reports/feature_summary.md` | The written summary, auto-generated |

---

## 5. The concepts, before any code

Read this whole section before opening an editor. It is the part that makes the code obvious instead of mysterious, and it is what you will be explaining on the PM track.

### 5.1 What feature engineering is actually doing here

The model cannot see what you see. It gets a table of numbers and looks for splits that separate fraud from not-fraud. Feature engineering is the work of putting things into that table that the model could not have worked out for itself.

Three examples from this dataset make it concrete.

The column `TransactionAmt` holds `59.00` and `59.34`. To the model those are just two nearby numbers. But `59.00` being a round figure and `59.34` having odd cents are meaningfully different things in card fraud, and no amount of splitting on the raw number will let the model express "this amount is round". Give it a column that says so and it can.

The column `card1` holds a code like `16075`. The model can split on whether the code is above or below some threshold, which is meaningless, because the codes are labels not quantities. What actually matters is whether this card is one that appears constantly in the data or one that has appeared four times ever. The model cannot count. We count for it.

The column `TransactionAmt` again, but relative: a 900 dollar transaction is ordinary for a card that usually spends 800, and extraordinary for a card that usually spends 30. The model sees one row at a time and has no idea what is usual for that card. We compute the usual and hand over the comparison.

That is the job. Everything in Section 5.6 is one of these three shapes: decompose something, count something, or compare something to its group.

### 5.2 The leakage rule, which is the one that matters most

Here is the rule, stated plainly:

> Anything learned **from** the data must be learned from the training rows only, and then applied to the validation and test rows unchanged.

Counting how often each `card1` value appears is learning from the data. So is computing the average transaction amount per card. So is deciding which text values exist and what number each gets.

If you count `card1` frequencies across the whole training file, including the last 20% that you set aside for validation, then the frequency number attached to each validation row was partly computed from that validation row. The model gets a hint about the answer sheet. Your validation score comes out better than reality, you believe the model is stronger than it is, and the gap shows up only after deployment when it is expensive.

The fix is structural, not a matter of remembering to be careful:

1. Split the training data by time first, before touching anything else
2. Fit the transformer on the earlier portion only
3. Apply it to all three sets: the earlier portion, the validation portion, and the test set

Step 3 does exactly this, in that order. That is D-24 and D-25.

One honest trade-off to note. Fitting on 80% of training rather than 100% means the frequency counts and averages are built on slightly less data than they could be. That makes the final test predictions marginally weaker than they might otherwise be. We accept it because it keeps the validation number trustworthy, which is worth more than a marginal gain. Step 4 revisits whether to refit on the full training set before producing final predictions.

### 5.3 Why the transformer must be a saved object

Picture Step 6. A transaction arrives at a web service. It is one row. To score it, the service needs to know:

- Which of the 435 original columns to keep
- What number `gmail.com` maps to
- How often `card1 = 16075` appeared in training
- What the average transaction amount was for that card

None of that can be worked out from a single row. It all has to come from training data that is long gone by then. So it must have been saved.

This is why the code is built as a class with `fit` and `transform` rather than as a run of pandas operations. `fit` learns and stores. `transform` applies what was stored. `joblib.dump` writes the whole thing to a file. Step 4 loads it. Step 6 loads it inside a container.

We make the class inherit from scikit-learn's `BaseEstimator` and `TransformerMixin`. That is a small amount of extra typing that buys real things: the object slots into a scikit-learn `Pipeline` alongside a model, `get_params` and `set_params` work for free, and anyone who knows scikit-learn immediately knows how to use it.

The failure this design prevents is called training and serving skew, and it is the most common way a working model quietly breaks in production. The model was trained on one set of transformations and is being fed another. Nothing errors. The predictions are just wrong.

### 5.4 Getting rid of columns that carry nothing

Two rules, applied in order, both computed on the training portion only.

**Rule one: drop columns with one distinct value.** If every row says `1.0`, there is nothing to split on. This is not a judgement call.

**Rule two: drop columns where one value covers 99% or more of all rows, counting blank as a value.** This catches the nine identity columns sitting above 99.12% missing. It also catches V columns that are almost always zero.

Rule two makes me slightly uneasy on its own, and it is worth saying why. Fraud happens 3.5% of the time. A column that is 99% one value has 1% something else, and 1% of 590,540 is 5,905 rows. If fraud were heavily concentrated in those 5,905 rows, that column would be one of the most useful things in the dataset and rule two would throw it away without a word.

So rule two comes with a rescue. For every column rule two flags, we check the rows that do **not** hold the dominant value and measure the fraud rate among them. If there are at least 500 such rows and their fraud rate is at least twice the overall rate, or at most half of it, the column is kept. A rare value that is strongly linked to fraud is a signal. A rare value that is strongly linked to safety is also a signal.

Every candidate, kept or dropped, is written to `reports/dropped_columns.csv` with its evidence. If you disagree with a decision later, you can see exactly why it was made and change the threshold.

The rescue uses the target, which makes it a form of supervised selection. That is legitimate because it runs on training rows only, which is where you are allowed to look at the target. But it is worth naming, because the same operation done on validation rows would be leakage.

### 5.5 Reducing the V columns, using your actual blocks

339 of your 435 columns are V columns. That is 78%. They are fully anonymised, so we cannot reason about their meaning. We can reason about their structure.

**Step one is done.** Your 15 blocks tell us which V columns came from a shared source, because they go blank together.

**Step two is the new work.** Within a single block, columns are often near-duplicates of each other. Two columns that move together at 0.98 correlation carry the same information twice. Keeping both costs memory and training time and gives the model two ways to say the same thing, which makes feature importance harder to read later.

So inside each block, we:

1. Take only the rows where that block is actually present. For block 1 at 77.91% missing, that is about 130,000 rows. Computing correlation on rows where everything is blank would be meaningless.
2. Compute the correlation of every column against every other column in the block.
3. Group them greedily. Start with the first column, pull in every other column correlated with it at 0.75 or above, and that is one cluster. Move to the first column not yet assigned, repeat, until every column is in a cluster.
4. From each cluster keep exactly one column: the one with the most distinct values. More distinct values means finer resolution, which means more the model can split on.

Why 0.75 and not 0.95? Because we are not trying to remove only exact duplicates. We are trying to collapse groups that say substantially the same thing. 0.75 is the threshold most commonly used on this dataset and it reduces the count meaningfully without being reckless. It lives in `config.py` so you can change it and re-run.

The greedy grouping is not the mathematically optimal clustering. A proper hierarchical clustering would be slightly better. Greedy is used because it is deterministic, it runs in seconds, and you can read the code and understand exactly what it did, which matters more here than the last small increment of quality.

I cannot tell you in advance how many columns survive, because it depends on correlations inside your data that I have not seen. Based on how this dataset usually behaves, expect somewhere between 100 and 170 out of 339. The code prints the exact number and writes the full mapping to `reports/v_column_reduction.csv`.

### 5.6 The features we are going to build

**From the time column.** Hour of day, and day of week. Both repeat: there is an hour 14 in December and an hour 14 in the following July, so the model can learn something in training that still applies at test time. Your chart `05_fraud_rate_by_hour.png` already showed that fraud rate varies by hour, so this is grounded in something you have seen rather than a guess.

**From the amount.** The log of the amount, which pulls in the long tail so the model is not dominated by a handful of enormous transactions. The cents portion, extracted as its own column. And a flag for whether the amount is a round number. This is the feature that decision D-18 in Step 2 protected the `float64` precision for, and this is where that pays off.

**Counting features, called frequency encoding.** For each of `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, both email domains, `DeviceInfo`, `id_30`, `id_31`, `id_33`, and some combinations, we count how often each value appeared in training and store the share. A card code seen 40,000 times gets a high number. One seen twice gets a number near zero. Rarity is a fraud signal, and this is how you hand it to the model.

We store the share rather than the raw count, so that a training set of 472,000 rows and a test set of 506,000 rows produce comparable numbers. Values never seen in training get 0, which is truthful: as far as training knows, this value does not exist.

**Comparison features, called aggregates.** For a set of grouping columns, we compute the average and spread of `TransactionAmt` within each group during training. Then for every row we attach the group average, the group spread, and the ratio of this transaction's amount to its group average. A ratio of 30 means this transaction is thirty times the typical spend for that card. That is the kind of thing a fraud analyst notices, and the model cannot see it without help. We do the same for `D15`.

**From the email domains.** We split `gmail.com` into a provider part and a suffix part, so that `yahoo.com`, `yahoo.co.uk`, and `yahoo.com.mx` are recognisably related instead of being three unrelated labels. We also add a flag for whether the purchaser and recipient domains match, which is a classic fraud indicator.

**From the device columns.** The first word of `DeviceInfo` gives a rough brand. The first word of `id_31` gives a browser family, so `chrome 62.0` and `chrome 63.0` group together. `id_33` holds a screen resolution like `1920x1080`, which we split into width, height, and area. All three of these turn messy free text into something a model can use.

**From the M columns.** These are agreement checks with values T and F. We turn each into a number, and add a count of how many said T and how many were blank. Your EDA shows the M family averages 49.92% missing, so how many are blank is itself informative.

**The uid.** This is the most interesting idea in the dataset and the one with the sharpest edge, so it gets its own explanation.

The data has no customer identifier. But `D1` measures days since the card was first seen. So if you take the day number of the transaction and subtract `D1`, you get the day that card first appeared, which is a fixed value for a given card no matter when you observe it. Combine that with `card1` and `addr1` and you have a reasonably stable fingerprint for one customer:

```
uid = card1 + "_" + addr1 + "_" + (day_number - D1)
```

This is powerful, because it lets you compute per-customer averages rather than per-card-code averages.

It is also risky, in two specific ways.

First, it is a guess. Two different customers can collide on the same fingerprint, and one customer can split into two if `addr1` changes.

Second, and more seriously, if you feed the uid to the model as a feature in its own right, the model will memorise individual customers. It learns "fingerprint 16075_315_82 commits fraud" rather than "this pattern of behaviour indicates fraud". That looks excellent in validation, where the same customers appear on both sides, and it is worth nothing on genuinely new customers.

So: **the uid is used for grouping and counting only.** It never appears as a feature itself. That is D-29, and it is the sort of restraint that separates a model that works from a model that scores well.

### 5.7 Why the raw time column is not a feature

`TransactionDT` runs from 86,400 to 15,811,131 in training. In test it runs from about 18.4 million to 34.2 million. There is no overlap at all.

Now think about what a decision tree does. It learns rules of the form "if `TransactionDT` is above 12,000,000 then lean towards fraud". Every single test row is above 12,000,000, because the entire test period is later than the entire training period. That rule fires for everything and separates nothing.

Tree models cannot extrapolate. They can only split within the range of values they were trained on. Give them a column whose test values are entirely outside the training range and you have given them a column that is worse than useless: it is a column that looked helpful during training and does nothing afterwards.

The same argument rules out an absolute day counter, a week number, or a month index.

What survives is anything **cyclical**. Hour of day runs 0 to 23 in both periods. Day of week runs 0 to 6 in both. Those genuinely transfer.

`TransactionDT` still travels through the pipeline, because we need it to sort rows and to build the split, and Step 5 needs it to group predictions by time period. It sits in the file as a carried-along column, clearly separated from the feature list. The code enforces this rather than relying on anyone remembering.

### 5.8 The split, in code

Take the training file. Find the `TransactionDT` value at the 80th percentile. Every row at or below it is training, every row above it is validation.

Roughly 472,000 training rows and 118,000 validation rows. The validation portion is the last five weeks or so of the training period, which is a reasonable stand-in for the 30 day gap between training and test.

The split is written into the file as a `split` column holding `train` or `valid`. Step 4 reads that column rather than recomputing the split, which guarantees that every experiment is scored on exactly the same rows. If the split were recomputed in each script, a small difference in rounding would make two runs quietly non-comparable.

---

## 6. Update `config/config.py`

### 6.1 What is being added

Step 3 paths, the pruning thresholds, and the lists that drive frequency encoding and aggregation. Putting those lists in config rather than burying them in code means you can add a grouping column later by editing one line.

### 6.2 The addition

**Append this to the end of `config/config.py`**, just before the `ensure_directories` function. Everything already in the file stays.

```python
# =========================================================
# STEP 3: FEATURE ENGINEERING
# =========================================================

# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

PREPROCESSOR_FILE = MODELS_DIR / "feature_engineer.joblib"
FEATURE_MANIFEST_FILE = REPORTS_DIR / "feature_manifest.csv"
DROPPED_COLUMNS_FILE = REPORTS_DIR / "dropped_columns.csv"
V_REDUCTION_FILE = REPORTS_DIR / "v_column_reduction.csv"
FEATURE_SUMMARY_FILE = REPORTS_DIR / "feature_summary.md"


# ---------------------------------------------------------
# The split column written into the processed files
# ---------------------------------------------------------

SPLIT_COLUMN = "split"
TRAIN_SPLIT_LABEL = "train"
VALID_SPLIT_LABEL = "valid"


# ---------------------------------------------------------
# Columns carried through the pipeline but NEVER used as features.
#
# TransactionID identifies a row and means nothing about fraud.
# TransactionDT is needed for sorting and splitting, but its test values
#   sit entirely outside the training range, so a tree cannot use it.
# isFraud is the answer.
# ---------------------------------------------------------

PASSTHROUGH_COLUMNS = [ID_COLUMN, TIME_COLUMN, TARGET_COLUMN]


# ---------------------------------------------------------
# Column pruning thresholds
# ---------------------------------------------------------

# Drop a column when one single value (blank counts as a value) covers
# this share of all rows or more.
NEAR_CONSTANT_THRESHOLD = 0.99

# A near-constant column is rescued when the rows that do NOT hold the
# dominant value are both numerous enough and unusual enough.
RESCUE_MIN_RARE_ROWS = 500
RESCUE_MIN_FRAUD_LIFT = 2.0

# Two V columns inside the same block are treated as near-duplicates
# when the absolute correlation between them reaches this level.
V_CORRELATION_THRESHOLD = 0.75


# ---------------------------------------------------------
# Text handling
# ---------------------------------------------------------

# Blank values become this label, so that "we do not know" is a real
# category the model can split on rather than a hole.
MISSING_LABEL = "(missing)"

# A value present in test but never seen in training gets this code.
UNSEEN_CATEGORY_CODE = -1


# ---------------------------------------------------------
# Frequency encoding: count how often each value appears in training.
#
# Includes derived columns (uid, card1_addr1, the email and device parts)
# which do not exist in the raw data. The code skips anything missing
# rather than failing, so this list is safe to edit.
# ---------------------------------------------------------

FREQUENCY_ENCODE_COLUMNS = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
    "card1_addr1",
    "uid",
    "P_email_provider",
    "R_email_provider",
    "device_brand",
    "browser_family",
]


# ---------------------------------------------------------
# Aggregate features: (group by this, summarise this).
#
# Each pair produces three columns: the group average, the group spread,
# and the ratio of this row's value to its group average.
# ---------------------------------------------------------

AGGREGATION_SPECS = [
    ("card1", "TransactionAmt"),
    ("addr1", "TransactionAmt"),
    ("card1_addr1", "TransactionAmt"),
    ("uid", "TransactionAmt"),
    ("card1", "D15"),
    ("uid", "D15"),
]


# ---------------------------------------------------------
# Columns used to build the uid customer fingerprint.
# ---------------------------------------------------------

UID_CARD_COLUMN = "card1"
UID_ADDRESS_COLUMN = "addr1"
UID_TIMEDELTA_COLUMN = "D1"
```

### 6.3 Check it

```powershell
python -c "from config.config import FREQUENCY_ENCODE_COLUMNS, AGGREGATION_SPECS, PASSTHROUGH_COLUMNS; print(len(FREQUENCY_ENCODE_COLUMNS), len(AGGREGATION_SPECS)); print(PASSTHROUGH_COLUMNS)"
```

**Expected output:**

```
18 6
['TransactionID', 'TransactionDT', 'isFraud']
```

---

## 7. Create `src/utils/column_selection.py`

### 7.1 What is in here

Everything to do with deciding which columns survive: the two pruning rules, the rescue check, and the V block reduction.

### 7.2 The file

```python
"""
Deciding which columns to keep.

Three jobs:
  1. Find columns with no information at all
  2. Find columns where one value dominates, and rescue the ones whose
     rare values are strongly linked to fraud
  3. Reduce the 339 V columns using their block structure

Everything here runs on TRAINING ROWS ONLY. Deciding what to keep is a
form of learning from the data, so it obeys the same rule as every other
fitted step.
"""

from __future__ import annotations

import pandas as pd


def top_value_share(series: pd.Series) -> tuple[object, float]:
    """
    Find the most common value in a column and what share of rows it covers.

    dropna=False makes blank count as a value. That is deliberate: a column
    that is 99% blank is just as uninformative as one that is 99% zero, and
    we want the same rule to catch both.

    Returns the value and its share, as a number between 0 and 1.
    """
    if len(series) == 0:
        return None, 1.0

    counts = series.value_counts(dropna=False, normalize=True)
    if counts.empty:
        return None, 1.0

    return counts.index[0], float(counts.iloc[0])


def find_constant_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    Find columns holding a single distinct value, or nothing at all.

    nunique(dropna=True) ignores blanks, so a column that is entirely blank
    returns 0 and a column holding one repeated value returns 1. Both are
    useless: there is nothing for the model to split on.
    """
    constant = []
    for column in columns:
        if frame[column].nunique(dropna=True) <= 1:
            constant.append(column)
    return constant


def assess_near_constant_columns(
    frame: pd.DataFrame,
    target: pd.Series,
    columns: list[str],
    threshold: float,
    min_rare_rows: int,
    min_fraud_lift: float,
) -> pd.DataFrame:
    """
    Judge every column against the near-constant rule, with a rescue check.

    For each column we find the dominant value and its share. If the share
    is below the threshold, the column is kept and nothing more happens.

    If the share is at or above the threshold, we look at the rows that do
    NOT hold the dominant value and measure the fraud rate among them. A
    column is rescued when there are enough of those rows and their fraud
    rate is far from the overall rate, in either direction. A rare value
    strongly linked to fraud is a signal. A rare value strongly linked to
    safety is also a signal.

    Returns one row per column with the full evidence, so that every
    decision can be inspected afterwards rather than taken on trust.
    """
    base_rate = float(target.mean())
    records = []

    for column in columns:
        series = frame[column]
        dominant_value, share = top_value_share(series)

        if share < threshold:
            records.append(
                {
                    "column": column,
                    "dominant_value": str(dominant_value),
                    "dominant_share": round(share, 5),
                    "rare_rows": int(len(series) - round(share * len(series))),
                    "rare_fraud_rate": None,
                    "fraud_lift": None,
                    "decision": "keep",
                    "reason": "not near-constant",
                }
            )
            continue

        # Rows that do not hold the dominant value. Note that when the
        # dominant value is a real value, blank rows count as "rare",
        # because a comparison against blank is always False in pandas.
        if pd.isna(dominant_value):
            rare_mask = series.notna()
        else:
            rare_mask = series != dominant_value

        rare_rows = int(rare_mask.sum())

        if rare_rows < min_rare_rows:
            records.append(
                {
                    "column": column,
                    "dominant_value": str(dominant_value),
                    "dominant_share": round(share, 5),
                    "rare_rows": rare_rows,
                    "rare_fraud_rate": None,
                    "fraud_lift": None,
                    "decision": "drop",
                    "reason": f"near-constant, only {rare_rows} rare rows",
                }
            )
            continue

        rare_fraud_rate = float(target[rare_mask].mean())
        lift = rare_fraud_rate / base_rate if base_rate else 0.0

        # Rescue when the rare group is much riskier OR much safer than
        # average. Both directions are useful information.
        rescued = (lift >= min_fraud_lift) or (lift <= 1 / min_fraud_lift)

        records.append(
            {
                "column": column,
                "dominant_value": str(dominant_value),
                "dominant_share": round(share, 5),
                "rare_rows": rare_rows,
                "rare_fraud_rate": round(rare_fraud_rate, 5),
                "fraud_lift": round(lift, 3),
                "decision": "keep" if rescued else "drop",
                "reason": (
                    "rescued: rare values linked to fraud"
                    if rescued
                    else "near-constant, rare values unremarkable"
                ),
            }
        )

    return pd.DataFrame(records)


def load_v_groups(path) -> list[list[str]]:
    """
    Read the V block structure written by the Step 2 EDA stage.

    The CSV stores each block's members as one comma-separated string, so
    we split it back into a list and strip the spaces.
    """
    table = pd.read_csv(path)
    groups = []
    for _, row in table.iterrows():
        members = [name.strip() for name in str(row["columns"]).split(",")]
        groups.append([name for name in members if name])
    return groups


def cluster_by_correlation(
    correlations: pd.DataFrame, threshold: float
) -> list[list[str]]:
    """
    Group columns that move together, using a simple greedy pass.

    Take the first unassigned column and open a cluster with it. Pull in
    every other unassigned column whose absolute correlation with it
    reaches the threshold. Close the cluster, move to the next unassigned
    column, repeat.

    We use the ABSOLUTE correlation because a column that moves exactly
    opposite to another carries the same information, just with the sign
    flipped, and a tree can flip a sign for free.

    This is not the mathematically best clustering available. A full
    hierarchical clustering would group slightly better. Greedy is used
    because it always gives the same answer, it runs in seconds, and you
    can read it and know exactly what it did.
    """
    columns = list(correlations.columns)
    assigned: set[str] = set()
    clusters: list[list[str]] = []

    for column in columns:
        if column in assigned:
            continue

        cluster = [column]
        assigned.add(column)

        for other in columns:
            if other in assigned:
                continue
            value = correlations.loc[column, other]
            # A correlation is blank when one of the columns never varies
            # over the rows being compared. Blank is not a match.
            if pd.notna(value) and abs(value) >= threshold:
                cluster.append(other)
                assigned.add(other)

        clusters.append(cluster)

    return clusters


def choose_representative(frame: pd.DataFrame, cluster: list[str]) -> str:
    """
    Pick one column to stand for a cluster of near-duplicates.

    We keep the one with the most distinct values. More distinct values
    means finer resolution, which means more places a tree can split, which
    means more of the original information survives.
    """
    if len(cluster) == 1:
        return cluster[0]

    distinct_counts = {column: frame[column].nunique() for column in cluster}
    return max(distinct_counts, key=lambda column: distinct_counts[column])


def reduce_v_columns(
    frame: pd.DataFrame,
    groups: list[list[str]],
    threshold: float,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Cut the V columns down using their block structure.

    For each block:
      1. Keep only the rows where the block is actually present. Every
         column in a block goes blank on the same rows, so dropna gives
         exactly those rows. Correlating rows that are entirely blank
         would tell us nothing.
      2. Correlate every column against every other column in the block.
      3. Group them greedily.
      4. Keep the column with the most distinct values from each group.

    Returns the surviving column names and a table recording which columns
    each survivor stands in for.
    """
    kept: list[str] = []
    records = []

    for group_index, group in enumerate(groups, start=1):
        available = [column for column in group if column in frame.columns]

        if not available:
            continue

        if len(available) == 1:
            kept.append(available[0])
            records.append(
                {
                    "block": group_index,
                    "cluster": 1,
                    "kept_column": available[0],
                    "n_represented": 1,
                    "represents": available[0],
                }
            )
            continue

        present_rows = frame[available].dropna()

        # If a block is blank on every training row there is nothing to
        # correlate, so keep the first column and move on.
        if len(present_rows) < 2:
            kept.append(available[0])
            records.append(
                {
                    "block": group_index,
                    "cluster": 1,
                    "kept_column": available[0],
                    "n_represented": len(available),
                    "represents": ", ".join(available),
                }
            )
            continue

        correlations = present_rows.corr()
        clusters = cluster_by_correlation(correlations, threshold)

        for cluster_index, cluster in enumerate(clusters, start=1):
            representative = choose_representative(present_rows, cluster)
            kept.append(representative)
            records.append(
                {
                    "block": group_index,
                    "cluster": cluster_index,
                    "kept_column": representative,
                    "n_represented": len(cluster),
                    "represents": ", ".join(cluster),
                }
            )

        if verbose:
            print(
                f"    block {group_index:>2}: {len(available):>3} columns -> "
                f"{len(clusters):>3} kept"
            )

    detail = pd.DataFrame(records)
    return kept, detail
```

---

## 8. Create `src/utils/feature_utils.py`

### 8.1 What is in here

Small functions that build derived columns. None of them learn anything from the data, so none of them can leak. They behave identically on training, validation, and test.

Keeping them separate from the transformer class means each one can be tested on its own in Step 5, and the class stays readable.

### 8.2 One shared idea: label series

Several things need a text version of a column: frequency counting, grouping, and building the uid. Numbers, text, and categories all have to end up as plain strings in a consistent way, with blanks turned into a real label.

`as_label_series` does that once, and everything else uses it. Doing it in one place is what stops training and test from drifting apart, for example one of them producing `"315"` and the other producing `"315.0"` for the same address.

### 8.3 The file

```python
"""
Small, self-contained functions that build derived columns.

Nothing here learns anything from the data. Each function does the same
thing to any row it is given, so these can be applied to training,
validation, and test with no risk of leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import (
    AMOUNT_COLUMN,
    MISSING_LABEL,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    TIME_COLUMN,
    UID_ADDRESS_COLUMN,
    UID_CARD_COLUMN,
    UID_TIMEDELTA_COLUMN,
)


def as_label_series(series: pd.Series) -> pd.Series:
    """
    Turn any column into plain text labels, with blanks made explicit.

    Why this exists: frequency counting, grouping, and building the uid all
    need text. If each of them converted numbers to text its own way, train
    and test could end up producing different strings for the same value,
    for example "315" against "315.0", and every lookup would silently miss.

    Numbers go through float64 first, so the result does not depend on
    whether a column happened to be stored as int16 in one file and int32
    in another. Rounding to six decimal places removes float noise that
    would otherwise create near-duplicate labels.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        values = series.astype("object")
    elif pd.api.types.is_numeric_dtype(series):
        values = series.astype("float64").round(6).astype("object")
    else:
        values = series.astype("object")

    labels = values.where(series.notna(), MISSING_LABEL)
    return labels.astype(str)


def combine_labels(*label_series: pd.Series) -> pd.Series:
    """Join several label columns into one, separated by underscores."""
    combined = label_series[0]
    for extra in label_series[1:]:
        combined = combined + "_" + extra
    return combined


def build_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build hour of day and day of week from the seconds counter.

    Only cyclical features are produced. Hour runs 0 to 23 in the training
    period and 0 to 23 in the test period, so anything the model learns
    about hour still applies later. An absolute day counter would not:
    every test value sits above every training value, and a tree cannot
    split usefully outside the range it was trained on.

    The day-of-week numbers may not line up with real calendar names,
    because the reference date is a convention rather than a fact. That
    does not matter. What matters is that the same real weekday always
    produces the same number, and it does, because both come from the same
    continuous seconds counter.
    """
    seconds = frame[TIME_COLUMN].astype("int64")

    day_number = seconds // SECONDS_PER_DAY

    return pd.DataFrame(
        {
            "hour": ((seconds // SECONDS_PER_HOUR) % 24).astype("int8"),
            "day_of_week": (day_number % 7).astype("int8"),
        },
        index=frame.index,
    )


def build_amount_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Break the transaction amount into parts the model can use.

    amount_log
        Amounts are heavily skewed: most are small, a few are enormous.
        The logarithm compresses the tail so a handful of huge transactions
        do not dominate every split. log1p is log(1 + x), which behaves
        properly when the amount is zero.

    amount_cents
        The part after the decimal point. This is the feature that decision
        D-18 in Step 2 kept float64 precision for. Legitimate purchases
        cluster on particular cent values, currency conversions produce
        distinctive ones, and fraud distributes differently.

    amount_is_round
        Whether the amount has no cents at all.
    """
    amount = frame[AMOUNT_COLUMN].astype("float64")

    cents = (amount - np.floor(amount)).round(4)

    return pd.DataFrame(
        {
            "amount_log": np.log1p(amount).astype("float32"),
            "amount_cents": cents.astype("float32"),
            "amount_is_round": (cents == 0).astype("int8"),
        },
        index=frame.index,
    )


def split_email_domain(labels: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Split an email domain into a provider part and a suffix part.

    "gmail.com" becomes provider "gmail" and suffix "com".
    "yahoo.co.uk" becomes provider "yahoo" and suffix "co.uk".

    The point is that yahoo.com, yahoo.co.uk, and yahoo.com.mx are the same
    provider. Left as whole strings they are three unrelated labels and the
    model has to learn about each separately, with far fewer examples each.

    The blank label has no dot in it, so it survives the split intact and
    stays its own category on both sides.
    """
    parts = labels.str.split(".", n=1, expand=True)

    provider = parts[0]
    if parts.shape[1] > 1:
        suffix = parts[1].fillna(MISSING_LABEL)
    else:
        suffix = pd.Series(MISSING_LABEL, index=labels.index)

    return pd.DataFrame(
        {
            f"{prefix}_email_provider": provider,
            f"{prefix}_email_suffix": suffix,
        },
        index=labels.index,
    )


def build_screen_features(labels: pd.Series) -> pd.DataFrame:
    """
    Turn a screen resolution like "1920x1080" into three numbers.

    errors="coerce" means anything that does not parse becomes blank rather
    than raising. The blank label does not contain an "x", so it lands in
    the first part and fails to convert, which is exactly what we want.
    """
    parts = labels.str.split("x", n=1, expand=True)

    width = pd.to_numeric(parts[0], errors="coerce")
    if parts.shape[1] > 1:
        height = pd.to_numeric(parts[1], errors="coerce")
    else:
        height = pd.Series(np.nan, index=labels.index)

    return pd.DataFrame(
        {
            "screen_width": width.astype("float32"),
            "screen_height": height.astype("float32"),
            "screen_area": (width * height).astype("float32"),
        },
        index=labels.index,
    )


def first_token(labels: pd.Series) -> pd.Series:
    """
    Take the first word of a messy text column.

    "SAMSUNG SM-G892A Build/NRD90M" becomes "samsung".
    "chrome 62.0" becomes "chrome".

    This collapses hundreds of near-identical values into a handful of
    useful groups. Splitting on "/" as well catches strings where the first
    word already carries a build path.
    """
    token = labels.str.split(" ", n=1).str[0]
    token = token.str.split("/", n=1).str[0]
    return token.str.lower()


def build_match_features(frame: pd.DataFrame, match_columns: list[str]) -> pd.DataFrame:
    """
    Summarise the M columns, which are agreement checks.

    M1 to M3 and M5 to M9 hold T or F. M4 is different, holding M0, M1, or
    M2, so it is left out of the "how many said T" count but still counted
    towards how many are blank.

    Two summaries are produced: how many checks passed, and how many were
    not available. The second matters because the M family averages just
    under 50% missing in this dataset, so the amount of missingness varies
    a lot from row to row.
    """
    available = [column for column in match_columns if column in frame.columns]
    if not available:
        return pd.DataFrame(index=frame.index)

    true_count = pd.Series(0, index=frame.index, dtype="int8")
    missing_count = pd.Series(0, index=frame.index, dtype="int8")

    for column in available:
        labels = as_label_series(frame[column])
        if column != "M4":
            true_count = true_count + (labels == "T").astype("int8")
        missing_count = missing_count + (labels == MISSING_LABEL).astype("int8")

    return pd.DataFrame(
        {"match_true_count": true_count, "match_missing_count": missing_count},
        index=frame.index,
    )


def build_uid(frame: pd.DataFrame) -> pd.Series:
    """
    Build a rough customer fingerprint.

    The dataset has no customer identifier. But D1 measures days since the
    card was first seen, so:

        day of this transaction  minus  D1  =  the day the card first appeared

    That subtraction gives the same answer for every transaction on the
    same card, whenever it happens. Combined with card1 and addr1 it is a
    reasonably stable fingerprint for one customer.

    Two warnings, both taken seriously elsewhere in the code.

    It is a guess, not a fact. Two customers can collide on one fingerprint,
    and one customer can split in two if their address changes.

    More importantly, it must never become a feature in its own right. A
    model given the fingerprint memorises individual customers, which looks
    excellent in validation, where the same customers appear on both sides
    of the split, and is worth nothing on customers it has never met. So
    the uid is used only for grouping and counting. That is decision D-29.
    """
    missing_columns = [
        column
        for column in (UID_CARD_COLUMN, UID_ADDRESS_COLUMN, UID_TIMEDELTA_COLUMN)
        if column not in frame.columns
    ]
    if missing_columns:
        raise KeyError(f"cannot build uid, missing columns: {missing_columns}")

    day_number = (frame[TIME_COLUMN].astype("int64") // SECONDS_PER_DAY)
    first_seen_day = day_number - frame[UID_TIMEDELTA_COLUMN].astype("float64")

    return combine_labels(
        as_label_series(frame[UID_CARD_COLUMN]),
        as_label_series(frame[UID_ADDRESS_COLUMN]),
        as_label_series(first_seen_day),
    )
```

---

## 9. Create the `src/features` package

### 9.1 Why a new folder

The transformer class is the largest single piece of code in the project so far, and it is not a small helper. Putting it in `src/utils/` would stretch what "utils" means past breaking point.

`src/features/` is a standard layout and it makes the project self-describing: `pipelines` runs stages, `features` builds features, `serving` answers requests, `monitoring` watches, `utils` supports all of them.

```powershell
# Create the package folder and its marker file
New-Item -ItemType Directory -Force -Path "src\features" | Out-Null
New-Item -ItemType File -Force -Path "src\features\__init__.py" | Out-Null
```

### 9.2 Create `src/features/engineer.py`

This is the heart of Step 3. Read the class docstring first, then work down.

```python
"""
FraudFeatureEngineer: the fitted transformer that turns the joined table
into a model-ready feature table.

Why this is a class and not a script
------------------------------------
Some transformations need to know things that can only be worked out from
training data. How often card1 = 16075 appeared. What number gmail.com maps
to. What the average transaction amount is for a given card.

In Step 6 a single transaction arrives at a web service and has to be
scored. None of those facts can be worked out from one row. They have to
have been saved.

So: fit() learns them and stores them on the object. transform() applies
what was stored. joblib.dump() writes the whole object to disk. Step 4 and
Step 6 load it and get transformations identical to training.

The class inherits from scikit-learn's BaseEstimator and TransformerMixin,
which costs almost nothing and means the object drops straight into a
scikit-learn Pipeline alongside a model.

The leakage rule
----------------
fit() must only ever see training rows. The pipeline stage in
src/pipelines/features.py enforces that by splitting on time first and
passing only the earlier portion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.config import (
    AGGREGATION_SPECS,
    FREQUENCY_ENCODE_COLUMNS,
    MISSING_LABEL,
    M_COLUMNS,
    NEAR_CONSTANT_THRESHOLD,
    PASSTHROUGH_COLUMNS,
    RESCUE_MIN_FRAUD_LIFT,
    RESCUE_MIN_RARE_ROWS,
    UNSEEN_CATEGORY_CODE,
    V_CORRELATION_THRESHOLD,
)
from src.utils.column_selection import (
    assess_near_constant_columns,
    find_constant_columns,
    reduce_v_columns,
)
from src.utils.feature_utils import (
    as_label_series,
    build_amount_features,
    build_match_features,
    build_screen_features,
    build_time_features,
    build_uid,
    combine_labels,
    first_token,
    split_email_domain,
)

# Derived text columns that are built inside this class rather than read
# from the raw data. They are treated as categories from then on.
DERIVED_LABEL_COLUMNS = [
    "P_email_provider",
    "P_email_suffix",
    "R_email_provider",
    "R_email_suffix",
    "device_brand",
    "browser_family",
    "card1_addr1",
]

# Built and used, but never handed to the model on its own. See D-29.
GROUPING_ONLY_COLUMNS = ["uid"]


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Learns feature transformations from training data and applies them."""

    def __init__(
        self,
        v_groups: list[list[str]] | None = None,
        near_constant_threshold: float = NEAR_CONSTANT_THRESHOLD,
        rescue_min_rare_rows: int = RESCUE_MIN_RARE_ROWS,
        rescue_min_fraud_lift: float = RESCUE_MIN_FRAUD_LIFT,
        v_correlation_threshold: float = V_CORRELATION_THRESHOLD,
        verbose: bool = True,
    ):
        # scikit-learn requires that __init__ only stores its arguments and
        # does no work. Anything computed here would be lost when the object
        # is cloned, which Pipeline and cross-validation both do.
        self.v_groups = v_groups
        self.near_constant_threshold = near_constant_threshold
        self.rescue_min_rare_rows = rescue_min_rare_rows
        self.rescue_min_fraud_lift = rescue_min_fraud_lift
        self.v_correlation_threshold = v_correlation_threshold
        self.verbose = verbose

    # -----------------------------------------------------
    # Small internal helpers
    # -----------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _tag(self, name: str, kind: str, source: str) -> None:
        """
        Record what a feature is and where it came from.

        Only runs while fitting. The result becomes reports/feature_manifest.csv,
        which is what lets you answer "where did this column come from" in
        four months without rereading the code.
        """
        if getattr(self, "_recording", False):
            self.feature_tags_.append({"feature": name, "kind": kind, "source": source})

    # -----------------------------------------------------
    # Building the label columns (text versions used for counting)
    # -----------------------------------------------------

    def _build_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Build the text version of every column that needs counting or grouping.

        These are intermediate working columns. Most of them never reach the
        final feature table directly; they get turned into codes and counts
        first.
        """
        labels = pd.DataFrame(index=frame.index)

        # Raw columns that survived pruning and are either text or an
        # identifier-style number such as card1.
        for column in self.label_source_columns_:
            if column in frame.columns:
                labels[column] = as_label_series(frame[column])
            else:
                labels[column] = MISSING_LABEL

        # Email domains split into provider and suffix.
        for prefix, column in (("P", "P_emaildomain"), ("R", "R_emaildomain")):
            if column in labels.columns:
                parts = split_email_domain(labels[column], prefix)
                labels[parts.columns] = parts
            else:
                labels[f"{prefix}_email_provider"] = MISSING_LABEL
                labels[f"{prefix}_email_suffix"] = MISSING_LABEL

        # Device brand and browser family, from the first word of each.
        labels["device_brand"] = (
            first_token(labels["DeviceInfo"])
            if "DeviceInfo" in labels.columns
            else MISSING_LABEL
        )
        labels["browser_family"] = (
            first_token(labels["id_31"])
            if "id_31" in labels.columns
            else MISSING_LABEL
        )

        # A combined card and address key. Two customers can share a card
        # code, and two can share an address, but sharing both is rarer, so
        # the pair is a sharper grouping than either alone.
        if "card1" in labels.columns and "addr1" in labels.columns:
            labels["card1_addr1"] = combine_labels(labels["card1"], labels["addr1"])
        else:
            labels["card1_addr1"] = MISSING_LABEL

        # The customer fingerprint. Grouping only.
        if self.can_build_uid_:
            labels["uid"] = build_uid(frame)
        else:
            labels["uid"] = MISSING_LABEL

        return labels

    # -----------------------------------------------------
    # fit
    # -----------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Learn everything that has to be remembered.

        X must be TRAINING ROWS ONLY. y is the fraud label for those rows,
        and is required, because the rescue rule in the pruning step needs
        to compare fraud rates.
        """
        if y is None:
            raise ValueError(
                "fit needs the target. The near-constant rescue rule compares "
                "fraud rates, so it cannot run without labels."
            )

        y = pd.Series(y).reset_index(drop=True)
        frame = X.reset_index(drop=True)

        self._log("  Selecting columns ...")

        # --- 1. work out which columns are even candidates ---------------
        candidates = [
            column for column in frame.columns if column not in PASSTHROUGH_COLUMNS
        ]

        # --- 2. drop columns with a single value -------------------------
        constant_columns = find_constant_columns(frame, candidates)
        self._log(f"    {len(constant_columns)} columns hold a single value")

        remaining = [c for c in candidates if c not in constant_columns]

        # --- 3. near-constant rule, with the rescue check ----------------
        assessment = assess_near_constant_columns(
            frame,
            y,
            remaining,
            threshold=self.near_constant_threshold,
            min_rare_rows=self.rescue_min_rare_rows,
            min_fraud_lift=self.rescue_min_fraud_lift,
        )

        near_constant_dropped = assessment.loc[
            assessment["decision"] == "drop", "column"
        ].tolist()
        rescued = assessment.loc[
            assessment["reason"].str.startswith("rescued"), "column"
        ].tolist()

        self._log(
            f"    {len(near_constant_dropped)} columns dropped as near-constant, "
            f"{len(rescued)} rescued because their rare values track fraud"
        )

        survivors = [c for c in remaining if c not in near_constant_dropped]

        # Record every drop with its evidence, so decisions are auditable.
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

        # --- 4. reduce the V columns using their blocks ------------------
        survivor_set = set(survivors)
        v_survivors = [
            column
            for group in (self.v_groups or [])
            for column in group
            if column in survivor_set
        ]
        non_v_survivors = [c for c in survivors if c not in set(v_survivors)]

        if v_survivors and self.v_groups:
            self._log(f"  Reducing {len(v_survivors)} surviving V columns ...")
            groups_after_pruning = [
                [column for column in group if column in survivor_set]
                for group in self.v_groups
            ]
            v_kept, v_detail = reduce_v_columns(
                frame,
                [group for group in groups_after_pruning if group],
                threshold=self.v_correlation_threshold,
                verbose=self.verbose,
            )
            self._log(f"    {len(v_survivors)} V columns -> {len(v_kept)} kept")
        else:
            v_kept, v_detail = [], pd.DataFrame()

        self.v_reduction_ = v_detail
        self.base_columns_ = non_v_survivors + v_kept

        # --- 5. work out which columns need text handling ----------------
        # Anything stored as a category, plus the identifier-style numbers
        # named in the frequency list, plus the columns the uid needs.
        category_like = [
            column
            for column in self.base_columns_
            if isinstance(frame[column].dtype, pd.CategoricalDtype)
        ]
        frequency_sources = [
            column for column in FREQUENCY_ENCODE_COLUMNS if column in self.base_columns_
        ]
        uid_sources = [
            column
            for column in ("card1", "addr1", "D1", "id_31", "DeviceInfo")
            if column in self.base_columns_
        ]

        self.label_source_columns_ = sorted(
            set(category_like) | set(frequency_sources) | set(uid_sources)
        )
        self.category_columns_ = category_like
        self.can_build_uid_ = all(
            column in self.base_columns_ for column in ("card1", "addr1", "D1")
        )
        if not self.can_build_uid_:
            self._log("    WARNING: uid cannot be built, a source column was pruned")

        # --- 6. learn the encodings --------------------------------------
        self._log("  Learning encodings ...")
        labels = self._build_labels(frame)

        # Frequency maps: the share of training rows holding each value.
        # A share rather than a raw count, so that a training set of 472,000
        # rows and a test set of 506,000 rows produce comparable numbers.
        self.frequency_maps_ = {}
        for column in FREQUENCY_ENCODE_COLUMNS:
            if column in labels.columns:
                self.frequency_maps_[column] = labels[column].value_counts(
                    normalize=True
                )
        self._log(f"    {len(self.frequency_maps_)} frequency maps")

        # Category maps: every text column gets an integer code. The blank
        # label gets a code like any other value, because "we do not know"
        # is real information. Values never seen in training get -1 at
        # transform time.
        self.category_maps_ = {}
        for column in self.category_columns_ + DERIVED_LABEL_COLUMNS:
            if column in labels.columns:
                distinct = sorted(labels[column].dropna().unique())
                self.category_maps_[column] = {
                    value: code for code, value in enumerate(distinct)
                }
        self._log(f"    {len(self.category_maps_)} category maps")

        # Aggregate maps: the average and spread of a value within each group.
        self.aggregate_maps_ = {}
        for group_column, value_column in AGGREGATION_SPECS:
            if group_column not in labels.columns:
                continue
            if value_column not in frame.columns:
                continue
            values = frame[value_column].astype("float64")
            summary = values.groupby(labels[group_column], observed=True).agg(
                ["mean", "std"]
            )
            self.aggregate_maps_[(group_column, value_column)] = summary
        self._log(f"    {len(self.aggregate_maps_)} aggregate maps")

        # --- 7. run one transform to fix the feature list ----------------
        # Running the real transform on the fit data is the only reliable
        # way to know exactly which columns come out and in what order.
        # Guessing the list separately is how the two drift apart.
        self.feature_tags_ = []
        self._recording = True
        produced = self._transform_frame(frame, labels)
        self._recording = False

        self.feature_names_ = list(produced.columns)
        self.manifest_ = pd.DataFrame(self.feature_tags_)

        self._log(f"  Fitted. {len(self.feature_names_)} features produced.")
        return self

    # -----------------------------------------------------
    # transform
    # -----------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply everything learned in fit. Safe on training, validation, or test."""
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("call fit before transform")

        frame = X.reset_index(drop=True)
        labels = self._build_labels(frame)
        produced = self._transform_frame(frame, labels)

        # Force the exact same columns in the exact same order as training.
        # A model does not read column names, it reads positions. A column
        # that arrives in a different place is silently the wrong number.
        return produced.reindex(columns=self.feature_names_)

    def _transform_frame(
        self, frame: pd.DataFrame, labels: pd.DataFrame
    ) -> pd.DataFrame:
        """Build the feature table. Shared by fit and transform, so they cannot differ."""
        pieces: list[pd.DataFrame] = []

        # --- numeric base columns, passed through as they are -------------
        numeric_base = [
            column
            for column in self.base_columns_
            if column in frame.columns and column not in self.category_columns_
        ]
        if numeric_base:
            pieces.append(frame[numeric_base])
            for column in numeric_base:
                self._tag(column, "base_numeric", column)

        # --- derived numeric features -------------------------------------
        time_features = build_time_features(frame)
        pieces.append(time_features)
        for column in time_features.columns:
            self._tag(column, "derived_time", "TransactionDT")

        amount_features = build_amount_features(frame)
        pieces.append(amount_features)
        for column in amount_features.columns:
            self._tag(column, "derived_amount", "TransactionAmt")

        match_features = build_match_features(frame, M_COLUMNS)
        if not match_features.empty:
            pieces.append(match_features)
            for column in match_features.columns:
                self._tag(column, "derived_match", "M1-M9")

        if "id_33" in labels.columns:
            screen_features = build_screen_features(labels["id_33"])
            pieces.append(screen_features)
            for column in screen_features.columns:
                self._tag(column, "derived_screen", "id_33")

        # Do the purchaser and recipient email domains match? A classic
        # fraud signal. Kept as three states: no, yes, and unknown.
        if "P_emaildomain" in labels.columns and "R_emaildomain" in labels.columns:
            both_known = (labels["P_emaildomain"] != MISSING_LABEL) & (
                labels["R_emaildomain"] != MISSING_LABEL
            )
            same = (labels["P_emaildomain"] == labels["R_emaildomain"]).astype("float32")
            pieces.append(
                pd.DataFrame(
                    {"email_domains_match": same.where(both_known, np.nan)},
                    index=frame.index,
                )
            )
            self._tag("email_domains_match", "derived_email", "P/R_emaildomain")

        # --- category codes -------------------------------------------------
        encoded = {}
        for column, mapping in self.category_maps_.items():
            if column not in labels.columns:
                continue
            name = f"{column}_code"
            encoded[name] = (
                labels[column]
                .map(mapping)
                .fillna(UNSEEN_CATEGORY_CODE)
                .astype("int32")
            )
            self._tag(name, "category_code", column)
        if encoded:
            pieces.append(pd.DataFrame(encoded, index=frame.index))

        # --- frequency counts -----------------------------------------------
        frequencies = {}
        for column, mapping in self.frequency_maps_.items():
            if column not in labels.columns:
                continue
            name = f"{column}_freq"
            # A value never seen in training gets 0. That is truthful:
            # as far as the training data knows, it does not exist.
            frequencies[name] = (
                labels[column].map(mapping).fillna(0.0).astype("float32")
            )
            self._tag(name, "frequency", column)
        if frequencies:
            pieces.append(pd.DataFrame(frequencies, index=frame.index))

        # --- group aggregates -------------------------------------------------
        aggregates = {}
        for (group_column, value_column), summary in self.aggregate_maps_.items():
            if group_column not in labels.columns or value_column not in frame.columns:
                continue

            keys = labels[group_column]
            values = frame[value_column].astype("float64")

            mean_name = f"{value_column}_mean_by_{group_column}"
            std_name = f"{value_column}_std_by_{group_column}"
            ratio_name = f"{value_column}_ratio_to_{group_column}_mean"

            group_mean = keys.map(summary["mean"])
            group_std = keys.map(summary["std"])

            aggregates[mean_name] = group_mean.astype("float32")
            aggregates[std_name] = group_std.astype("float32")

            # Dividing by zero gives infinity, which no model handles well.
            # Turn those into blanks so the tree routes them like any other
            # missing value.
            ratio = values / group_mean
            aggregates[ratio_name] = (
                ratio.replace([np.inf, -np.inf], np.nan).astype("float32")
            )

            for name in (mean_name, std_name, ratio_name):
                self._tag(name, "aggregate", f"{value_column} by {group_column}")

        if aggregates:
            pieces.append(pd.DataFrame(aggregates, index=frame.index))

        result = pd.concat(pieces, axis=1)

        # Guard against a duplicated column name, which would make the
        # feature list ambiguous and is easy to introduce by accident when
        # adding a new feature later.
        duplicated = result.columns[result.columns.duplicated()].tolist()
        if duplicated:
            raise ValueError(f"duplicate feature names produced: {duplicated}")

        return result
```

---

## 10. Create `src/pipelines/features.py`

### 10.1 What this stage does

1. Load the joined training table
2. Work out the time boundary and label every row `train` or `valid`
3. Fit the transformer on the `train` rows only
4. Transform the whole training table, and separately the test table
5. Run four verification checks
6. Save the two Parquet files, the fitted transformer, and three reports

### 10.2 The file

```python
"""
Feature engineering stage.

Input:  data/interim/train_joined.parquet
        data/interim/test_joined.parquet
        reports/v_column_missing_groups.csv
Output: data/processed/train_features.parquet
        data/processed/test_features.parquet
        models/feature_engineer.joblib
        reports/feature_manifest.csv
        reports/dropped_columns.csv
        reports/v_column_reduction.csv
        reports/feature_summary.md

Run with:
    python run.py --step features
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config.config import (
    DROPPED_COLUMNS_FILE,
    FEATURE_MANIFEST_FILE,
    FEATURE_SUMMARY_FILE,
    FEATURES_TEST_FILE,
    FEATURES_TRAIN_FILE,
    ID_COLUMN,
    JOINED_TEST_FILE,
    JOINED_TRAIN_FILE,
    PREPROCESSOR_FILE,
    REFERENCE_DATETIME,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_SPLIT_LABEL,
    V_GROUPS_FILE,
    V_REDUCTION_FILE,
    VALID_SPLIT_LABEL,
    VALIDATION_FRACTION,
    ensure_directories,
)
from src.features.engineer import FraudFeatureEngineer
from src.utils.column_selection import load_v_groups
from src.utils.memory_utils import memory_usage_mb


def _as_date(seconds: float) -> str:
    """Turn a TransactionDT value into a readable date, for reporting only."""
    reference = pd.Timestamp(REFERENCE_DATETIME)
    return (reference + pd.to_timedelta(int(seconds), unit="s")).date().isoformat()


def _assign_split(frame: pd.DataFrame) -> tuple[pd.Series, float]:
    """
    Label each row train or valid, cutting on time rather than at random.

    The boundary is the TransactionDT value at the 80th percentile. Rows at
    or below it are training, rows above it are validation.

    A random split would put January and June transactions on both sides,
    which lets the model see the future while learning. This dataset makes
    that leak especially bad: the same card appears many times over the six
    months, the D columns measure elapsed time from earlier events, and
    fraud arrives in bursts that a shuffle would scatter across both sides.
    """
    boundary = float(frame[TIME_COLUMN].quantile(1 - VALIDATION_FRACTION))
    labels = np.where(
        frame[TIME_COLUMN] <= boundary, TRAIN_SPLIT_LABEL, VALID_SPLIT_LABEL
    )
    return pd.Series(labels, index=frame.index, dtype="object"), boundary


def _verify(train_features: pd.DataFrame, test_features: pd.DataFrame) -> list[str]:
    """
    Four checks that catch the mistakes which do not raise an error.

    Each of these has silently ruined a model somewhere. None of them shows
    up as a crash, which is exactly why they are worth checking explicitly.
    """
    problems = []

    # 1. The answer must not be sitting in the features.
    if TARGET_COLUMN in train_features.columns:
        problems.append(f"{TARGET_COLUMN} is in the feature table")

    # 2. Train and test must have identical columns in identical order.
    #    Models read positions, not names.
    if list(train_features.columns) != list(test_features.columns):
        only_train = set(train_features.columns) - set(test_features.columns)
        only_test = set(test_features.columns) - set(train_features.columns)
        problems.append(
            f"feature columns differ. only in train: {sorted(only_train)[:5]}, "
            f"only in test: {sorted(only_test)[:5]}"
        )

    # 3. A column that is blank everywhere is dead weight.
    all_blank = [
        column for column in train_features.columns if train_features[column].isna().all()
    ]
    if all_blank:
        problems.append(f"{len(all_blank)} feature columns are entirely blank: {all_blank[:5]}")

    # 4. Infinity breaks most models and is easy to create by dividing.
    numeric = train_features.select_dtypes(include=[np.number])
    infinite = numeric.columns[np.isinf(numeric.to_numpy()).any(axis=0)].tolist()
    if infinite:
        problems.append(f"{len(infinite)} columns contain infinity: {infinite[:5]}")

    return problems


def _write_summary(results: dict) -> None:
    """Write the human-readable summary of what this stage did."""
    lines: list[str] = []
    add = lines.append

    add("# Feature Engineering Summary")
    add("")
    add("Generated automatically by `src/pipelines/features.py`. "
        "Do not edit by hand, it is overwritten on every run.")
    add("")

    add("## 1. Column reduction")
    add("")
    add("| Stage | Columns |")
    add("|-------|---------|")
    add(f"| Joined training table | {results['input_columns']} |")
    add(f"| Dropped: single value | {results['constant_dropped']} |")
    add(f"| Dropped: near-constant | {results['near_constant_dropped']} |")
    add(f"| Rescued from near-constant | {results['rescued']} |")
    add(f"| V columns before reduction | {results['v_before']} |")
    add(f"| V columns after reduction | {results['v_after']} |")
    add(f"| **Final feature count** | **{results['feature_count']}** |")
    add("")
    add("Every dropped column, with the evidence behind the decision, is in "
        "`reports/dropped_columns.csv`. The V column mapping is in "
        "`reports/v_column_reduction.csv`.")
    add("")

    add("## 2. Feature types")
    add("")
    add(results["kind_counts"].to_markdown(index=False))
    add("")

    add("## 3. The time split")
    add("")
    add("| Portion | Rows | Frauds | Fraud rate | First | Last |")
    add("|---------|------|--------|------------|-------|------|")
    for row in results["split_rows"]:
        add(
            f"| {row['portion']} | {row['rows']:,} | {row['frauds']:,} | "
            f"{row['fraud_rate']:.4%} | {row['start']} | {row['end']} |"
        )
    add("")
    add(f"The boundary sits at TransactionDT {results['boundary']:,.0f}, "
        f"which is {results['boundary_date']}.")
    add("")
    add("The transformer was fitted on the `train` portion only. The `valid` "
        "portion and the test set were transformed using what was learned "
        "there, and contributed nothing to it. Any frequency count or group "
        "average attached to a validation row was computed without that row.")
    add("")

    add("## 4. Test set")
    add("")
    add(f"- Rows: **{results['test_rows']:,}**")
    add(f"- Features: **{results['feature_count']}**, identical to training "
        "and in the same order")
    add(f"- Values never seen during training, across all counted columns: "
        f"**{results['unseen_share']:.2%}** of lookups returned zero")
    add("")

    add("## 5. Verification")
    add("")
    if results["problems"]:
        for problem in results["problems"]:
            add(f"- PROBLEM: {problem}")
    else:
        add("- The target is not present in the feature table")
        add("- Training and test features match exactly, in name and order")
        add("- No feature column is blank on every row")
        add("- No feature column contains infinity")
    add("")

    add("## 6. Carried into Step 4")
    add("")
    add("1. Read the `split` column rather than recomputing the split, so "
        "every experiment is scored on exactly the same rows.")
    add("2. `TransactionID` and `TransactionDT` are present in the files but "
        "are not features. Drop them before training.")
    add("3. Load `models/feature_engineer.joblib` for scoring, never rebuild "
        "the transformations by hand.")
    add("4. PR-AUC is primary, baseline 0.035. ROC-AUC secondary. Recall at a "
        "1% review rate is the business headline.")
    add("")

    FEATURE_SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {FEATURE_SUMMARY_FILE.name}")


def run_features() -> dict:
    """Run the whole feature engineering stage."""
    print("=" * 60)
    print("STAGE: FEATURE ENGINEERING")
    print("=" * 60)

    ensure_directories()

    if not JOINED_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"{JOINED_TRAIN_FILE} not found.\n"
            f"Run  python run.py --step ingestion  first."
        )
    if not V_GROUPS_FILE.exists():
        raise FileNotFoundError(
            f"{V_GROUPS_FILE} not found.\n"
            f"Run  python run.py --step eda  first."
        )

    # --- load and split ----------------------------------------------------
    print(f"\n  Loading {JOINED_TRAIN_FILE.name} ...")
    train = pd.read_parquet(JOINED_TRAIN_FILE)
    print(
        f"    {train.shape[0]:,} rows x {train.shape[1]} columns, "
        f"{memory_usage_mb(train):,.1f} MB"
    )

    split_labels, boundary = _assign_split(train)
    fit_mask = split_labels == TRAIN_SPLIT_LABEL

    print(f"\n  Time split at TransactionDT {boundary:,.0f} ({_as_date(boundary)})")
    print(f"    train portion: {int(fit_mask.sum()):,} rows")
    print(f"    valid portion: {int((~fit_mask).sum()):,} rows")

    # --- fit on the training portion only ----------------------------------
    print("\n  Fitting the feature engineer on the train portion only ...")
    engineer = FraudFeatureEngineer(v_groups=load_v_groups(V_GROUPS_FILE))
    engineer.fit(train.loc[fit_mask], train.loc[fit_mask, TARGET_COLUMN])

    # --- transform training -------------------------------------------------
    print("\n  Transforming the training table ...")
    train_features = engineer.transform(train)
    train_features[ID_COLUMN] = train[ID_COLUMN].to_numpy()
    train_features[TIME_COLUMN] = train[TIME_COLUMN].to_numpy()
    train_features[TARGET_COLUMN] = train[TARGET_COLUMN].to_numpy()
    train_features[SPLIT_COLUMN] = split_labels.to_numpy()
    print(
        f"    {train_features.shape[0]:,} rows x "
        f"{len(engineer.feature_names_)} features"
    )

    # Statistics needed for the report, gathered before the raw table goes.
    split_rows = []
    for portion in (TRAIN_SPLIT_LABEL, VALID_SPLIT_LABEL):
        mask = split_labels == portion
        subset_target = train.loc[mask, TARGET_COLUMN]
        split_rows.append(
            {
                "portion": portion,
                "rows": int(mask.sum()),
                "frauds": int(subset_target.sum()),
                "fraud_rate": float(subset_target.mean()),
                "start": _as_date(train.loc[mask, TIME_COLUMN].min()),
                "end": _as_date(train.loc[mask, TIME_COLUMN].max()),
            }
        )

    input_columns = train.shape[1]
    del train

    print(f"\n  Writing {FEATURES_TRAIN_FILE.name} ...")
    train_features.to_parquet(FEATURES_TRAIN_FILE, index=False, engine="pyarrow")
    print(f"    {FEATURES_TRAIN_FILE.stat().st_size / 1024 ** 2:,.1f} MB on disk")

    # --- transform test -----------------------------------------------------
    print(f"\n  Loading {JOINED_TEST_FILE.name} ...")
    test = pd.read_parquet(JOINED_TEST_FILE)
    print(f"    {test.shape[0]:,} rows x {test.shape[1]} columns")

    print("  Transforming the test table ...")
    test_features = engineer.transform(test)
    test_features[ID_COLUMN] = test[ID_COLUMN].to_numpy()
    test_features[TIME_COLUMN] = test[TIME_COLUMN].to_numpy()

    test_rows = len(test)
    del test

    # How much of the test set holds values that training never saw. A high
    # number here is an early warning that the two periods differ, which
    # matters for the drift work in Step 5.
    frequency_columns = [
        f"{column}_freq"
        for column in engineer.frequency_maps_
        if f"{column}_freq" in test_features.columns
    ]
    if frequency_columns:
        unseen_share = float(
            (test_features[frequency_columns] == 0).to_numpy().mean()
        )
    else:
        unseen_share = 0.0
    print(f"    unseen-value lookups in test: {unseen_share:.2%}")

    print(f"\n  Writing {FEATURES_TEST_FILE.name} ...")
    test_features.to_parquet(FEATURES_TEST_FILE, index=False, engine="pyarrow")
    print(f"    {FEATURES_TEST_FILE.stat().st_size / 1024 ** 2:,.1f} MB on disk")

    # --- verify --------------------------------------------------------------
    print("\n  Verifying ...")
    feature_only_train = train_features[engineer.feature_names_]
    feature_only_test = test_features[engineer.feature_names_]
    problems = _verify(feature_only_train, feature_only_test)

    if problems:
        for problem in problems:
            print(f"    PROBLEM: {problem}")
    else:
        print("    all four checks passed")

    # --- save the fitted transformer -------------------------------------------
    print(f"\n  Saving {PREPROCESSOR_FILE.name} ...")
    joblib.dump(engineer, PREPROCESSOR_FILE)
    print(f"    {PREPROCESSOR_FILE.stat().st_size / 1024 ** 2:,.1f} MB on disk")

    # --- reports -----------------------------------------------------------------
    manifest = engineer.manifest_.copy()
    manifest["missing_pct_train"] = [
        round(float(feature_only_train[name].isna().mean()) * 100, 2)
        for name in manifest["feature"]
    ]
    manifest["missing_pct_test"] = [
        round(float(feature_only_test[name].isna().mean()) * 100, 2)
        for name in manifest["feature"]
    ]
    manifest.to_csv(FEATURE_MANIFEST_FILE, index=False)
    print(f"  Wrote {FEATURE_MANIFEST_FILE.name}")

    engineer.column_decisions_.to_csv(DROPPED_COLUMNS_FILE, index=False)
    print(f"  Wrote {DROPPED_COLUMNS_FILE.name}")

    if not engineer.v_reduction_.empty:
        engineer.v_reduction_.to_csv(V_REDUCTION_FILE, index=False)
        print(f"  Wrote {V_REDUCTION_FILE.name}")

    decisions = engineer.column_decisions_
    v_after = int(engineer.v_reduction_["kept_column"].nunique()) if not engineer.v_reduction_.empty else 0
    v_before = (
        int(engineer.v_reduction_["n_represented"].sum())
        if not engineer.v_reduction_.empty
        else 0
    )

    results = {
        "input_columns": input_columns,
        "constant_dropped": int((decisions["reason"] == "single distinct value").sum()),
        "near_constant_dropped": int(
            decisions["reason"].str.startswith("near-constant").sum()
        ),
        "rescued": int(decisions["reason"].str.startswith("rescued").sum()),
        "v_before": v_before,
        "v_after": v_after,
        "feature_count": len(engineer.feature_names_),
        "kind_counts": (
            manifest.groupby("kind", observed=True)
            .size()
            .reset_index(name="features")
            .sort_values("features", ascending=False)
        ),
        "split_rows": split_rows,
        "boundary": boundary,
        "boundary_date": _as_date(boundary),
        "test_rows": test_rows,
        "unseen_share": unseen_share,
        "problems": problems,
    }
    _write_summary(results)

    # --- headline ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING HEADLINES")
    print("=" * 60)
    print(f"  Input columns         : {input_columns}")
    print(f"  Final features        : {results['feature_count']}")
    print(f"  V columns             : {v_before} -> {v_after}")
    print(f"  Dropped, single value : {results['constant_dropped']}")
    print(f"  Dropped, near-constant: {results['near_constant_dropped']}")
    print(f"  Rescued               : {results['rescued']}")
    print(f"  Split boundary        : {results['boundary_date']}")
    print(f"  Train / valid rows    : {split_rows[0]['rows']:,} / {split_rows[1]['rows']:,}")
    print(f"  Verification          : {'PASSED' if not problems else 'SEE PROBLEMS ABOVE'}")
    print(f"\n  Full report: {FEATURE_SUMMARY_FILE}")

    return results
```

---

## 11. Update `run.py`

Two small changes. Add a function for the new stage, and add it to the choices.

**Add this function** below `run_eda_stage`:

```python
def run_features_stage(args: argparse.Namespace) -> dict:
    from src.pipelines.features import run_features

    return run_features()
```

**Change the `--step` choices line** from:

```python
        choices=["ingestion", "eda", "all"],
```

to:

```python
        choices=["ingestion", "eda", "features", "all"],
```

**Change the dispatch block** at the bottom of `main` from:

```python
    if args.step == "ingestion":
        run_ingestion_stage(args)
    elif args.step == "eda":
        run_eda_stage(args)
    elif args.step == "all":
        run_ingestion_stage(args)
        run_eda_stage(args)
```

to:

```python
    if args.step == "ingestion":
        run_ingestion_stage(args)
    elif args.step == "eda":
        run_eda_stage(args)
    elif args.step == "features":
        run_features_stage(args)
    elif args.step == "all":
        run_ingestion_stage(args)
        run_eda_stage(args)
        run_features_stage(args)
```

**Also update the docstring** at the top of `run.py` to mention the new stage:

```python
Stages available so far:
    ingestion   Load raw CSVs, join transaction to identity, save Parquet
    eda         Profile the joined training data, write reports and charts
    features    Prune columns, build features, split by time, save processed data
    all         Every stage above, in order
```

---

## 12. Run it

### 12.1 Start the branch

```powershell
git switch main
git pull
git switch -c step-03-features
git branch
```

### 12.2 Quick import check before the real run

Catching a typo now is far cheaper than catching it eight minutes into a run.

```powershell
python -c "from src.features.engineer import FraudFeatureEngineer; from src.pipelines.features import run_features; print('imports OK')"
```

**Expected output:** `imports OK`

### 12.3 The real run

```powershell
python run.py --step features
```

Expect **8 to 15 minutes**. The slow parts are the correlation work inside the V blocks and building the text label columns.

Peak memory will be around 5 to 7 GB. You have 32 GB, so it is fine, but close anything large if your machine is busy.

**Expected output, abbreviated:**

```
============================================================
STAGE: FEATURE ENGINEERING
============================================================

  Loading train_joined.parquet ...
    590,540 rows x 435 columns, 927.2 MB

  Time split at TransactionDT 12,xxx,xxx (2018-0x-xx)
    train portion: 472,4xx rows
    valid portion: 118,1xx rows

  Fitting the feature engineer on the train portion only ...
  Selecting columns ...
    x columns hold a single value
    xx columns dropped as near-constant, x rescued because their rare values track fraud
  Reducing xxx surviving V columns ...
    block  1:  46 columns ->  xx kept
    ... one line per block ...
    xxx V columns -> xxx kept
  Learning encodings ...
    18 frequency maps
    xx category maps
    6 aggregate maps
  Fitted. xxx features produced.

  Transforming the training table ...
  Writing train_features.parquet ...

  Loading test_joined.parquet ...
  Transforming the test table ...
    unseen-value lookups in test: xx.xx%

  Verifying ...
    all four checks passed

  Saving feature_engineer.joblib ...
  Wrote feature_manifest.csv
  Wrote dropped_columns.csv
  Wrote v_column_reduction.csv
  Wrote feature_summary.md
```

### 12.4 Confirm the outputs

```powershell
Get-ChildItem data\processed | Select-Object Name, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,1)}}
Get-ChildItem models | Select-Object Name
Get-ChildItem reports -File | Select-Object Name
```

### 12.5 Prove the saved transformer actually works

This is the check that matters most, because it is the thing Step 6 depends on. Load the saved object in a fresh Python process and transform a handful of rows with it.

```powershell
python -c "import joblib, pandas as pd; from config.config import PREPROCESSOR_FILE, JOINED_TEST_FILE; e = joblib.load(PREPROCESSOR_FILE); rows = pd.read_parquet(JOINED_TEST_FILE).head(5); out = e.transform(rows); print('features:', out.shape); print('matches training list:', list(out.columns) == e.feature_names_)"
```

**Expected output:** a shape of 5 rows by your feature count, and `matches training list: True`.

If that works, the transformer survives being saved and reloaded in a different process, which means the API in Step 6 will work.

---

## 13. Reading your results

### 13.1 The five things to check

| Check | What good looks like | If it is off |
|-------|----------------------|--------------|
| Verification | All four checks passed | Send me the problem text. Do not go to Step 4 with a failing check. |
| V reduction | 339 down to somewhere around 100 to 170 | Below 60 means the threshold is too aggressive. Above 250 means the V columns are less redundant than expected. Either way, tell me the number. |
| Split rows | Roughly 472,000 and 118,000 | A big imbalance means `VALIDATION_FRACTION` is not 0.2. |
| Validation fraud rate | In the same neighbourhood as 3.5%, but not identical | A time split does not preserve the fraud rate, and it should not. If the last five weeks were riskier, the validation rate is higher. That is real information. |
| Unseen lookups in test | Some meaningful percentage | Near 0% would be suspicious, suggesting the counting is not working. A large number is expected and normal, because many cards in the test period never appeared in training. |

### 13.2 What each report is for

**`reports/feature_summary.md`.** Read this first. It is the narrative of what happened.

**`reports/dropped_columns.csv`.** Open it in Excel and sort by `fraud_lift` descending. This shows you every near-constant column and how unusual its rare values were. Any column with a lift above 2 was rescued. Skim the ones just below the cutoff: if something at 1.9 looks important to you, we can lower the threshold and re-run. This file is your defence when someone asks why a column was removed.

**`reports/v_column_reduction.csv`.** Each row is one cluster: which column was kept and which ones it stands for. If a kept column later turns out to be important in Step 4, this file tells you which other columns were saying the same thing.

**`reports/feature_manifest.csv`.** The dictionary for the whole feature table. Sort by `kind` to see the balance between raw columns, counts, aggregates, and derived features. Sort by `missing_pct_test` descending and compare against `missing_pct_train`: a feature that is 10% blank in training and 60% blank in test is a drift warning worth knowing about before Step 5.

### 13.3 One thing to look at specifically

In `feature_manifest.csv`, compare `missing_pct_train` against `missing_pct_test` for the identity-derived features. Your identity coverage rose from 24.4% in training to 28.0% in test, so those features should be **less** blank in test than in training.

That is the drift signal from Section 2.5 showing up in the feature table, where it can actually be measured. Note the size of the gap now. Step 5 builds the monitoring that would catch this automatically.

---

## 14. A small patch to the Step 2 report

Section 2.3 showed that the "3.75x" line in `reports/eda_summary.md` is misleading. The report is regenerated every time the EDA stage runs, so the fix belongs in the code that writes it.

Open `src/pipelines/eda.py` and find this in `_write_summary`:

```python
    add(f"Fraud is **{lift:.2f}x** as likely among transactions that have an "
        "identity record. The presence or absence of that record is therefore "
        "informative in itself, which is why `has_identity` is kept as a feature.")
```

Replace it with:

```python
    add(f"Fraud is **{lift:.2f}x** as likely among transactions that have an "
        "identity record. Read that figure carefully. The table below shows "
        "that identity coverage is almost entirely decided by `ProductCD`: "
        "product W never has an identity record, and every other product "
        "almost always does. Since W also has the lowest fraud rate and makes "
        "up most of the data, the bulk of this gap is a product effect rather "
        "than an identity effect. Restricted to the non-W products, where the "
        "flag actually varies, the lift is closer to 1.4x. `has_identity` is "
        "kept as a feature, but it is expected to rank low.")
```

Then regenerate the report so it carries the corrected wording:

```powershell
python run.py --step eda
```

That takes about 30 seconds and rewrites `reports/eda_summary.md` and the figures.

Why bother: the EDA report is a document you will show people. A confident number with a hidden caveat is worse than no number, because the reader has no way to know it needs one.

---

## 15. Data version control with DVC

This closes open question Q-03. It is a separate concern from feature engineering, so it lives in its own section. You can do it now or defer it to Step 5, but doing it now means the processed data is versioned from the moment it exists.

### 15.1 What DVC does, and why Git cannot

Git is built for text. It stores every version of every file forever, and it compares them line by line. That works beautifully for code and terribly for a 100 MB Parquet file, where a small change rewrites the whole binary and Git has no way to store just the difference.

DVC handles this by splitting the job in two. The actual data file goes into a storage location outside the repository. What goes **into** Git is a tiny text file, a few lines long, holding a fingerprint of the data. Git versions the fingerprint. DVC uses the fingerprint to fetch the matching data.

The result is that `git checkout` of an old commit followed by `dvc checkout` gives you the exact data that went with that code. That is genuine reproducibility, and it is one of the clearest signals that a project is engineered rather than assembled.

### 15.2 Install it

```powershell
pip install "dvc>=3.55"
```

Add it to `requirements-dev.txt`, under the data acquisition heading:

```text
dvc>=3.55                # versions data files alongside the code that made them
```

Then refresh the lock file, since your environment has changed:

```powershell
pip freeze > requirements.lock.txt
```

### 15.3 The `.gitignore` problem, which will bite you

DVC refuses to track a file that is already listed in `.gitignore`. It reports something like `output is already ignored`. This is deliberate on DVC's part: it wants to manage the ignoring itself, and having two systems ignoring the same path leads to confusing behaviour.

Your `.gitignore` from Step 1 contains this:

```text
data/processed/*
!data/processed/.gitkeep
```

Those two lines have to go, because DVC is taking over that folder. Everything else in the data section stays: raw data is downloaded from Kaggle and interim data is rebuilt in seconds, so neither needs versioning.

**Edit `.gitignore`** and delete exactly those two lines. Leave `data/raw/*`, `data/interim/*`, `data/external/*` and their `.gitkeep` exceptions alone.

When you run `dvc add`, DVC writes its own `data/processed/.gitignore` covering the files it now manages.

### 15.4 Set it up

```powershell
# Start DVC in this repository. Adds a .dvc folder and a .dvcignore file.
dvc init

# Create a storage folder OUTSIDE the project, so it is not inside Git's view
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\dvcstore" | Out-Null

# Register it as the default remote. "-d" means default.
dvc remote add -d localstore "$env:USERPROFILE\dvcstore"

# Confirm
dvc remote list
```

**Expected output:** `localstore  C:\Users\Dauda Agbonoga\dvcstore`

### 15.5 Track the processed data

```powershell
# Hand the two feature tables to DVC
dvc add data/processed/train_features.parquet
dvc add data/processed/test_features.parquet

# Look at what DVC created: two small text files, a few hundred bytes each
Get-ChildItem data\processed | Select-Object Name, Length
```

You should now see `train_features.parquet.dvc` and `test_features.parquet.dvc` next to the real files. Open one in VS Code. It holds a hash, a size, and a path. That is the whole thing, and that is what goes into Git.

```powershell
# Commit the pointers, not the data
git add data/processed/*.dvc data/processed/.gitignore .dvc/config .dvcignore .gitignore requirements-dev.txt requirements.lock.txt
git commit -m "chore: track processed feature tables with dvc"

# Copy the actual data into the storage folder
dvc push
```

### 15.6 Prove it works

```powershell
# Delete a feature table
Remove-Item data\processed\train_features.parquet

# Bring it back from DVC storage
dvc pull

# Confirm it is there again
Get-ChildItem data\processed\train_features.parquet
```

If the file comes back, DVC is working.

### 15.7 Moving to cloud storage later

Nothing about the project changes when you switch. One command points DVC somewhere else, then you push again:

```powershell
# Example only, do not run this now
dvc remote add -d s3store s3://your-bucket/dvcstore
dvc push
```

The `.dvc` pointer files, the code, and the workflow all stay exactly the same. That is the point of the design.

---

## 16. The updated README

You asked where the "Exploratory analysis" section belongs, and you asked for the project to stand on its own rather than being described as a follow-on to something else. Both are handled below.

**On placement:** findings belong immediately after the dataset description and before the technical material. A reader's questions arrive in a fixed order. What problem is this? What data? What did you find in it? How did you build it? How do I run it? Putting findings after Quickstart, where it was, interrupts someone who is trying to run the thing. So the new order is Problem, Dataset, What the data shows, Approach, Quickstart, Pipeline, Roadmap, Results, Tech stack, Licence.

I have also renamed it "What the data shows", because "Exploratory analysis" names the activity while the reader wants to know the payoff.

**Other changes:** the placeholder GitHub username is replaced with the real one, the status line is updated, the PR-AUC baseline is filled in with the real 0.035, the identity coverage figure is corrected with the caveat from Section 2.3, a pipeline table is added so the stages are visible at a glance, and the background section is rewritten to describe this project on its own terms.

**Replace the entire contents of `README.md`** with this.

````markdown
# IEEE-CIS Fraud Detection

An end-to-end machine learning and MLOps project that detects fraudulent card
transactions, covering the full lifecycle from raw data to a monitored,
containerised, deployed service with an interactive dashboard.

> Status: in progress. Steps 1 to 3 of 7 complete.

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

The goal is a model that ranks transactions by risk well enough to be useful at
a realistic review budget, plus the engineering around it that makes it
deployable, observable, and maintainable.

## Dataset

IEEE-CIS Fraud Detection (Kaggle competition, data provided by Vesta
Corporation).

| Table | Rows | Columns | Contents |
|-------|------|---------|----------|
| `train_transaction` | 590,540 | 394 | Transaction level, carries the `isFraud` label |
| `train_identity` | 144,233 | 41 | Device and network signals, only for some transactions |
| `test_transaction` | 506,691 | 393 | No label |
| `test_identity` | 141,907 | 41 | No label |

The two tables join on `TransactionID`. Most columns are anonymised: 339 of them
are engineered features supplied by Vesta with no published meaning, and the
identity columns are similarly masked.

Data is not stored in this repository. See Quickstart to download it.

## What the data shows

Full findings in [`reports/eda_summary.md`](reports/eda_summary.md), with charts
in [`reports/figures/`](reports/figures/).

![Class balance](reports/figures/01_class_balance.png)

**The imbalance.** 20,663 fraudulent transactions out of 590,540, a rate of
3.4990%, roughly one in twenty-nine.

**Time matters more than anything else.** Training covers 2017-12-01 to
2018-05-31. The test set covers 2018-07-01 to 2018-12-30. There is a deliberate
30 day gap between them. The test set is entirely in the future, so validation
here is a time-based split, never a random one. A random split would let the
model learn from transactions that happened after the ones it is scored on.

**Fraud concentrates in identifiable places.** Product code C runs at 11.69%
against product W at 2.04%. Credit cards run at 6.68% against debit at 2.43%.
Mobile devices run at 10.17% against desktop at 6.52%. Some email domains run
above 18% while others sit below 2.5%.

**Identity records are almost decided by product type.** Only 24.4% of
transactions have a matching identity record, and the raw figures suggest fraud
is 3.75 times as likely among those that do. That comparison is confounded:
product W never produces an identity record and also has the lowest fraud rate,
while every other product almost always produces one. Restricted to the products
where the flag actually varies, the difference is closer to 1.4x.

**The 339 anonymous V columns have hidden structure.** They fall into 15 blocks
that go blank on exactly the same rows, which is the fingerprint of features
built in batches from shared source data. Eight of the fifteen blocks interleave
through each other's number ranges, so the structure is invisible unless you
compare the actual missing patterns. Correlation clustering inside each block
reduces the 339 columns substantially without discarding them arbitrarily.

## Approach

- **Metric.** PR-AUC is primary, with a baseline of 0.035 equal to the fraud
  rate. ROC-AUC is reported alongside it. Recall at a 1% manual review rate is
  the headline business figure. Accuracy is not reported.
- **Validation.** A time-based split: the last 20% of the training period by
  `TransactionDT`.
- **Missing values are left missing.** LightGBM, XGBoost, and CatBoost all learn
  a direction for blanks at every split. Filling them with an average would
  assert something untrue.
- **No leakage by construction.** Every learned transformation is fitted on the
  training portion only and saved as an object, so training and serving cannot
  drift apart.

## Architecture

_Diagram added in Step 6._

## Quickstart

```bash
# 1. Clone
git clone https://github.com/Dee-ui/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection

# 2. Environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS or Linux
pip install -r requirements.lock.txt

# 3. Data (requires a Kaggle account that has joined the competition)
kaggle auth login
python scripts/download_data.py
python scripts/verify_data.py

# 4. Build everything
python run.py --step all
```

## Pipeline

| Stage | Command | Input | Output |
|-------|---------|-------|--------|
| Ingestion | `python run.py --step ingestion` | Raw CSVs | `data/interim/*_joined.parquet` |
| EDA | `python run.py --step eda` | Joined Parquet | `reports/eda_summary.md`, charts |
| Features | `python run.py --step features` | Joined Parquet | `data/processed/*_features.parquet`, `models/feature_engineer.joblib` |

Every stage reads a file and writes a file, so any one of them can be run and
debugged on its own.

## Project structure

See [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for the annotated structure,
the full decision log, and current status.

## Roadmap

- [x] Step 1: Dataset acquisition, scaffold, repo, environment
- [x] Step 2: Exploratory data analysis and data understanding
- [x] Step 3: Feature engineering and preprocessing pipeline
- [ ] Step 4: Model training with MLflow experiment tracking
- [ ] Step 5: MLOps layer: CI/CD, testing, model registry, drift monitoring
- [ ] Step 6: Dockerisation and deployment
- [ ] Step 7: Dashboard and portfolio packaging

## Results

_Populated in Step 4._

| Metric | Baseline | Best model |
|--------|----------|------------|
| PR-AUC | 0.035 | TBD |
| ROC-AUC | 0.500 | TBD |
| Recall at 1% review rate | 0.010 | TBD |

The baselines are what random guessing achieves. PR-AUC for a random model
equals the fraud rate. ROC-AUC for a random model is 0.5. Reviewing a random 1%
of transactions catches 1% of fraud.

## Tech stack

Python 3.11, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, MLflow, SHAP,
DVC, FastAPI, Docker, GitHub Actions, Streamlit.

## Licence

MIT. See [`LICENSE`](LICENSE).
````

---

## 17. Commit, merge, tag

```powershell
# Configuration
git add config/config.py
git commit -m "feat: extend config with feature engineering settings"

# Column selection and helper functions
git add src/utils/column_selection.py src/utils/feature_utils.py
git commit -m "feat: add column pruning, v-block reduction, and feature helpers"

# The transformer
git add src/features/
git commit -m "feat: add FraudFeatureEngineer, a fitted sklearn transformer"

# The pipeline stage and entry point
git add src/pipelines/features.py run.py
git commit -m "feat: add feature engineering stage with time-based split"

# The EDA report correction
git add src/pipelines/eda.py reports/eda_summary.md reports/figures/
git commit -m "fix: note the ProductCD confound in the identity coverage finding"

# Generated reports
git add reports/feature_manifest.csv reports/dropped_columns.csv reports/v_column_reduction.csv reports/feature_summary.md
git commit -m "docs: add feature manifest and column decision reports"

# README and step documentation
git add README.md docs/
git commit -m "docs: rewrite readme and add step 3 guide"

git log --oneline
```

DVC changes were committed separately in Section 15.5. If you skipped that section, skip that commit too.

```powershell
git push -u origin step-03-features

gh pr create --base main --head step-03-features `
  --title "Step 3: feature engineering and preprocessing" `
  --body "Column pruning with a fraud-lift rescue rule, V-column reduction using the 15 missing-pattern blocks, engineered time, amount, email, device, and aggregate features, a saved scikit-learn transformer, the time-based split, and DVC tracking of processed data."

gh pr merge --squash --delete-branch

git switch main
git pull
git tag -a v0.3.0-step3 -m "Step 3 complete: feature engineering and preprocessing"
git push origin v0.3.0-step3
```

---

## 18. Verification checklist

**Setup**
- [ ] Branch `step-03-features` created
- [ ] `config/config.py` extended, the check prints `18 6`
- [ ] `src/features/` package created with `__init__.py`
- [ ] All four new code files created
- [ ] `run.py` updated with the `features` step
- [ ] The import check prints `imports OK`

**Run**
- [ ] `python run.py --step features` completed
- [ ] All four verification checks passed
- [ ] Both Parquet files exist in `data/processed`
- [ ] `models/feature_engineer.joblib` exists
- [ ] Four report files written
- [ ] The reload check prints `matches training list: True`

**Sanity**
- [ ] Split is roughly 472,000 train and 118,000 valid
- [ ] V columns reduced to somewhere in the 100 to 170 range
- [ ] Train and test have identical feature columns
- [ ] `dropped_columns.csv` opens and the decisions look reasonable

**Corrections**
- [ ] `src/pipelines/eda.py` patched with the ProductCD caveat
- [ ] EDA re-run so `eda_summary.md` carries the corrected wording

**DVC**, if you did Section 15
- [ ] `dvc remote list` shows `localstore`
- [ ] Two `.dvc` pointer files exist in `data/processed`
- [ ] `dvc push` succeeded
- [ ] Delete and `dvc pull` restored the file

**Git**
- [ ] `git status` shows no Parquet or joblib files
- [ ] README replaced
- [ ] Branch pushed, pull request merged
- [ ] Tag `v0.3.0-step3` pushed

---

## 19. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'src.features'` | The `__init__.py` file is missing | `New-Item -ItemType File -Force -Path "src\features\__init__.py"` |
| `ValueError: fit needs the target` | `fit` called without `y` | The pipeline passes it. If you called `fit` yourself, pass the label column too. |
| `KeyError: cannot build uid, missing columns` | `card1`, `addr1`, or `D1` was pruned | Should not happen with these thresholds. Send me the `dropped_columns.csv` rows for those three. |
| `MemoryError` during the transform | Another large program is holding memory | Close it and re-run. Peak here is 5 to 7 GB. |
| `duplicate feature names produced` | Two features ended up with the same name | Send me the names, it is a naming collision in the code. |
| Verification reports columns only in train | A category value exists in one file and not the other | Should be impossible, `reindex` forces the same list. Send me the message. |
| Verification reports infinity | A division produced infinity that was not caught | Send me the column names. |
| V reduction keeps almost everything | Correlation threshold too high, or the blocks are genuinely varied | Try `V_CORRELATION_THRESHOLD = 0.6` and compare. Tell me both numbers. |
| V reduction keeps almost nothing | Threshold too low | Raise it towards 0.9 and compare. |
| `dvc add` says the output is already ignored | The `data/processed/*` lines are still in `.gitignore` | Section 15.3. |
| `dvc push` fails with a path error | The remote folder does not exist | Re-run the `New-Item` command from Section 15.4. |
| The stage takes over 25 minutes | Correlation work on large blocks is slow on a busy machine | Tell me the timing. Block 2 with 43 columns over 590,000 rows is the heaviest. |
| `joblib.load` fails in a new terminal | The environment is not activated | `.\.venv\Scripts\Activate.ps1` |

---

## 20. Your Streamlit dashboard question, recorded for Step 7

You asked whether the dashboard charts should run on training data or on the
data being scored during use. That is exactly the right question to ask early,
because it changes what Step 5 has to produce.

**The short answer: both, in separate places, and they should not be mixed.**

A production fraud dashboard normally has three distinct areas, and they draw
from different sources for different reasons.

**One: the model profile.** What the model is and what it learned from. Class
balance, feature importance, the validation scores, the precision and recall
curve. This is **static**. It describes a specific trained model version and it
does not change until you retrain. It should be built once at training time and
saved as small artifacts, then simply displayed. Most of your current EDA charts
belong here.

**Two: live operations.** What is happening right now in the data being scored.
Transaction volume, the distribution of risk scores, how many crossed the alert
threshold, the queue for manual review. This draws from **scored production
data** and refreshes constantly. In this project, the test set stands in for
production data, which is one more reason Step 2 processed it.

**Three: drift monitoring.** This is the only place the two sources meet, and
they meet deliberately: the training distribution as a fixed reference line, the
recent scored data overlaid on top of it. That comparison is the whole point of
the chart. Your identity coverage moving from 24.4% to 28.0% is exactly the kind
of thing that shows up here.

A fourth area is usually worth adding: a single-transaction explainer, where you
paste in one transaction and get its score plus the SHAP breakdown of what drove
it. It is the most persuasive thing to demonstrate live, because it turns an
abstract model into a specific answerable question.

**One engineering rule that matters more than it sounds.** Do not have the
dashboard compute charts from the raw 590,540 row table when a page loads.
Precompute the aggregates into small files, a few hundred kilobytes each, and
have Streamlit read those. Otherwise every page load takes fifteen seconds, and
a slow dashboard reads as a broken dashboard.

This is now recorded as decision D-33 in `PROJECT_STATE.md`, and Step 5 will be
built with it in mind so the monitoring outputs are already in the shape the
dashboard needs. Open question Q-12 tracks the one thing I still need from you
before Step 7: whether you want the dashboard aimed at a fraud analyst working a
review queue, or at a manager watching overall performance. They lead to
genuinely different layouts.

---

## 21. What to send me before Step 4

1. **The full terminal output** of `python run.py --step features`
2. **The contents of `reports/feature_summary.md`**
3. **`reports/feature_manifest.csv`** as an attachment. This determines what
   Step 4 trains on, and I want the real feature list rather than my estimate
   of it.
4. **`reports/dropped_columns.csv`** as an attachment, so we can review the
   pruning decisions together before building on them.
5. **Whether you did the DVC section**, and if so whether `dvc pull` restored
   the deleted file
6. **Any checklist item that did not tick**, with the error text
7. **Q-04:** where should the service be deployed in Step 6? Render or Railway
   free tier, Hugging Face Spaces, a cloud provider, or local Docker only.
8. **Q-05:** do you want a Kaggle late submission in Step 4? The competition is
   closed but late submissions still score, which would give you an external,
   independently verified number for the README. It costs about twenty minutes.
9. **Q-07:** does the project manager have any cost figures, even rough ones,
   for a missed fraud and for a false alarm? With them, threshold selection in
   Step 4 becomes a cost optimisation with a currency answer instead of a
   statistical exercise. That is a much stronger story on the PM track.
10. **Q-12:** who is the Step 7 dashboard for, an analyst or a manager?

Questions 7 to 10 are not blocking. Step 4 can proceed without them, but 9 in
particular changes what Step 4 produces, so it is worth asking your PM now.

---

## 22. What Step 4 covers

- Loading the processed features and reading the `split` column
- A baseline model first, so every later number has something to be compared
  against
- LightGBM, XGBoost, and CatBoost, trained and compared on identical splits
- MLflow set up from scratch, targeting the 3.x API since you have 3.15.1: what
  a run is, what a parameter is, what an artifact is, and how to read the UI
- Every run logged: settings, metrics, the model file, the feature list
- PR-AUC, ROC-AUC, and recall at 1% and 5% review rates, computed properly
- Threshold selection tied to a review capacity rather than left at 0.5
- Time-aware cross-validation, where each fold trains on an expanding window of
  the past and validates on the period straight after it
- SHAP explainability: which features drive the model overall, and which drove
  one specific prediction
- Checking whether `has_identity` behaves as Section 2.3 predicts, as a test of
  whether the analysis was right
- Choosing one model and registering it as the candidate for Step 5

---

*End of Step 3. `PROJECT_STATE.md` follows as a separate document.*
