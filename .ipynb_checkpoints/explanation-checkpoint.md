# Mercedes-Benz EDA Notebook — Full Walkthrough

This document explains every section and cell group in the notebook, what it's doing, why it matters, and how the results connect to the downstream goal of SHAP-based managerial insights.

---

## Section 1: Setup & Data Loading

### Imports Cell
Loads the core libraries used throughout the notebook:

- **pandas/numpy**: Data manipulation. Pandas DataFrames are the primary data structure; numpy handles numeric operations.
- **matplotlib/seaborn**: Visualization. Seaborn sits on top of matplotlib and makes statistical plots easier.
- **sklearn (preprocessing, decomposition, cluster, feature_selection)**: Machine learning utilities. `StandardScaler` normalizes data before PCA. `PCA` reduces dimensions. `KMeans` finds clusters. `VarianceThreshold` identifies features with too little variation to be useful.
- **scipy.stats**: Statistical tests like ANOVA and correlation significance tests.

The settings at the bottom (`pd.set_option`, `sns.set_style`, `plt.rcParams`) just control display defaults — wider tables, clean plot styling, consistent figure sizes.

### Data Loading Cell
Reads `train.csv` (4,209 rows × 378 columns) and `test.csv` (4,209 × 377). Test has one fewer column because it doesn't include `y` (the target). Both datasets have the same structure otherwise. The row count being identical for train and test is just how this particular Kaggle competition was structured — it doesn't always work that way.

---

## Section 2: High-Level Data Overview

### `train.head(10)`
Displays the first 10 rows so you can visually inspect the data. Key observations: the first column is `ID` (just a row identifier, not useful for modeling), followed by `y` (the target), then 8 categorical columns (`X0` through `X8`, containing letters like `k`, `v`, `at`), and then hundreds of binary columns (`X10` onward) containing only 0s and 1s.

### `train.info()`
Shows column types and memory usage. The output confirms: 1 float column (`y`), 369 integer columns (binary features + `ID`), and 8 string/object columns (the categoricals). Importantly, it shows **no null values** in any column, which means we don't need to handle missing data — one less preprocessing step.

### Column Type Identification Cell
Programmatically separates columns into three groups:

- **cat_cols** (8): The categorical string columns (`X0`–`X8`). These represent different car configuration options — things like model variant, engine type, interior package, etc. (anonymized, so we don't know exactly what).
- **binary_cols** (368): Numeric columns with only 2 unique values (0 and 1). These likely represent whether a specific manufacturing process or quality check applies to a given car.
- **num_cols** (370): All numeric columns including `ID` and `y`.

This separation matters because categoricals and binary features require different analysis approaches and different preprocessing.

### Missing Values Check
Confirms zero missing values across all columns. This is unusual for real-world data and simplifies preprocessing. In most projects you'd need to decide on imputation strategies here.

### Descriptive Statistics
Shows summary stats for `y` and the first 10 binary columns. The key takeaway is how sparse the binary features are — most have means very close to 0, meaning the process is only active for a small fraction of cars. For instance, `X10` has a mean of 0.013 (active for only 1.3% of cars), while `X14` has a mean of 0.428 (active for ~43%). This sparsity is important because features that are almost always 0 carry very little information.

---

## Section 3: Target Variable (`y`) Analysis

This section is about understanding what we're trying to predict before building any model. The distribution of the target variable affects model choice, evaluation metrics, and whether transformations are needed.

### Descriptive Stats + Skewness/Kurtosis
- **Mean 100.67s, median 99.15s**: The mean being higher than the median tells you the distribution is right-skewed (pulled by high values).
- **Range 72.11 to 265.32**: A wide range. Most cars take 72–130 seconds, but some take much longer.
- **Skewness 1.207**: Positive skew — a longer right tail. Values above 1 suggest a log transform might help normalize the distribution for linear models.
- **Kurtosis 7.911**: Very high (normal distribution is 3). This means heavy tails — more extreme values than you'd expect. A few cars take much longer than the bulk.

### Distribution Plots (Histogram, Boxplot, Q-Q Plot)
Three complementary views of the same distribution:

- **Histogram**: Shows the shape directly. You should see a roughly bell-shaped peak around 95–105s with a long right tail.
- **Boxplot**: Shows the median (line), IQR (box), and outliers (dots beyond whiskers). The dots on the right side are the high-y outliers.
- **Q-Q Plot**: Compares the actual distribution against a theoretical normal distribution. Points on the diagonal = normal. Where they curve upward at the right end = heavy right tail. This tells you a normal distribution assumption (used by many linear models) doesn't quite hold.

### Outlier Detection (IQR Method)
The IQR (Interquartile Range) method defines outliers as anything below Q1 − 1.5×IQR or above Q3 + 1.5×IQR.

- Q1 = 90.82, Q3 = 109.01, so IQR = 18.19
- Lower bound: 63.53 (no values below this)
- Upper bound: 136.30 (50 rows above this, or 1.2%)

The 50 "outlier" rows aren't necessarily errors — they're cars with legitimately complex configurations that take longer to test. The single extreme value at 265.32 is the real outlier worth removing, while keeping the other 49 preserves meaningful signal about what makes testing slow.

---

## Section 4: Categorical Feature Exploration

These 8 features are the most interpretable part of the dataset. They represent high-level configuration choices that management can actually act on (e.g., "change how we handle configuration X0=ak").

### Unseen Test Levels Cell
Checks whether the test set contains categorical values that never appear in training. For example, X0 has 6 levels in test that don't exist in train (`p`, `ag`, `bb`, `av`, `an`, `ae`). This matters because a model trained on the training data has never learned anything about these levels. The "other" bucket fix discussed earlier handles this.

### Value Counts Bar Charts
Shows how many cars belong to each categorical level. The key insight is that distributions are highly imbalanced — for X0, a few levels like `az` and `t` might have hundreds of cars while others have 1–3. Levels with very few observations give unreliable statistics, so you should focus managerial insights on the well-represented levels.

### Mean y by Categorical Level
For each categorical feature, this groups rows by their level and computes the mean, median, standard deviation, and count of `y`. This is the most important table in the section because it directly answers: **which configurations take longest?**

For X0:
- `aa` averages 134.4s but only has 1 observation (unreliable)
- `ak` averages 111.9s with 342 observations (reliable and high — a target for efficiency improvement)
- `j` averages 111.5s with 178 observations (another high-time configuration)

For X2:
- `s` averages 115.9s with 90 observations (the most reliably slow X2 level)

These are the configurations that SHAP analysis should confirm as important, and they're where management should focus efficiency efforts.

### Boxplots by Categorical Level
Visual version of the table above, showing the full distribution of `y` within each level rather than just the mean. This reveals whether a high mean is driven by a consistent pattern (tight box) or a few extreme values (wide box with outliers). Tight, high boxes are the most actionable findings — they represent configurations that reliably take a long time.

### ANOVA Tests
ANOVA (Analysis of Variance) tests whether the mean `y` differs significantly across levels of each categorical. The null hypothesis is "all levels have the same mean y."

- **X0: F=166.44, p≈0**: Overwhelmingly significant. Different X0 configurations lead to very different testing times. This is your most important categorical.
- **X2: F=35.82, p≈0**: Also very significant, second most important.
- **X3: F=37.53**: Significant despite fewer levels — the 7 levels of X3 have meaningfully different means.
- **X4: F=3.36, p=0.018**: Barely significant. Only 4 levels, and they don't differ much. Low priority for insights.
- **X5: F=2.01, p=0.001**: Significant overall, but with 29 levels and small effect sizes, individual level differences are small.

The F-statistic magnitude tells you how strong the effect is, and the p-value tells you if it's statistically real. For managerial purposes, focus on X0, X2, X3, and X6.

---

## Section 5: Binary Feature Analysis

With 368 binary features, this section separates signal from noise.

### Activation Rate Bar Chart
Shows the proportion of 1s (active) for each binary feature. Most features are very sparse (close to 0), meaning the process only applies to a small subset of cars. Features that are always 0 or always 1 carry zero information — they're the same for every car, so they can't explain differences in testing time.

Result: 13 features are always 0 (constant). None are always 1.

### Removing Constant Columns
These 13 features (like `X11`, `X93`, `X233`, etc.) are identical for every row. They literally cannot contribute to a prediction since there's no variation to learn from. Dropping them is unambiguous — there's no downside.

### Low-Variance Filter (VarianceThreshold)
This goes a step further than removing constants. A feature with 1% activation rate (only ~42 out of 4,209 rows have a 1) has a variance of only 0.0099. With so few positive examples, any apparent correlation with `y` is likely noise — you can't reliably estimate the effect of a process that only applies to 42 cars.

The threshold of 0.01 identified 135 additional low-variance columns. These are candidates for removal to reduce noise, though some judgment is needed — a feature with 1% activation but a huge effect on `y` could still be meaningful. In practice, with this dataset, most of these 135 are just noise.

### Point-Biserial Correlation
This is the right correlation measure for binary-vs-continuous variable pairs. For each binary feature, it computes the correlation with `y` and a p-value testing significance.

Results (top 20):
- **X314 (r=0.649)**: When this process is active, testing time is substantially higher. This is the single most predictive binary feature.
- **X261 (r=0.628)**: Similar strong positive effect.
- **X127 (r=−0.540)**: Negative correlation — when this process is active, testing is *shorter*. This is valuable for management: activating X127 (whatever it represents) is associated with faster testing.
- **X232, X279, X263, X29 (|r|≈0.41)**: A cluster of moderately correlated features, possibly related to each other.

The absolute correlation `abs_r` is what matters for feature importance — the sign just tells you the direction.

### Top Binary Features Boxplots
Visual confirmation of the correlations. Each subplot shows `y` when the feature is 0 vs 1. For X314, the boxplot for "1" should be noticeably higher than for "0" — confirming that cars with X314 active take longer to test.

### Process Burden (n_active_binary)
This is the first engineered feature: simply counting how many binary processes are active for each car. The idea is intuitive — more processes = more things to test = longer testing time.

The result (r=0.124, p<0.001) confirms a weak but real positive relationship. It's statistically significant but not a strong predictor on its own. This makes sense — it's not just about *how many* processes, but *which* processes. That's why the weighted version (weighting by each feature's correlation with `y`) is a better engineered feature.

---

## Section 6: Correlation & Multicollinearity

This section deals with a critical problem: many binary features encode the same or overlapping information. If X88 and X122 are always active together (both 0 or both 1 for every car), they're effectively the same feature. Including both confuses models and splits SHAP importance between them.

### Pairwise Correlation Among Binary Features
Computes the correlation between every pair of binary features. With ~355 features, that's ~63,000 pairs. Pairs with |r| > 0.9 are flagged as problematic.

Result: **221 highly correlated pairs**. This is severe — it means large groups of features are near-duplicates of each other. This is typical in manufacturing data where processes are bundled together (e.g., if you have feature A, you always have features B and C too).

### Identifying Features to Drop
For each pair with |r| ≥ 0.98 (near-perfect correlation), the code keeps the one more correlated with `y` and drops the other. Result: **71 columns flagged for removal**.

The logic is: if two features contain the same information, keep the one that's more useful for predicting the target. This is a conservative approach — you could also drop at a lower threshold (say 0.9) to be more aggressive about deduplication.

### Correlation Heatmap
Visualizes correlations among the top 25 features (by correlation with `y`) plus `y` itself. Look for blocks of red/blue — these indicate groups of features that move together. These groups likely represent process bundles or configuration packages.

---

## Section 7: Feature Interactions & Clustering

This section looks for higher-order structure in the data — patterns that aren't visible from individual features alone.

### 7a: PCA (Principal Component Analysis)
PCA takes the hundreds of binary features and finds the "axes" along which the data varies most. The first principal component (PC1) is the direction of maximum variance, PC2 is the next most variable direction perpendicular to PC1, and so on.

**Key finding: PC1 alone explains 80% of variance.** This is remarkable — it means the vast majority of variation in the binary features collapses onto a single dimension. In practical terms, there's one dominant axis that separates cars into "types," and everything else is secondary. This dominant axis likely corresponds closely to X0 (the strongest categorical), since X0 also captures the biggest configuration differences.

The cumulative variance plot shows how many components you'd need to capture a given amount of information. 80% from 1 component means PCA is very effective here as a compression tool.

### PCA Scatter Colored by y
Plots each car as a point in PC1–PC2 space, colored by testing time. If you see a color gradient (e.g., blue on the left, yellow on the right), it means the PCA dimensions capture testing time differences — which confirms the binary features collectively encode meaningful information about `y`. Scattered colors would mean the binary features don't explain `y` well.

### 7b: KMeans Clustering
KMeans groups cars into k clusters based on similarity of their binary feature patterns. The goal is to find natural "configuration families" — groups of cars that go through similar sets of processes.

**Elbow plot**: Shows inertia (within-cluster sum of squared distances) for different values of k. You want the "elbow" — the point where adding more clusters stops giving meaningful improvement. A smooth curve with no clear elbow means the data doesn't have strongly separated groups.

**Cluster assignment**: With k=15 (which is likely too many), cluster sizes range from 20 to 670. The very small clusters (20–30 cars) may represent rare or unique configurations. Reducing to k=5–7 would give more stable, interpretable clusters.

### Cluster vs y Boxplot
Shows the distribution of `y` within each cluster. If clusters have noticeably different median testing times, it means the clustering captured meaningful configuration differences. This is valuable for management: "Cars in Cluster 3 take an average of 115 seconds, while Cluster 0 cars take 92 seconds."

### Cluster Profiles Heatmap
For each cluster, shows the average activation rate of the most differentiating binary features. This tells you *what makes each cluster different*. For example, if Cluster 3 has high activation of X314 and X261 (both positively correlated with `y`), that explains why Cluster 3 has longer testing times. This directly translates to a managerial insight: "Cluster 3 cars trigger processes X314 and X261, which add ~15 seconds to testing."

### 7c: Categorical × Binary Interactions
Tests whether the effect of a binary feature on `y` depends on the categorical level. For example, X314 being active might add 20 seconds for `X0=az` cars but only 5 seconds for `X0=t` cars. If so, the interaction matters more than either feature alone.

The boxplots show `y` split by both the categorical level (x-axis) and the binary feature (color). If the colored boxes are consistently separated across all x-axis groups, the binary feature has a main effect. If the separation varies by group, there's an interaction.

---

## Section 8: One-Hot Encoding & Final Preprocessing

This is the data transformation pipeline that produces the modeling-ready dataset.

### 8a: Train/Test Concatenation
Stacks train and test into one DataFrame before encoding. This is necessary because `pd.get_dummies` creates a column for each unique level it sees. If you encode train and test separately, a level that appears only in test would create a column in the test encoding that doesn't exist in train — and the model can't handle mismatched columns.

The `_is_train` flag lets us split them back apart after encoding.

### 8b: One-Hot Encoding
`pd.get_dummies` converts each categorical column into multiple binary columns — one per level. For example, X0 with 47 levels becomes 47 columns: `X0_a`, `X0_aa`, `X0_ab`, etc. Each is 1 if the car has that level, 0 otherwise.

Result: 197 new columns created, bringing the total from 379 to 568.

`drop_first=False` keeps all dummy levels. Some practitioners drop one to avoid the "dummy variable trap" (perfect multicollinearity among the dummies), but tree-based models handle this fine, and for SHAP interpretation it's cleaner to keep all levels visible.

### 8c: Dropping Cleaned Columns
Removes the constant columns (13) and near-duplicate columns (71) identified in earlier sections. The current code does NOT drop the 135 low-variance columns — that's the fix we discussed adding.

Result: 84 columns dropped, from 568 to 484.

### 8d: Engineered Features
Re-computes `n_active_binary` on the cleaned column set (since some binary columns were dropped, the count needs to be recalculated). The commented-out line would add cluster labels as a feature — this is optional and should only be done if clustering proved informative.

### 8e: Splitting Back
Separates the combined DataFrame back into train (4,159 rows after outlier removal) and test (4,209 rows) using the `_is_train` flag. Includes sanity checks to verify row counts match and no training target values are missing.

### Final Feature Matrix
Creates the actual X (features) and y (target) arrays for modeling. Drops `ID` and `y` from the feature set — `ID` is just an identifier with no predictive value, and `y` is what we're predicting.

Final dimensions: **4,159 rows × 482 features** for training.

---

## Section 9: Export

Saves the processed train and test DataFrames to CSV files. These are the inputs for the modeling notebook.

---

## How It All Connects to the End Goal

The entire notebook funnels toward one objective: producing a clean, well-understood dataset that a model can learn from, and that SHAP analysis can interpret meaningfully.

- **Sections 2–3** establish what we're predicting and its characteristics
- **Sections 4–5** identify which raw features matter most (X0, X2 among categoricals; X314, X261, X127 among binaries)
- **Section 6** removes redundancy so SHAP doesn't split importance across duplicate features
- **Section 7** discovers higher-order structure (clusters, interactions) that enrich the feature set
- **Section 8** transforms everything into a numeric matrix the model can consume

The features that survive this pipeline — and especially the engineered ones like target encoding and weighted burden — are designed to produce SHAP values that translate directly into statements like: "Configuration X0=ak adds 12 seconds to testing time. Process X314 adds 8 seconds. Eliminating process X314 from the workflow would save the most testing time per car."K