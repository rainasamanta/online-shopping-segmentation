# Behavioral Segmentation of Online Shopping Sessions

An unsupervised machine learning project: careful preparation of skewed behavioral data, PCA, a comparison of four clustering methods, stability and outcome validation, and five named visit types with one business action each.

## Dataset

Built on the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) from the UCI Machine Learning Repository (Sakar and Kastro, 2018, CC BY 4.0): 12,330 browsing sessions from one online store over one year, 15.5% of which ended in a purchase. Each session comes from a different user, so the segments describe types of visit, not types of customer.

## Workflow

### 1. Data Preparation and Exploration
- Confirmed the data matches its documentation: 12,330 rows, 18 columns, no missing values, 1,908 purchases
- Kept 125 exact duplicate rows, which are identical one-page bounces most likely from different users
- Investigated odd records: zero-second pages, sessions over 8 hours, and both rates capped at exactly 0.2
- Plotted distributions and correlations of the eight behavioral columns
### 2. Feature Engineering and PCA
- Created total pages, total duration, average seconds per page, product page share and product time share
- Dropped the two totals from clustering, as they were near-copies of the product columns (correlation 0.99)
- Applied log1p to page counts and durations, then StandardScaler to the 11 clustering features
- Reduced the features to 4 PCA components holding 91% of the variance: engagement, product focus, informational reading and pace
### 3. Choosing Parameters
- Compared inertia, silhouette, Davies-Bouldin and Calinski-Harabasz for k = 2 to 10
- Chose GMM components with BIC, after fixing a collapse caused by 700 identical bounce sessions (reg_covar = 0.01)
- Compared Ward, complete, average and single linkage on a 3,000-session sample
- Chose DBSCAN's eps from a k-distance plot and checked a small eps / min_samples grid with the noise share
### 4. Clustering and Comparison
- Fitted K-Means and GMM on all sessions, Ward on the sample and DBSCAN on the PCA components
- Compared all four methods on the same 3,000 sessions with the Adjusted Rand Index (ARI)
### 5. Validation and Profiling
- Tested K-Means stability on 50 bootstrap samples
- Compared conversion rates across segments, with a chi-square test and Cramér's V
- Profiled each segment with a standardized heatmap and its mix of visitor type, month, weekend and traffic source
- Named the segments only after profiling, and checked which ones the other methods confirm

## Results

| Method | Clusters | Agreement with K-Means (ARI) |
|---|---|---|
| K-Means (main model) | 5 | – |
| Gaussian Mixture Model | 5 | 0.75 |
| Agglomerative (Ward) | 5 | 0.76 |
| DBSCAN | 5 + 4.1% noise | 0.37 |

**Main model (K-Means, k = 5):** silhouette 0.456, stable across 50 bootstrap samples (ARI 0.979 to 0.997), with a real but moderate link between segment and purchase (Cramér's V 0.20).

| Segment | Share of Sessions | Conversion | Confirmed by Other Methods | Action to Test |
|---|---|---|---|---|
| Thorough researchers | 16.6% | 24.8% | All three | Show shipping and returns details on product pages |
| Account + product browsers | 28.7% | 21.8% | GMM and Ward | Basket and checkout reminders rather than discounts |
| Short account-focused visits | 7.4% | 11.3% | Ward only | Review the login and registration flow |
| Product-only browsers | 40.8% | 10.4% | GMM and Ward | On-site recommendations or saved-basket prompts |
| Instant bounces | 6.5% | 0.4% | All three | Audit landing pages for speed and relevance |

Conversion ranges from 0.4% to 24.8% against 15.5% overall, even though purchase data was never used to form the segments.

## Evaluation Metrics

- Inertia, silhouette, Davies-Bouldin and Calinski-Harabasz (choosing k)
- BIC (choosing GMM components)
- Noise share and silhouette on non-noise points (DBSCAN)
- Adjusted Rand Index (stability and agreement between methods)
- Conversion rate per segment, chi-square test and Cramér's V

## Limitations

- **One store, one year:** segments may not generalize to other sites or seasons, and January and April are missing.
- **Visit types, not customers:** each session is a different user, so repeat behavior and lifetime value cannot be studied.
- **A continuum, not clumps:** sessions form one continuous cloud, so segment boundaries are partly arbitrary. DBSCAN sees the two largest segments as one group.
- **Partial stability testing:** only K-Means was stability-tested, and Ward ran on a 3,000-session sample.
- **Not causal:** a segment that converts more does not show that changing other visitors' behavior would raise sales. Actions should be A/B tested.
- **Leakage avoided:** `PageValues` was excluded because it is built from completed purchases.

## Tools

Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn · Jupyter
