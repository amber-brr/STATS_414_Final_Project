import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import chi2_contingency
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score


label_col = "label"  


REAL_DF_PATH='synthdata/realdf.csv'
CTGAN_DF_PATH='synthdata/ctgan_df.csv'
DPCTGAN_1_5_DF_PATH='synthdata/dpctgan1_5_df.csv'
DPCTGAN_3_DF_PATH='synthdata/dpctgan3_df.csv'
GREAT_DF_PATH='synthdata/greatdf.csv'
RTF_DF_PATH='synthdata/rtfdf.csv'

dataset_paths = {
    'Real': REAL_DF_PATH,
    'CTGAN': CTGAN_DF_PATH,
    'DP-CTGAN (ε=1.5)': DPCTGAN_1_5_DF_PATH,
    'DP-CTGAN (ε=3)': DPCTGAN_3_DF_PATH,
    'GReaT': GREAT_DF_PATH,
    'REaLTabFormer': RTF_DF_PATH,
}
# load data
dfs = {name: pd.read_csv(path) for name, path in dataset_paths.items()}

common_cols = set.intersection(*(set(df.columns) for df in dfs.values()))

if label_col not in common_cols:
    raise ValueError(f"{label_col} not present in all datasets.")
common_cols = [label_col] + sorted([c for c in common_cols if c != label_col])

dfs = {name: df[common_cols].copy() for name, df in dfs.items()}



# separate numeric/categorical
def get_feature_types(df, label_col):
    feat_df = df.drop(columns=[label_col])
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feat_df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

numeric_cols, categorical_cols = get_feature_types(dfs["Real"], label_col)

print("Numeric features:", numeric_cols)
print("Categorical features:", categorical_cols)


def total_variation_distance(p, q):
    """
    p, q: probability vectors, aligned.
    TVD = 0.5 * sum |p - q|
    """
    return 0.5 * np.abs(p - q).sum()

def tvd_numeric_feature(real_series, other_series, n_bins=10):
    real_vals = real_series.dropna()
    other_vals = other_series.dropna()

    if real_vals.nunique() == 0:
        return np.nan

    bins = np.histogram_bin_edges(real_vals, bins=n_bins)
    hist_real, _ = np.histogram(real_vals, bins=bins)
    hist_other, _ = np.histogram(other_vals, bins=bins)

    # convert to probabilities
    if hist_real.sum() == 0 or hist_other.sum() == 0:
        return np.nan

    p = hist_real / hist_real.sum()
    q = hist_other / hist_other.sum()

    return total_variation_distance(p, q)

def tvd_categorical_feature(real_series, other_series):
    real_probs = real_series.value_counts(normalize=True)
    other_probs = other_series.value_counts(normalize=True)

    # align categories
    all_cats = real_probs.index.union(other_probs.index)
    p = real_probs.reindex(all_cats, fill_value=0).values
    q = other_probs.reindex(all_cats, fill_value=0).values

    return total_variation_distance(p, q)

def compute_tvd_all_features(dfs, label_col, numeric_cols, categorical_cols):
    """
    Compares each synthetic dataset to 'real' using TVD per feature.
    Returns a DataFrame with rows (feature, dataset, tvd).
    """
    real_df = dfs["Real"]
    rows = []

    for feature in numeric_cols + categorical_cols:
        real_series = real_df[feature]
        for name, df in dfs.items():
            if name == "Real":
                continue
            other_series = df[feature]

            if feature in numeric_cols:
                tvd = tvd_numeric_feature(real_series, other_series)
            else:
                tvd = tvd_categorical_feature(real_series, other_series)
            rows.append({
                "Feature": feature,
                "Dataset": name,
                "tvd_vs_real": tvd
            })

    tvd_df = pd.DataFrame(rows)
    return tvd_df

tvd_df = compute_tvd_all_features(dfs, label_col, numeric_cols, categorical_cols)
print("\nTotal Variation Distance (vs REAL) per feature:")
print(tvd_df.head())

# heatmap of TVD (features x datasets)
tvd_pivot = tvd_df.pivot(index="Feature", columns="Dataset", values="tvd_vs_real")
plt.figure(figsize=(10, max(4, len(tvd_pivot) * 0.3)))
sns.heatmap(tvd_pivot, annot=False, cmap="viridis")
plt.title("TVD of Feature Marginals (Synthetic vs Real)")
plt.tight_layout()
plt.show()


# histograms for 5 most important features
# 5 most important features
five_features = ['u_refreshTimes', 'article_ctr', 'slot_freq', 'user_total_views', 'device_size']

print("using features for histogram comparison:", five_features)


def plot_histograms_comparison(dfs, features):
    n_datasets = len(dfs)

    for feature in features:
        fig, axes = plt.subplots(
            1, n_datasets,
            figsize=(3.5 * n_datasets, 3),
            sharey=True  
        )

        if n_datasets == 1:
            axes = [axes]

        for ax, (name, df) in zip(axes, dfs.items()):
            sns.histplot(
                df[feature].dropna(),
                bins=20,
                color="green",
                edgecolor="black",
                ax=ax
            )
            ax.set_title(name, fontsize=12)
            ax.set_xlabel(feature, fontsize=11)

            if ax != axes[0]:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)  
            else:
                ax.set_ylabel("Count", fontsize=11)

            ax.margins(x=0.05)
            ax.grid(False)

        plt.tight_layout()
        plt.show()


plot_histograms_comparison(dfs, five_features)



def fit_propensity_model(real_df, synth_df, numeric_cols, categorical_cols,
                         label_col="label", random_state=42):

    common_feats = [c for c in real_df.columns if c in synth_df.columns and c != label_col]

    real_feats = real_df[common_feats].copy()
    synth_feats = synth_df[common_feats].copy()

    X = pd.concat([real_feats, synth_feats], axis=0, ignore_index=True)
    y = np.concatenate([
        np.ones(len(real_feats)),    
        np.zeros(len(synth_feats))  
    ])

    all_numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    all_categorical = [c for c in X.columns if c not in all_numeric]

    # preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), all_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), all_categorical)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # train/test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=0.3,
        stratify=y,
        random_state=random_state
    )

    # compute class imbalance weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    # propensity scores
    prop_scores = xgb_model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, prop_scores)
    ap = average_precision_score(y_test, prop_scores)

    return {
        "model": xgb_model,
        "preprocessor": preprocessor,
        "X_test": X_test,
        "y_test": y_test,
        "propensity_scores": prop_scores,
        "precision": precision,
        "recall": recall,
        "average_precision": ap,
    }


# fit for each synthetic dataset
propensity_results = {}
real_df = dfs["Real"]

for name, df in dfs.items():
    if name == "Real":
        continue
    print(f"\nFitting XGBoost propensity model: REAL vs {name}")
    res = fit_propensity_model(real_df, df, numeric_cols, categorical_cols, label_col=label_col)
    propensity_results[name] = res
    print(f"{name}: Average Precision (AP) = {res['average_precision']:.3f}")


# propensity score distribution plot
def plot_propensity_distributions(propensity_results):
    plt.figure(figsize=(8, 5))
    for name, res in propensity_results.items():
        sns.kdeplot(res["propensity_scores"],
                    label=f"{name} (AP={res['average_precision']:.2f})",
                    )
    plt.xlabel("Propensity score: P(sample is REAL | X)")
    plt.ylabel("Density")
    plt.title("Propensity Score Distributions (XGBoost)")
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

plot_propensity_distributions(propensity_results)



