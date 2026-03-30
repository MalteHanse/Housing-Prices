import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.inspection import permutation_importance

def correlation_matrix(X):
    # this function assumes, that only numeric values can be correlated
    # for investigating coreleation on categorical valus a more complex function must be written
    return X.corr(numeric_only=True)

def plot_correlation_matrix(X, show=True):
    X_corr = correlation_matrix(X)
    heatmap = sns.heatmap(X_corr)
    plt.tight_layout()
    if show:
        plt.show()
    fig = heatmap.get_figure()
    fig.savefig("Figures/corr_heatmap.png", dpi=300)

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def cross_val(model, X, y, cv=5, scoring="r2"):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"CV {scoring}: mean={scores.mean():.3f}, std={scores.std():.3f}")
    return scores.mean()

def get_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

def get_feature_importance(model):
    feature_importance = model.feature_importances_
    return feature_importance

def get_original_feature_importance(model, X_train):
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_features = ohe.get_feature_names_out().tolist()

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    importances = regressor.feature_importances_

    original_importances = {}

    for i, col in enumerate(numeric_cols):
        original_importances[col] = importances[i]

    for cat_col in categorical_cols:
        mask = [f.startswith(f"{cat_col}_") for f in encoded_cat_features]
        if any(mask):
            encoded_importances = importances[len(numeric_cols):][mask]
            original_importances[cat_col] = np.sum(encoded_importances)

    return original_importances

def get_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    return permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)


def get_original_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    preprocessor = model.named_steps["preprocessor"]

    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_features = ohe.get_feature_names_out().tolist()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    all_encoded_names = numeric_cols + encoded_cat_features

    perm_result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    encoded_importances = perm_result.importances

    original_importances = {}

    for i, col in enumerate(numeric_cols):
        original_importances[col] = encoded_importances[i]

    for cat_col in categorical_cols:
        mask = [f.startswith(f"{cat_col}_") for f in encoded_cat_features]
        if any(mask):
            # Stack all one-hot encoded columns for this feature
            cat_idx = [all_encoded_names.index(f) for f in encoded_cat_features if f.startswith(f"{cat_col}_")]
            original_importances[cat_col] = np.sum(encoded_importances[cat_idx], axis=0)

    return original_importances

def plot_feature_importance(feature_importance, ax, top=10, show=True):
    sorted_importances = sorted(feature_importance.items(), key=lambda x: x[1])
    n = len(sorted_importances)
    if top > n:
        raise ValueError(f"top must satisfy top>n, but top={top} and n={n}")
    names = [pair[0] for pair in sorted_importances][n - top:]
    values = np.array([pair[1] for pair in sorted_importances])[n - top:] 
    pos = np.arange(values.shape[0]) + 0.5
    ax.barh(pos, values, align="center")
    ax.set_yticks(pos, np.array(names))
    ax.set_title("Feature Importance")
    if show:
        plt.show()

def plot_permutation_importance(ax, model, X, y, top=10, n_repeats=10, random_state=42, show=True):
    perm_importance = get_original_permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)

    # Convert dict to arrays for plotting
    names = list(perm_importance.keys())
    values = np.array(list(perm_importance.values()))
    n = len(names)
    if top > n:
        raise ValueError(f"top must satisfy top>n, but top={top} and n={n}")
    sorted_idx = values.mean(axis=1).argsort()[n - top:]
    ax.boxplot(
        values[sorted_idx].T,
        vert=False,
        labels=np.array(names)[sorted_idx]
    )
    ax.set_title("Permutation Importance")
    if show:
        plt.show()
