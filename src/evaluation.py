import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error

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

def plot_feature_importance(feature_importance, ax, show=True):
    sorted_importances = sorted(feature_importance.items(), key=lambda x: x[1])
    names = [pair[0] for pair in sorted_importances]
    values = np.array([pair[1] for pair in sorted_importances])
    pos = np.arange(values.shape[0]) + 0.5
    ax.barh(pos, values, align="center")
    ax.set_yticks(pos, np.array(names))
    ax.set_title("Feature Importance")
    if show:
        plt.show()
