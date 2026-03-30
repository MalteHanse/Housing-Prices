import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_train
from src.models import build_model
from src.preprocessing import build_preprocessor, process_data
import src.evaluation as eval

MODEL_TYPE = "XGBoost"  # "Linear": LinearRegression, "Ridge":RidgeRegression, "RF":RandomForrest, "XGBoost":GradientBoosting
USE_SCALING = True
REMOVE_UNINFORMATIVE = True

X = load_train()

if REMOVE_UNINFORMATIVE:
    X, _ = process_data(X)  # remove data with high amount of missing mvalues

y = X["SalePrice"].to_numpy()
X = X.drop("SalePrice", axis=1)

# split the data into train and test sets
X_train, X_test, y_train, y_test = eval.split_data(X, y)

# build the preprocessor
preprocessor = build_preprocessor(X_train, use_scaling=USE_SCALING)

# build and fit the model
model = build_model(preprocessor, model_type=MODEL_TYPE)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = eval.get_rmse(y_test, y_pred)
print(f"RMSE estimation: {rmse:.3f}")

# evaluate the model
score = model.score(X_test, y_test)
print(f"Score on test set: {score:.3f}")

# cross validation
cross_val_score = eval.cross_val(model, X_train, y_train)

importances = eval.get_original_feature_importance(model, X_train)

# determine correlation  matrix
eval.plot_correlation_matrix(X_train, show=False)

# visualize feature importance
fig, ax = plt.subplots(1, 2)
eval.plot_feature_importance(importances, ax[0], show=False)
eval.plot_permutation_importance(ax[1], model, X_test, y_test, show=False)
fig.tight_layout()
fig.savefig("Figures/feature_importance.png", dpi=300)

