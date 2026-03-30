import numpy as np
from src.data_loader import load_train
from src.models import build_model
from src.preprocessing import build_preprocessor, process_data
from src.evaluation import split_data, cross_val, get_rmse

MODEL_TYPE = "XGBoost"  # "Linear": LinearRegression, "Ridge":RidgeRegression, "RF":RandomForrest, "XGBoost":GradientBoosting
USE_SCALING = True
REMOVE_UNINFORMATIVE = True

X = load_train()

if REMOVE_UNINFORMATIVE:
    X, _ = process_data(X)  # remove data with high amount of missing mvalues

y = X["SalePrice"].to_numpy()
X = X.drop("SalePrice", axis=1)

# split the data into train and test sets
X_train, X_test, y_train, y_test = split_data(X, y)

# build the preprocessor
preprocessor = build_preprocessor(X_train, use_scaling=USE_SCALING)

# build and fit the model
model = build_model(preprocessor, model_type=MODEL_TYPE)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = get_rmse(y_test, y_pred)
print(f"RMSE estimation: {rmse:.3f}")

# evaluate the model
score = model.score(X_test, y_test)
print(f"Score on test set: {score:.3f}")

# cross validation
cross_val_score = cross_val(model, X_train, y_train)
