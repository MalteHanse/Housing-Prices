import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def load_train():
    data = pd.read_csv("../Data/train.csv", skiprows=0)
    ID = data["Id"]
    data = data.drop("Id", axis=1)
    return data

def load_test():
    features = pd.read_csv("../Data/test.csv")
    labels = pd.read_csv("../Data/sample_submission.csv")
    ID = features["Id"]
    y = labels.drop("Id", axis=1).to_numpy()
    features = features.drop("Id", axis=1)
    return ID, features, y

def process_data(data, nan_threshold=0.9):
    data = data.loc[:, data.isna().mean() < nan_threshold]  # drop if missing is over 90%
    columns_kept = data.drop("SalePrice", axis=1).columns
    return data, columns_kept

if __name__ == "__main__":
    data_train = load_train()
    data_train, columns_kept = process_data(data_train)

    X_train = data_train.drop("SalePrice", axis=1)
    y_train = data_train["SalePrice"]

    numeric_cols = X_train.select_dtypes(include=["number"]).columns
    categorial_cols = X_train.select_dtypes(include=["object"]).columns

    # build the pipeline
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorial_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorial_transformer, categorial_cols)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("LinearModel", LinearRegression())
    ])
    X_train_processed = model.named_steps["preprocessor"].fit_transform(X_train)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names
        )

    print(X_train)
    print(X_train_processed)

    model.fit(X_train, y_train)

    ID, X_test, y_test = load_test()
    X_test = X_test[columns_kept]
    test_score = model.score(X_test, y_test)
    print("Score on test set: ", test_score, end=" ")

