from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def process_data(data, nan_threshold=0.9):
    data = data.loc[:, data.isna().mean() < nan_threshold]  # drop if missing is over 90%
    columns_kept = data.drop("SalePrice", axis=1).columns
    return data, columns_kept

def build_preprocessor(X, use_scaling=True):
    numeric_cols = X.select_dtypes(include=["number"]).columns
    categorial_cols = X.select_dtypes(include=["object"]).columns

    # build the pipeline
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaling:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(numeric_steps)
    
    categorial_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorial_transformer, categorial_cols)
    ])
    return preprocessor