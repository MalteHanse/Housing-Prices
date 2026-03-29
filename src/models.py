from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

def build_model(preprocessor, model_type="Ridge"):
    if model_type == "Ridge":
        regressor = Ridge(alpha=1.0)
    elif model_type == "Linear":
        regressor = LinearRegression()
    elif model_type == "RF":
        regressor = RandomForestRegressor()
    
    else:
        raise ValueError(f"No model named {model_type}")
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    return model