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