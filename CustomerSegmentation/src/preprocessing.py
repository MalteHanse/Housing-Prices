import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# the data must be preprocessed into the customer specific features
# here we use Recency (how long ago was the last purchase),
# frequency (how often does the customer buy), and
# value (for how much money does the customer buy)

def preprocessing(X):
    # total value per purchase
    X["Value"] = X["Quantity"] * X["UnitPrice"]

    # recency
    X["InvoiceDate"] = X["InvoiceDate"].dt.date
    reference_date = max(X["InvoiceDate"]) + datetime.timedelta(days=1)  # the latest purchase +1d is the reference to dertmine recency value
    X = X.groupby("CustomerID")
    X = X.agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,  # how recent
        "InvoiceNo": "count",                                      # how often
        "Value": "sum"                                             # how much
    })
    X.rename(columns = {
        "InvoiceDate": "recency",
        "InvoiceNo": "frequency",
        "Value": "value"}, inplace=True)
    return X

def build_preprocessor(X):
    # as the data visualization shows, is there not much variance in the data, thus a algorithm would have a hard
    # time sepeerating the data into groups. Thus we must see if we can apply a transform that gives more sepeartion in the
    # data
    col_transformer = ColumnTransformer([
        ("freq_recency", PowerTransformer(method='yeo-johnson'), ["recency", "frequency"]),
        ("value", FunctionTransformer(lambda x: np.sign(x) * np.log1p(np.abs(x))), ["value"])  # np.log1p(x) = np.log(x + 1)
    ], remainder='passthrough')

    preprocessor = Pipeline([
        ("transformer", col_transformer),
        ("scaler", StandardScaler())
    ])
    return preprocessor