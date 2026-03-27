import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def load_train():
    data = pd.read_csv("../Data/train.csv", skiprows=0)  
    features = data.drop(["SalePrice", "Id"], axis=1)
    labels = data["SalePrice"]
    ID = data["Id"]
    return features, labels

def load_test():
    features = pd.read_csv("../Data/test.csv")
    labels = pd.read_csv("../Data/sample_submission.csv")
    ID = features["Id"]
    features = features.drop("Id", axis=1)
    return ID, features, labels

if __name__ == "__main__":
    X_train, y_train = load_train()
    encoder = OneHotEncoder(sparse_output=False)
    X_train = encoder.fit_transform(X_train)
    print(X_train)

    model = LinearRegression()
    model.fit(X_train, y_train)

    ID, X_test, y_test = load_test()
    print(X_test)
    X_test = encoder.transform(X_test)
    model.predict(X_test)
    

