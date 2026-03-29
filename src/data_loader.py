import pandas as pd

def load_train(path="data/train.csv"):
    data = pd.read_csv(path)
    ID = data["Id"]
    data = data.drop("Id", axis=1)
    return data

def load_test(features_path="data/test.csv"):
    features = pd.read_csv(features_path)
    ID = features["Id"]
    features = features.drop("Id", axis=1)
    return ID, features
