import pandas as pd
import datetime

def load_data(file="data/retail.csv"):
    data = pd.read_csv(file)
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], format="%d/%m/%Y %H.%M")
    return data