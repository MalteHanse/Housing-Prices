import datetime

# the data must be preprocessed into the customer specific features
# here we use Recency (how long ago was the last purchase),
# frequency (how often does the customer buy), and
# value (for how much money does the customer buy)

def preprcoessing(data):
    # total value per purchase
    data["Value"] = data["Quantity"] * data["UnitPrice"]

    # recency
    data["InvoiceDate"] = data["InvoiceDate"].dt.date
    reference_date = max(data["InvoiceDate"]) + datetime.timedelta(days=1)  # the latest purchase +1d is the reference to dertmine recency value
    data = data.groupby("CustomerID")
    data = data.agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,  # how recent
        "InvoiceNo": "count",                                      # how often
        "Value": "sum"                                             # how much
    })
    data.rename(columns = {
        "InvoiceDate": "recency",
        "InvoiceNo": "frequency",
        "Value": "value"}, inplace=True)
    return data
