import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.preprocessing import preprcoessing
import src.evaluation as eval
from sklearn.preprocessing import PowerTransformer

data = load_data()
data = data.sample(10000, random_state=50)
data = preprcoessing(data)
recency_freq = data.drop("value", axis=1)
value = data["value"]

def signed_log(x):
    return np.sign(x) * np.log(np.abs(x))  # signed log since some values are negative

transformer = PowerTransformer()
transformed_recency_freq = transformer.fit_transform(recency_freq)
value_trans = signed_log(value)
data = pd.DataFrame({
    "recency": transformed_recency_freq[:, 0],
    "frequency": transformed_recency_freq[:, 1],
    "value": value_trans
})


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(data.columns):
    axes[i].hist(data[col], density=True, bins=20, edgecolor='k', alpha=0.5)  

plt.show()

# colclusion: recency and frequency show good distribution with the yeo-johnson transform (powertransfomer) and
# the value shows the best gaussian when it is log transformed

    