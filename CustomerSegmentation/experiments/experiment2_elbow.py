# this file finds out how many clusters are best for the kmeans
# silence warning
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.models import build_model
from src.preprocessing import preprocessing, build_preprocessor

data = load_data()
data = data.sample(10000, random_state=50)

n_clusters = np.arange(1, 10)
intertias = []
data = preprocessing(data)

for n in n_clusters:
    preprocessor = build_preprocessor(data)
    model = build_model(preprocessor, model_type="kmeans", clusters=n)
    model.fit(data)
    kmeans = model.named_steps["model"]
    # inertia is calculated as
    # inertia = \sum_{i=0}^n dist(x_i, c_j)^2
    intertias.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(n_clusters, intertias, marker="^", label="Kmeans Elbow")
ax.set_xlabel("Clusters")
ax.set_ylabel("Inertia")
ax.set_xticks(n_clusters)
ax.legend()
fig.savefig("figures/elbow.png", dpi=300)
plt.show()

# this experiment shows that 3 clusters is a optimal choice here
# one could also argue 4 or 5, but it has been observed that kmeans 
# with this amount of clusters is in consistent with the grouping
# also the cluster visulization in main.py suggets 3
