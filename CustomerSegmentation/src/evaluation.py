import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def visualize_data(X, bins=20, show=True):
    # df = X.drop("CustomerID", axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for col, ax in zip(X.columns, axes):
        ax.hist(X[col], bins=bins, density=True, alpha=0.5, edgecolor='k')
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

    fig.savefig("figures/data.png", dpi=300)
    if show:
        fig.tight_layout()
        plt.show()


def visualize_clusters(ax, data, labels, show=True):
    # reduce dim with pca for visualization
    # pca = PCA(2)
    # data_reduced = pca.fit_transform(data)

    ax.scatter(data["recency"], data["frequency"], data["value"], c=labels, s=3, alpha=0.3)
    ax.set_xlabel("$PCA_1$")
    ax.set_ylabel("$PCA_2$")
    
    if show:
        plt.show()

