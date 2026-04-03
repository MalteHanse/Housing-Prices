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

def evaulate_groups(data, labels):
    data["cluster"] = labels
    cunclusion_data = data.groupby("cluster").agg({
        "recency": "mean",
        "frequency": "mean",
        "value": "mean"
    }).round(2)
    return cunclusion_data

def visualize_boxplots(data, labels, show=True):
    fig, axes = plt.subplots(1, len(data.columns) - 1, figsize=(12, 5))
    data["cluster"] = labels
    
    feature_cols = [col for col in data.columns if col != "cluster"]
    for ax, col in zip(axes, feature_cols):
        positions = []
        for cluster in sorted(np.unique(labels)):
            df = data[data["cluster"] == cluster]
            bp = ax.boxplot(df[col], positions=[cluster], widths=0.6)
            positions.append(cluster)
        ax.set_xticks(positions)
        ax.set_xticklabels(positions)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(col)
    
    fig.tight_layout()
    fig.savefig("figures/boxplots.png", dpi=300)

    if show:
        plt.show()





