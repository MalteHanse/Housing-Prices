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

def visualize_boxplots(data, labels, preprocessor=None, clip_percentile=(5, 95), show_fliers=False, show=True):
    feature_cols = ["recency", "frequency", "value"]

    # apply preprocessing 
    if preprocessor is not None:
        data_transformed = preprocessor.fit_transform(data[feature_cols])
        data_plot = data.copy()
        for i, col in enumerate(feature_cols):
            data_plot[col] = data_transformed[:, i]
    else:
        data_plot = data.copy()

    # clip outliers 
    if clip_percentile is not None:
        for col in feature_cols:
            lower = np.percentile(data_plot[col], clip_percentile[0])
            upper = np.percentile(data_plot[col], clip_percentile[1])
            data_plot[col] = data_plot[col].clip(lower, upper)

    fig, axes = plt.subplots(1, len(feature_cols), figsize=(12, 5))
    data_plot["cluster"] = labels

    for ax, col in zip(axes, feature_cols):
        positions = []
        means = []
        labels = sorted(np.unique(labels))
        for cluster in labels:
            df = data_plot[data_plot["cluster"] == cluster]
            bp = ax.boxplot(df[col], positions=[cluster], widths=0.6, showfliers=show_fliers)
            mean = np.mean(df[col])
            means.append(mean)
            positions.append(cluster)
        ax.plot(labels, means, marker='^', label="Mean")
        ax.set_xticks(positions)
        ax.set_xticklabels(positions)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(col + (" (normalized)" if preprocessor else ""))
        ax.legend()

    fig.tight_layout()
    fig.savefig("figures/boxplots.png", dpi=300)

    if show:
        plt.show()





