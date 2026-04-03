from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def build_model(preprocessor, model_type="kmeans", clusters=3):
    if model_type == "kmeans":
        model = KMeans(clusters)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
        ])
    return pipeline