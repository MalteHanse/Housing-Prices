from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def build_model(preprocessor, model_type="kmeans"):
    if model_type == "kmeans":
        model = KMeans(3)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
        ])
    return pipeline