from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering

N_CLUSTERS=20

class HierarchicalClusterer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.clusterer = AgglomerativeClustering(n_clusters=N_CLUSTERS)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        clusters = self.clusterer.fit_predict(documents)
        return clusters