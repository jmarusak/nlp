from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering

from reader import CorpusReader
from normalizer import TextNormalizer
from vectorizer import OneHotVectorizer
from clusterer import HierarchicalClusterer

N_CLUSTERS = 20

if __name__ == '__main__':
    
    '''
    reader = CorpusReader()
    normalizer = TextNormalizer()
    docs = normalizer.fit_transform(reader.docs())
           
    vectorizer = OneHotVectorizer()
    docs = vectorizer.fit_transform(docs)
    
    clusterer = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    clusterer.fit_predict(docs)
    clusters = clusterer.labels_
    
    wikipediaids = reader.ids()
    for idx in range(10):
        print("Plot with WikipediaID '{}' assigned to cluster {}.".format(wikipediaids[idx],clusters[idx]))
    '''
    
    model = Pipeline([
        ('normalizer', TextNormalizer()),
        ('vectorizer', OneHotVectorizer()),
        ('clusterer', HierarchicalClusterer())
    ]) 
    
    reader = CorpusReader()
    clusters = model.fit_transform(reader.docs())
    
    wikipediaids = reader.ids()
    for idx in range(10):
        print("Plot with WikipediaID '{}' assigned to cluster {}.".format(wikipediaids[idx],clusters[idx]))
    