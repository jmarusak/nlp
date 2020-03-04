from sklearn.cluster import AgglomerativeClustering

from reader import CorpusReader
from normalizer import TextNormalizer
from vectorizer import OneHotVectorizer

N_CLUSTERS = 3

if __name__ == '__main__':

    reader = CorpusReader()
    normalizer = TextNormalizer()
    docs = normalizer.fit_transform(reader.docs([23890098, 31186339, 24225279, 20532852]))
        
    vectorizer = OneHotVectorizer()
    docs = vectorizer.fit_transform(docs)
    
    clusterer = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    clusterer.fit_predict(docs)
    labels = clusterer.labels_
    
    print(labels)