from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

from reader import CorpusReader
from normalizer import TextNormalizer

class OneHotVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = CountVectorizer(binary=True)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        freqs = self.vectorizer.fit_transform(documents)
        return freqs.toarray()

if __name__ == '__main__':

    reader = CorpusReader()
    normalizer = TextNormalizer()
    docs = normalizer.fit_transform(reader.docs([23890098, 31186339]))
        
    vectorizer = OneHotVectorizer()
    docs = vectorizer.fit_transform(docs)
    
    for doc in docs:
        print()
        print(doc)