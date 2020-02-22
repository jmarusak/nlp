from sklearn.feature_extraction.text import TfidfVectorizer

from reader import PickledCorpusReader
from normalizer import TextNormalizer

def identity(words):
    return words


corpus = PickledCorpusReader('../../corpora/Pickled_Corpus_Sample')

normalizer = TextNormalizer()
docs = normalizer.fit_transform(corpus.docs())

vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)

vectors = vectorizer.fit_transform(docs)

print(vectors.shape)